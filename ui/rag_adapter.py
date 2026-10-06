"""Adapter: the single boundary between the interface and the RAG backend.

`app.py` only knows this module and the contract (`contracts.py`).  This is
where the mock is swapped for the real engine once it exists:

    MockRAG (today)  →  src/rag/engine.py (tomorrow, RAG team)

No Chroma, LangChain, embeddings, prompts or LLM providers in the UI.

Observed real contract (PRs #25-#28 + RAGPipeline on main, pre-integration):

* the pipeline exposes ``answer_query(question) -> dict`` returning
  ``{"answer": str, "source_documents": list[Document]}``;
* documents carry ``page_content`` plus retrieval metadata (``source``,
  ``page``, ``section_label``, …); ``grounded``/``score``/``latency_ms``
  are NOT provided yet;
* ``session_id`` is bound when the backend *constructs* its retriever, not
  per query — :func:`create_backend` receives it as the integration seam.

Integration (ONLY this file is touched):

    def create_backend(session_id: str | None = None):
        from src.rag.engine import RagEngine   # RAG team
        return RagEngine(session_id=session_id)

The engine may expose ``query(question, documents)``, ``ask(...)`` or the
pipeline's ``answer_query(question)`` (sync or async): the adapter inspects
the signature and passes ``documents`` only when the method accepts them.
Whatever it returns is normalized with ``RAGResponse.from_any``, so missing
metadata, LangChain-like objects or ``None`` lists never break the
interface.
"""

from __future__ import annotations

import inspect
from typing import Any, List, Optional

try:
    from .contracts import RAGResponse
    from .mock_rag import MockRAG
except ImportError:
    from contracts import RAGResponse
    from mock_rag import MockRAG

_UNSET = object()

# Conversational history boundary (UI → RAG). The UI transports the turns of
# the CURRENT conversation only; the window keeps the payload predictable for
# the backend until it defines its own limit (20 messages ≈ 10 round-trips:
# enough for follow-up references, small enough for any LLM context).
HISTORY_WINDOW = 20
_HISTORY_ROLES = ("user", "assistant")


def normalize_history(history: Any) -> Optional[List[dict]]:
    """Validates/normalizes a conversation history at the UI→RAG boundary.

    Policy (deterministic, documented):

    * ``None`` → ``None`` (not provided: caller behaves as before);
    * non-list input → ``None`` (controlled discard, never raises);
    * only ``dict`` entries with ``role`` in {``user``, ``assistant``} and a
      usable ``content`` survive; everything else is dropped;
    * ``content`` must be a string (or a scalar safely convertible with
      ``str()``); ``dict``/``list``/``None``/blank content is dropped;
    * chronological order is preserved exactly as given;
    * the window keeps the **last** ``HISTORY_WINDOW`` valid entries.

    The function only filters — it never rewrites, summarizes or invents
    content (contextual rewriting belongs to Backend/RAG).
    """
    if history is None:
        return None
    if not isinstance(history, (list, tuple)):
        return None
    entries: List[dict] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if role not in _HISTORY_ROLES:
            continue
        content = entry.get("content")
        if content is None or isinstance(content, (dict, list, tuple)):
            continue
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            continue
        entries.append({"role": role, "content": content})
    if len(entries) > HISTORY_WINDOW:
        entries = entries[-HISTORY_WINDOW:]
    return entries


def _accepts_keyword(method: Any, name: str) -> bool:
    """True when ``method`` can be called with the keyword ``name``."""
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def create_backend(session_id: Optional[str] = None) -> Optional[Any]:
    """Single connection point with the real RAG engine (backend team).

    Args:
        session_id: validated identity of the current Chainlit session.
            The future engine passes it to ``get_retriever(session_id=...)``
            so private documents stay scoped to their session. The UI only
            supplies the identifier: validation is a backend concern and
            nothing here implements authentication.

    Returns:
        The engine object (e.g. ``RagEngine()``) or ``None`` while it does
        not exist: then the UI runs on the demo mock.

    Note: once the real engine is connected, the resulting instance will have
    ``is_mock = False`` and the interface will stop showing the demo notice.
    """
    return None


def _accepts_documents(method: Any) -> bool:
    """True when ``method`` can be called as ``method(question, documents)``.

    A second positional parameter **named ``history`` is not documents**: if
    Backend/RAG extends the surface to ``answer_query(question, history=None)``
    the documents argument must never land in the history slot, so the
    two-argument call is skipped in that case (history then travels by
    keyword only).
    """
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return True  # unknown signature → keep the documented 2-arg call
    if any(
        p.kind == inspect.Parameter.VAR_POSITIONAL
        for p in sig.parameters.values()
    ):
        return True
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2 and positional[1].name == "history":
        return False
    return len(positional) >= 2


async def _call_backend(
    backend: Any,
    question: str,
    documents: list[str],
    history: Optional[list] = None,
) -> Any:
    """Invokes ``query()``/``ask()``/``answer_query()``, sync or async.

    ``RAGPipeline.answer_query(question)`` only takes the question, so
    ``documents`` are forwarded only when the exposed signature accepts a
    second positional argument.

    INTEGRATION SEAM (Backend/RAG): ``history`` is forwarded **only when the
    backend method explicitly accepts a ``history`` keyword** (or ``**kwargs``)
    and ``history is not None``. The current pipeline does not → behaviour is
    byte-identical to before. When Backend/RAG extends the signature (e.g.
    ``answer_query(question, history=None)``) the transport activates with no
    further UI change.
    """
    method = (
        getattr(backend, "query", None)
        or getattr(backend, "ask", None)
        or getattr(backend, "answer_query", None)
    )
    if method is None:
        raise TypeError("Backend exposes neither query(), ask() nor answer_query()")
    kwargs: dict = {}
    if history is not None and _accepts_keyword(method, "history"):
        kwargs["history"] = history
    result = (
        method(question, documents, **kwargs)
        if _accepts_documents(method)
        else method(question, **kwargs)
    )
    if inspect.isawaitable(result):
        result = await result
    return result


class RagAdapter:
    """Translates backend responses into the frontend contract.

    Always check ``RagAdapter().is_mock`` (instance): it is ``True`` while
    no real engine is connected in :func:`create_backend`.
    """

    is_mock: bool = True

    def __init__(self, backend: Any = _UNSET, session_id: str | None = None) -> None:
        """Args:
        backend: engine to query. When omitted, :func:`create_backend` is
            used (today it returns ``None`` → demo mock).
        session_id: identity of the current Chainlit session, forwarded to
            :func:`create_backend` so a future engine can scope its retriever
            to this session. Not authentication: the backend validates.
        """
        self.session_id = session_id
        if backend is _UNSET:
            backend = create_backend(session_id)
        self.backend = backend
        self.is_mock = backend is None

    async def ask(
        self,
        question: str,
        documents: list[str],
        labels: list[str] | None = None,
        history: list[dict] | None = None,
    ) -> RAGResponse:
        """Queries the backend and returns a normalized RAGResponse.

        Args:
            question: the user's question (raw; never rewritten here).
            documents: identifiers/paths of the documentation loaded in the
                session (today consumed by the mock; the real engine may
                filter or ignore this list).
            labels: original visible file names of the session (used by the
                demo mock to keep its simulated sources coherent with the
                documents the user actually uploaded).
            history: previous turns of the CURRENT conversation as
                ``[{"role": "user"|"assistant", "content": str}, ...]`` in
                chronological order — context for the current question, the
                question itself excluded. ``None``/omitted = legacy behaviour.
                Validated by :func:`normalize_history` and forwarded only if
                the backend signature accepts ``history`` (see
                :func:`_call_backend`). The demo mock never receives it
                (it stays stateless by design).

        Returns:
            RAGResponse ready for the UI.

        Raises:
            Exception: any backend failure propagates and the UI turns it
                into a friendly message (details are never exposed).
        """
        normalized = normalize_history(history)
        if self.backend is None:
            mock = MockRAG()
            return await mock.ask(question, documents, labels=labels)
        raw = await _call_backend(self.backend, question, documents, normalized)
        return self.format_response(raw)

    def format_response(self, raw: Any) -> RAGResponse:
        """Normalizes any raw response into the frontend contract.

        Accepts RAGResponse, dict with the contract keys, object with
        attributes, plain text or ``None``.  Never raises on missing or
        wrongly typed optional metadata.
        """
        return RAGResponse.from_any(raw)
