"""Adapter: the single boundary between the interface and the RAG backend.

`app.py` only knows this module and the contract (`contracts.py`).  This is
where the mock is swapped for the real engine once it exists:

    MockRAG (today)  →  src/rag/engine.py (tomorrow, RAG team)

No Chroma, LangChain, embeddings, prompts or LLM providers in the UI.

Integration (ONLY this file is touched):

    def create_backend():
        from src.rag.engine import RagEngine   # RAG team
        return RagEngine()

The engine must expose ``query(question, documents)`` or ``ask(question,
documents)`` (sync or async) and return a dict/object with the contract keys
(`contracts.py`).  Whatever it returns is normalized with
``RAGResponse.from_any``, so missing metadata, LangChain-like objects or
``None`` lists never break the interface.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

try:
    from .contracts import RAGResponse
    from .mock_rag import MockRAG
except ImportError:
    from contracts import RAGResponse
    from mock_rag import MockRAG

_UNSET = object()


def create_backend() -> Optional[Any]:
    """Single connection point with the real RAG engine (backend team).

    Returns:
        The engine object (e.g. ``RagEngine()``) or ``None`` while it does
        not exist: then the UI runs on the demo mock.

    Note: once the real engine is connected, the resulting instance will have
    ``is_mock = False`` and the interface will stop showing the demo notice.
    """
    return None


async def _call_backend(backend: Any, question: str, documents: list[str]) -> Any:
    """Invokes the engine's ``query()`` or ``ask()``, sync or async."""
    method = getattr(backend, "query", None) or getattr(backend, "ask", None)
    if method is None:
        raise TypeError("Backend exposes neither query() nor ask()")
    result = method(question, documents)
    if inspect.isawaitable(result):
        result = await result
    return result


class RagAdapter:
    """Translates backend responses into the frontend contract.

    Always check ``RagAdapter().is_mock`` (instance): it is ``True`` while
    no real engine is connected in :func:`create_backend`.
    """

    is_mock: bool = True

    def __init__(self, backend: Any = _UNSET) -> None:
        """Args:
        backend: engine to query. When omitted, :func:`create_backend` is
            used (today it returns ``None`` → demo mock).
        """
        if backend is _UNSET:
            backend = create_backend()
        self.backend = backend
        self.is_mock = backend is None

    async def ask(self, question: str, documents: list[str]) -> RAGResponse:
        """Queries the backend and returns a normalized RAGResponse.

        Args:
            question: the user's question.
            documents: identifiers/paths of the documentation loaded in the
                session (today consumed by the mock; the real engine may
                filter or ignore this list).

        Returns:
            RAGResponse ready for the UI.

        Raises:
            Exception: any backend failure propagates and the UI turns it
                into a friendly message (details are never exposed).
        """
        if self.backend is None:
            mock = MockRAG()
            return await mock.ask(question, documents)
        raw = await _call_backend(self.backend, question, documents)
        return self.format_response(raw)

    def format_response(self, raw: Any) -> RAGResponse:
        """Normalizes any raw response into the frontend contract.

        Accepts RAGResponse, dict with the contract keys, object with
        attributes, plain text or ``None``.  Never raises on missing or
        wrongly typed optional metadata.
        """
        return RAGResponse.from_any(raw)
