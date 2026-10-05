"""Asesor Fiscal IA — Chainlit interface (Frontend & UX).

UI orchestrator. Contains NO RAG logic: every query goes through
`RagAdapter`, which returns the `RAGResponse` contract (`contracts.py`).

Question flow:
    on_message → validation → cl.Step(processing state) → adapter
    → answer | no information | error → collapsible sources panel.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Sequence, Tuple
from uuid import uuid4

import chainlit as cl

sys.path.insert(0, str(Path(__file__).parent))

try:
    from . import formatters as fmt
    from .contracts import RAGResponse
    from .guidance import (
        GUIDANCE_SUGGESTIONS,
        GUIDANCE_SUGGESTIONS_WHILE_CHAT,
        is_ambiguous,
    )
    from .rag_adapter import RagAdapter
except ImportError:
    import formatters as fmt
    from contracts import RAGResponse
    from guidance import (
        GUIDANCE_SUGGESTIONS,
        GUIDANCE_SUGGESTIONS_WHILE_CHAT,
        is_ambiguous,
    )
    from rag_adapter import RagAdapter

logger = logging.getLogger("asesor_fiscal.ui")

# --------------------------------------------------------------------------- #
# UI configuration
# --------------------------------------------------------------------------- #

MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}
MAX_QUESTION_LENGTH = 800
EXAMPLE_QUESTION = "¿Qué gastos son deducibles en el IRPF de un autónomo?"

# Short commands that open the upload assistant (exact match so legitimate
# questions containing the word "archivo" are not hijacked).
UPLOAD_COMMANDS = {
    "cargar",
    "subir",
    "cargar archivo",
    "subir archivo",
    "cargar documentos",
    "subir documentos",
    "cargar documento",
    "subir documento",
    "cargar documentación",
    "subir documentación",
}

FILE_ACCEPT = {
    "application/pdf": [".pdf"],
    "text/plain": [".txt", ".md"],
    "text/markdown": [".md"],
    "text/x-markdown": [".md"],
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def _reset_session() -> None:
    cl.user_session.set("documents", [])
    cl.user_session.set("document_labels", {})
    cl.user_session.set("has_asked", False)


_FALLBACK_SESSION_ID: str | None = None


def _session_id() -> str:
    """Stable identifier of the current Chainlit session — NOT authentication.

    Prefers Chainlit's own session id (``cl.context.session.id``); outside a
    session (unit tests, early startup) falls back to a process-wide id so
    the adapter seam (:func:`rag_adapter.create_backend`) always receives a
    value. Validating the identity is a backend concern, never this UI's.
    """
    try:
        sid = cl.context.session.id
        if sid:
            return str(sid)
    except Exception:  # noqa: BLE001 - no context yet (tests/startup)
        pass
    global _FALLBACK_SESSION_ID
    if _FALLBACK_SESSION_ID is None:
        _FALLBACK_SESSION_ID = uuid4().hex
    return _FALLBACK_SESSION_ID


def _get_documents() -> List[str]:
    return cl.user_session.get("documents") or []


def _register_documents(items: Sequence[Tuple[str, str]]) -> None:
    """Registers (visible name, path) of documents without duplicates.

    The visible name is the original filename; the storage path is
    provided by Chainlit (uuid) and must not be shown to the user.
    """
    paths = _get_documents()
    labels: dict = dict(cl.user_session.get("document_labels") or {})
    for name, path in items:
        if not path:
            continue
        if path not in paths:
            paths.append(path)
        if name and not labels.get(path):
            labels[path] = name
    cl.user_session.set("documents", paths)
    cl.user_session.set("document_labels", labels)


def _document_names() -> List[str]:
    """Visible document names, in upload order."""
    labels = cl.user_session.get("document_labels") or {}
    return [labels.get(p) or Path(p).name for p in _get_documents()]


# --------------------------------------------------------------------------- #
# File validation (real ingestion belongs to the backend, here only reception)
# --------------------------------------------------------------------------- #


def _validate_files(
    items: Sequence[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Validates (name, path) and returns (valid pairs, error messages)."""
    valid: List[Tuple[str, str]] = []
    errors: List[str] = []
    for name, path in items:
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(fmt.format_file_error(name, "type"))
            continue
        if not path or not Path(path).exists():
            errors.append(fmt.format_file_error(name, "read"))
            continue
        if Path(path).stat().st_size == 0:
            errors.append(fmt.format_file_error(name, "empty"))
            continue
        valid.append((name, path))
    return valid, errors


def _attached_files(message: cl.Message) -> List[Tuple[str, str]]:
    """Extracts (name, path) from files attached to the message."""
    files: List[Tuple[str, str]] = []
    for element in getattr(message, "elements", None) or []:
        name = getattr(element, "name", None) or "archivo"
        path = getattr(element, "path", None)
        if path:
            files.append((str(name), str(path)))
    return files


# --------------------------------------------------------------------------- #
# UI states
# --------------------------------------------------------------------------- #


async def _notify(text: str, actions: List[cl.Action] | None = None) -> None:
    await cl.Message(content=text, actions=actions).send()


def _load_action() -> cl.Action:
    return cl.Action(
        name="cargar_documentacion",
        payload={"intent": "upload"},
        label="Cargar documentación",
        tooltip="Adjuntar un archivo PDF, TXT o Markdown",
        icon="upload",
    )


# --------------------------------------------------------------------------- #
# Task lifecycle for action callbacks (limitación upstream)
# --------------------------------------------------------------------------- #
# Chainlit wraps typed messages in process_message() with a balanced
# task_start/task_end, but action callbacks travel through the HTTP
# endpoint (server.py) WITHOUT that wrapper, while AskFileMessage.send()
# always emits an orphan task_start in its finally block (emitter.py
# ``send_ask_user``). Left unbalanced, the frontend keeps the composer in
# "Stop" state forever. The helpers below mirror the process_message
# wrapper around action callbacks so the UI bookkeeping always closes.


async def _emit_task(kind: str) -> None:
    """Emits task_start/task_end; never breaks the flow on failure."""
    try:
        emitter = cl.context.emitter
        if kind == "start":
            await emitter.task_start()
        else:
            await emitter.task_end()
    except Exception:  # noqa: BLE001 - UI bookkeeping must not break actions
        logger.debug("task event not emitted (%s)", kind)


# --------------------------------------------------------------------------- #
# Contextual suggestions (shown right after the first upload)
# --------------------------------------------------------------------------- #

SUGGESTION_QUESTIONS: List[Tuple[str, str]] = [
    ("IVA soportado", "¿Qué es el IVA soportado y cómo se deduce?"),
    ("Gastos deducibles", "¿Qué gastos son deducibles en el IRPF de un autónomo?"),
    ("Modelo 303", "¿Cuándo se presenta el modelo 303?"),
    ("Obligaciones", "¿Qué obligaciones fiscales tengo como autónomo?"),
]


def _has_asked() -> bool:
    return bool(cl.user_session.get("has_asked"))


def _mark_asked() -> None:
    cl.user_session.set("has_asked", True)


def _suggestion_actions(asked: bool) -> List[cl.Action] | None:
    """Spanish example questions; only while the conversation has no question."""
    if asked:
        return None
    return [
        cl.Action(
            name=f"sugerencia_{i}",
            payload={"intent": "question", "text": question},
            label=f"💡 {title}",
            tooltip="Hacer esta pregunta de ejemplo",
        )
        for i, (title, question) in enumerate(SUGGESTION_QUESTIONS)
    ]


def _guidance_actions() -> List[cl.Action]:
    """Real suggestion buttons for vague openers (guidance.py).

    Each button carries an example question in its payload: clicking it
    sends that question through the normal RAG flow (a real action, not
    decoration). 5 suggestions on a fresh conversation; only the first 2
    once the chat has started (the conversation itself takes priority).
    """
    suggestions = GUIDANCE_SUGGESTIONS
    if _has_asked():
        suggestions = suggestions[:GUIDANCE_SUGGESTIONS_WHILE_CHAT]
    return [
        cl.Action(
            name=f"orientacion_{i}",
            payload={"intent": "question", "text": question},
            label=title,
            tooltip=f"Sugerencia: {title}",
        )
        for i, (title, question) in enumerate(suggestions)
    ]


async def _show_guidance() -> None:
    """Answers a clearly ambiguous query with orientation, never with the RAG."""
    await _notify(
        fmt.format_clarify(
            has_docs=bool(_get_documents()), in_conversation=_has_asked()
        ),
        actions=_guidance_actions(),
    )


async def _register_and_report(
    items: Sequence[Tuple[str, str]], suggest: bool = False
) -> None:
    """Validates (name, path), registers them and shows the documentation state.

    ``suggest`` marks uploads made without an accompanying question: then,
    while the conversation is still empty, the state message also guides the
    user with a prompt and contextual suggestion actions (§12).
    """
    valid, errors = _validate_files(items)
    for message in errors:
        await _notify(message)
    if valid:
        _register_documents(valid)
        content = fmt.format_documents_state(_document_names())
        suggestions = _suggestion_actions(_has_asked()) if suggest else None
        if suggestions:
            content += "\n\n**¿Qué quieres consultar?**"
        await _notify(content, actions=suggestions)


# --------------------------------------------------------------------------- #
# Guided document upload (AskFileMessage)
# --------------------------------------------------------------------------- #


ASK_TIMEOUT_S = 120


async def _ask_for_files() -> None:
    files = await cl.AskFileMessage(
        content=(
            "Adjunta la documentación fiscal que quieres consultar "
            f"(PDF, TXT o Markdown, hasta {MAX_FILE_SIZE_MB} MB por archivo).\n\n"
            f"Este aviso se cierra solo en {ASK_TIMEOUT_S // 60} minutos."
        ),
        accept=FILE_ACCEPT,
        max_size_mb=MAX_FILE_SIZE_MB,
        max_files=5,
        timeout=ASK_TIMEOUT_S,
    ).send()

    if not files:
        await _notify(
            "No se ha seleccionado ningún archivo. Puedes volver a intentarlo "
            "cuando quieras."
        )
        return

    await _register_and_report([(f.name, f.path) for f in files], suggest=True)


# --------------------------------------------------------------------------- #
# RAG query (processing state + answer / no information / error)
# --------------------------------------------------------------------------- #


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


async def _query_engine(question: str) -> Tuple[RAGResponse | None, str | None]:
    """Runs the adapter inside a retrieval Step.

    Returns ``(response, error_kind)``; ``error_kind`` is ``None`` on success.
    """
    documents = _get_documents()
    adapter = RagAdapter(session_id=_session_id())
    response: RAGResponse | None = None
    error_kind: str | None = None

    # The Step shows the processing state; NEVER let an exception escape
    # the block: the Step itself would send str(exc) to the client.
    # Note: the Step name is used as avatar (/avatars/<name>) and that
    # endpoint rejects accents and symbols → ASCII name; the friendly,
    # accented copy lives in the output (visible while it runs).
    async with cl.Step(name="Consultando la documentacion", type="retrieval") as step:
        step.output = "🔎 Consultando la documentación…"
        await step.update()
        try:
            raw = await adapter.ask(question, documents, labels=_document_names())
            response = adapter.format_response(raw)
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("RAG engine unavailable: %s", type(exc).__name__)
            error_kind = "connection"
            step.output = "No se pudo conectar con el motor de consulta"
        except Exception as exc:
            logger.exception("RAG backend error: %s", type(exc).__name__)
            error_kind = "unknown"
            step.output = "Error al procesar la consulta"
        else:
            n_sources = len(response.sources) if response else 0
            if response and response.grounded and n_sources:
                step.output = f"Recuperado: {_plural(n_sources, 'fragmento')}"
            else:
                step.output = "Sin resultados en la documentación"
        await step.update()
    return response, error_kind


async def _sources_panel(response: RAGResponse) -> None:
    """Collapsible traceability panel (native Chainlit accordion).

    Renders document · page · section · exact snippet inside a ``cl.Step``
    so the main thread stays tidy. The step name feeds ``/avatars/<name>``
    → ASCII only; the count is visible in the name (collapsed) and in the
    output title (expanded).
    """
    if not response.sources:
        return
    n = len(response.sources)
    async with cl.Step(
        name=f"Fuentes utilizadas {n}",
        type="tool",
        default_open=False,     # starts collapsed: does not clutter the chat
        auto_collapse=True,     # folds again while navigating
    ) as step:
        step.output = fmt.format_sources_block(response.sources)


async def _render_response(
    response: RAGResponse | None, error_kind: str | None
) -> None:
    """Maps the engine result to the right UI state."""
    if error_kind or response is None:
        await _notify(fmt.format_error(error_kind or "unknown"))
        return
    if not response.grounded or not response.has_answer:
        await _notify(fmt.format_no_answer(response))
        return
    await _notify(fmt.format_answer(response))
    await _sources_panel(response)


async def _answer_question(question: str) -> None:
    _mark_asked()
    if not _get_documents():
        await _notify(fmt.format_no_documents(), actions=[_load_action()])
        return
    response, error_kind = await _query_engine(question)
    await _render_response(response, error_kind)


# --------------------------------------------------------------------------- #
# Actions (welcome buttons)
# --------------------------------------------------------------------------- #


@cl.action_callback("cargar_documentacion")
async def _on_load_documents(action: cl.Action) -> None:
    # Balanced start/end: action callbacks have no process_message wrapper.
    await _emit_task("start")
    try:
        await _ask_for_files()
    finally:
        await _emit_task("end")


@cl.action_callback("ejemplo_pregunta")
async def _on_example_question(action: cl.Action) -> None:
    await _emit_task("start")
    try:
        await cl.Message(content=EXAMPLE_QUESTION, type="user_message").send()
        await _answer_question(EXAMPLE_QUESTION)
    finally:
        await _emit_task("end")


async def _on_suggestion(action: cl.Action) -> None:
    question = str((action.payload or {}).get("text") or "").strip()
    if not question:
        return
    await _emit_task("start")
    try:
        await cl.Message(content=question, type="user_message").send()
        await _answer_question(question)
    finally:
        await _emit_task("end")


for _i, _ in enumerate(SUGGESTION_QUESTIONS):
    cl.action_callback(f"sugerencia_{_i}")(_on_suggestion)

for _g, _ in enumerate(GUIDANCE_SUGGESTIONS):
    cl.action_callback(f"orientacion_{_g}")(_on_suggestion)


# --------------------------------------------------------------------------- #
# Chainlit events
# --------------------------------------------------------------------------- #


@cl.on_chat_start
async def on_chat_start() -> None:
    """Welcome: identity, how it works and grounding limit."""
    _reset_session()
    is_mock = RagAdapter(session_id=_session_id()).is_mock
    await cl.Message(
        content=fmt.format_welcome(is_mock),
        actions=[_load_action()],
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Validates input, registers attached files and answers."""
    try:
        await _handle_message(message)
    except Exception:
        # Last safety net: never leave the user without an answer.
        logger.exception("Unhandled error while processing the message")
        await _notify(fmt.format_error("unknown"))


async def _handle_message(message: cl.Message) -> None:
    attached = _attached_files(message)
    content = (message.content or "").strip()

    if attached:
        # Suggestions only when the upload comes without a question: then
        # the conversation has not started yet (§12).
        await _register_and_report(attached, suggest=not content)

    if not content:
        if attached:
            return  # only documents were attached
        await _notify(fmt.format_empty_question())
        return

    if len(content) > MAX_QUESTION_LENGTH:
        await _notify(fmt.format_question_too_long(MAX_QUESTION_LENGTH))
        return

    if content.lower().rstrip("¿?¡!.") in UPLOAD_COMMANDS:
        await _ask_for_files()
        return

    # Conversational UX guidance: clearly vague openers get a friendly
    # orientation state instead of a generic RAG answer (see guidance.py).
    if is_ambiguous(content):
        await _show_guidance()
        return

    await _answer_question(content)
