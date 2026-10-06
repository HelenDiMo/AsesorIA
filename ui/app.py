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

import chainlit as cl

sys.path.insert(0, str(Path(__file__).parent))

try:
    from . import formatters as fmt
    from .contracts import RAGResponse
    from .rag_adapter import RagAdapter
except ImportError:
    import formatters as fmt
    from contracts import RAGResponse
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


async def _register_and_report(items: Sequence[Tuple[str, str]]) -> None:
    """Validates (name, path), registers them and shows the documentation state."""
    valid, errors = _validate_files(items)
    for message in errors:
        await _notify(message)
    if valid:
        _register_documents(valid)
        await _notify(fmt.format_documents_state(_document_names()))


# --------------------------------------------------------------------------- #
# Guided document upload (AskFileMessage)
# --------------------------------------------------------------------------- #


async def _ask_for_files() -> None:
    files = await cl.AskFileMessage(
        content=(
            "Adjunta la documentación fiscal que quieres consultar "
            f"(PDF, TXT o Markdown, hasta {MAX_FILE_SIZE_MB} MB por archivo)."
        ),
        accept=FILE_ACCEPT,
        max_size_mb=MAX_FILE_SIZE_MB,
        max_files=5,
        timeout=120,
    ).send()

    if not files:
        await _notify(
            "No se ha seleccionado ningún archivo. Puedes volver a intentarlo "
            "cuando quieras."
        )
        return

    await _register_and_report([(f.name, f.path) for f in files])


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
    adapter = RagAdapter()
    response: RAGResponse | None = None
    error_kind: str | None = None

    # The Step shows the processing state; NEVER let an exception escape
    # the block: the Step itself would send str(exc) to the client.
    # Note: the Step name is used as avatar (/avatars/<name>) and that
    # endpoint rejects accents and symbols → ASCII name ("Buscando fuentes").
    async with cl.Step(name="Buscando fuentes", type="retrieval") as step:
        step.output = "Consultando la base documental…"
        await step.update()
        try:
            raw = await adapter.ask(question, documents)
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
    await _ask_for_files()


@cl.action_callback("ejemplo_pregunta")
async def _on_example_question(action: cl.Action) -> None:
    await cl.Message(content=EXAMPLE_QUESTION, type="user_message").send()
    await _answer_question(EXAMPLE_QUESTION)


# --------------------------------------------------------------------------- #
# Chainlit events
# --------------------------------------------------------------------------- #


@cl.on_chat_start
async def on_chat_start() -> None:
    """Welcome: identity, how it works and grounding limit."""
    _reset_session()
    is_mock = RagAdapter().is_mock
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
    if attached:
        await _register_and_report(attached)

    content = (message.content or "").strip()

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

    await _answer_question(content)
