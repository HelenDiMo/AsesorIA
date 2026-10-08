"""Upload allow-list — every advertised extension regardless of browser MIME.

`ui/app.py::FILE_ACCEPT` (AskFile dialog) and
`ui/.chainlit/config.toml` (composer paperclip) feed Chainlit's
server-side validator, which matches both the reported MIME type and the
file extension. Chromium reports an empty MIME type for `.md`, which must
not reject the file the dialog explicitly advertises (PDF, TXT o Markdown).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from chainlit.server import validate_file_mime_type
from chainlit.types import AskFileSpec

from ui.app import FILE_ACCEPT

_CONFIG_TOML = Path(__file__).resolve().parents[1] / "ui" / ".chainlit" / "config.toml"

_ALLOWED = [
    ("manual.pdf", "application/pdf"),
    ("notas.txt", "text/plain"),
    ("notas.md", "text/markdown"),
    ("notas.md", ""),
    ("notas.md", "application/octet-stream"),
]

_REJECTED = [
    ("foto.png", "image/png"),
    ("hoja.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("script.exe", ""),
]


def _spec(accept) -> AskFileSpec:
    return AskFileSpec(
        accept=accept,
        max_files=5,
        max_size_mb=50,
        timeout=120,
        type="file",
        step_id="ask-1",
    )


def _paperclip_accept():
    with open(_CONFIG_TOML, "rb") as fh:
        data = tomllib.load(fh)
    return data["features"]["spontaneous_file_upload"]["accept"]


class TestAskFileDialog:
    @pytest.mark.parametrize("filename, content_type", _ALLOWED)
    def test_allowed(self, filename, content_type):
        validate_file_mime_type(
            SimpleNamespace(filename=filename, content_type=content_type),
            _spec(FILE_ACCEPT),
        )

    @pytest.mark.parametrize("filename, content_type", _REJECTED)
    def test_rejected(self, filename, content_type):
        with pytest.raises(ValueError, match="not allowed"):
            validate_file_mime_type(
                SimpleNamespace(filename=filename, content_type=content_type),
                _spec(FILE_ACCEPT),
            )


class TestComposerPaperclip:
    @pytest.mark.parametrize("filename, content_type", _ALLOWED)
    def test_allowed(self, filename, content_type):
        validate_file_mime_type(
            SimpleNamespace(filename=filename, content_type=content_type),
            _spec(_paperclip_accept()),
        )

    @pytest.mark.parametrize("filename, content_type", _REJECTED)
    def test_rejected(self, filename, content_type):
        with pytest.raises(ValueError, match="not allowed"):
            validate_file_mime_type(
                SimpleNamespace(filename=filename, content_type=content_type),
                _spec(_paperclip_accept()),
            )
