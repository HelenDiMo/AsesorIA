"""Tests for pure helper logic in ui/app.py (no Chainlit server needed)."""

import ui.app as app


class TestValidateFiles:
    def test_valid_pdf(self, tmp_path):
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4 test")
        valid, errors = app._validate_files([("doc.pdf", str(path))])
        assert valid == [("doc.pdf", str(path))]
        assert errors == []

    def test_valid_markdown(self, tmp_path):
        path = tmp_path / "notas.md"
        path.write_text("# Hola")
        valid, errors = app._validate_files([("notas.md", str(path))])
        assert valid == [("notas.md", str(path))]
        assert errors == []

    def test_wrong_extension(self, tmp_path):
        path = tmp_path / "ejecutable.exe"
        path.write_bytes(b"MZ")
        valid, errors = app._validate_files([("ejecutable.exe", str(path))])
        assert valid == []
        assert len(errors) == 1
        assert "No puedo aceptar" in errors[0]

    def test_missing_file(self, tmp_path):
        valid, errors = app._validate_files([("perdido.pdf", str(tmp_path / "no.pdf"))])
        assert valid == []
        assert "No se ha podido leer" in errors[0]

    def test_empty_file(self, tmp_path):
        path = tmp_path / "vacio.pdf"
        path.write_bytes(b"")
        valid, errors = app._validate_files([("vacio.pdf", str(path))])
        assert valid == []
        assert "vacío" in errors[0]

    def test_mixed_batch_keeps_valid_files(self, tmp_path):
        good = tmp_path / "bueno.pdf"
        good.write_bytes(b"%PDF")
        bad = tmp_path / "malo.exe"
        bad.write_bytes(b"MZ")
        valid, errors = app._validate_files(
            [("bueno.pdf", str(good)), ("malo.exe", str(bad))]
        )
        assert valid == [("bueno.pdf", str(good))]
        assert len(errors) == 1

    def test_keeps_original_display_name(self, tmp_path):
        # Chainlit stores the file as <uuid>.pdf: the name the user
        # sees must be the original, not the storage name.
        path = tmp_path / "aaaa-bbbb.pdf"
        path.write_bytes(b"%PDF")
        valid, _ = app._validate_files([("manual-iva-2025.pdf", str(path))])
        assert valid == [("manual-iva-2025.pdf", str(path))]


class TestPlural:
    def test_singular(self):
        assert app._plural(1, "fragmento") == "1 fragmento"

    def test_plural(self):
        assert app._plural(3, "fragmento") == "3 fragmentos"


class TestUploadCommandMatching:
    def test_upload_commands_are_exact(self):
        # Matching is exact after normalization: questions are not hijacked.
        assert "cargar" in app.UPLOAD_COMMANDS
        assert "cargar documentos" in app.UPLOAD_COMMANDS
        # Legitimate questions containing the word are NOT commands.
        assert "¿qué gastos puedo cargar en el libro?" not in app.UPLOAD_COMMANDS
