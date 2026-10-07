"""Fase 8 — Tracing MLflow: off por defecto, spans reales y aislamiento de fallos.

Contrato cubierto:
* Sin MLFLOW_TRACKING_URI → cero spans, cero imports de mlflow, consultas idénticas.
* Con MLFLOW_TRACKING_URI (sqlite en tmp) → spans ``rag.query`` y ``rag.rewrite``
  con entradas/salidas; el contrato de RAGEngine.query no cambia.
* Un fallo de mlflow al abrir un span NO interrumpe el serving.
"""

import os

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.common import tracing
from src.rag import pipeline as pipeline_module
from src.rag.engine import RAGEngine
from src.rag.pipeline import RAGPipeline, resolve_standalone_query


def _reset_state(monkeypatch):
    monkeypatch.setattr(tracing, "_state", {"configured": False, "usable": False})


def _search():
    import mlflow

    # set_tracking_uri es global en el proceso: re-alinear con el env del test
    # (los fixtures ponen la uri en MLFLOW_TRACKING_URI).
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(tracing.DEFAULT_EXPERIMENT)
    if exp is None:
        return []
    return mlflow.search_traces(
        locations=[exp.experiment_id], return_type="list", flush=True
    )


def _span_names(traces):
    return [s.name for t in traces for s in t.data.spans]


def _span(traces, name):
    for t in traces:
        for s in t.data.spans:
            if s.name == name:
                return s
    return None


# --------------------------------------------------------------------------- #
# A — Off por defecto (cero efecto sin variable de entorno)
# --------------------------------------------------------------------------- #


class TestDisabledByDefault:
    @pytest.fixture(autouse=True)
    def _off(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        _reset_state(monkeypatch)

    def test_tracing_enabled_false(self):
        assert tracing.tracing_enabled() is False

    def test_setup_is_noop(self):
        tracing.setup_tracing()
        assert tracing._state == {"configured": False, "usable": False}

    def test_span_yields_none_without_mlflow(self):
        with tracing.trace_span("rag.query", {"question": "x"}) as span:
            assert span is None


# --------------------------------------------------------------------------- #
# B — Spans reales con tracking sqlite en tmp (no toca ./mlruns)
# --------------------------------------------------------------------------- #


class FakePipeline:
    """Respuestas fijas sin red; imita la firma de RAGPipeline.answer_query."""

    def __init__(self):
        self.calls = []

    def answer_query(self, question, history=None):
        self.calls.append({"question": question, "history": history})
        return {
            "answer": "Respuesta grounded.",
            "source_documents": [
                Document(
                    page_content="Fragmento fiscal recuperado.",
                    metadata={"source": "manual.pdf", "page": 3},
                )
            ],
            "metrics": {
                "retrieval_latency_s": 0.1,
                "generation_latency_s": 0.2,
                "total_latency_s": 0.3,
            },
        }


class TestTracingEnabled:
    @pytest.fixture(autouse=True)
    def _on(self, tmp_path, monkeypatch):
        db = (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
        _reset_state(monkeypatch)
        yield
        # El estado se restaura por monkeypatch; el uri global de mlflow se
        # sobrescribe en el siguiente test que configure tracing.

    def test_setup_configures_experiment(self):
        tracing.setup_tracing()
        assert tracing._state == {"configured": True, "usable": True}

        import mlflow

        exp = mlflow.get_experiment_by_name(tracing.DEFAULT_EXPERIMENT)
        assert exp is not None

    def test_engine_query_records_rag_query_span(self):
        engine = RAGEngine(pipeline=FakePipeline())
        result = engine.query(
            "¿Qué impuestos paga un autónomo?",
            history=[{"role": "user", "content": "hola"}],
        )

        assert set(result) == {
            "question",
            "answer",
            "sources",
            "metrics",
            "latency_ms",
            "raw_documents",
        }
        assert result["answer"] == "Respuesta grounded."
        assert result["latency_ms"] == pytest.approx(300.0)

        traces = _search()
        assert "rag.query" in _span_names(traces)
        span = _span(traces, "rag.query")
        assert span.inputs["question"] == "¿Qué impuestos paga un autónomo?"
        assert span.outputs["sources"] == 1
        assert span.outputs["answer_chars"] == len("Respuesta grounded.")
        assert span.attributes["latency_ms"] == pytest.approx(300.0)

    def test_span_opening_failure_does_not_break_query(self, monkeypatch):
        import mlflow

        def boom(*args, **kwargs):
            raise RuntimeError("mlflow caído")

        monkeypatch.setattr(mlflow, "start_span", boom)

        engine = RAGEngine(pipeline=FakePipeline())
        result = engine.query("¿Qué impuestos paga un autónomo?")

        assert result["answer"] == "Respuesta grounded."
        assert result["sources"]


# --------------------------------------------------------------------------- #
# C — Span de reescritura (solo con historial; fast-path sin span)
# --------------------------------------------------------------------------- #


class TestRewriteSpan:
    @pytest.fixture(autouse=True)
    def _on(self, tmp_path, monkeypatch):
        db = (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
        _reset_state(monkeypatch)

    def test_rewrite_span_records_standalone_query(self):
        history = [
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "¿En qué te ayudo?"},
        ]
        standalone = resolve_standalone_query(
            FakeListChatModel(responses=["alta como autónomo"]),
            "¿y presencial?",
            history,
        )
        assert standalone == "alta como autónomo"

        traces = _search()
        assert "rag.rewrite" in _span_names(traces)
        span = _span(traces, "rag.rewrite")
        assert span.outputs["standalone_query"] == "alta como autónomo"
        assert span.inputs["history_turns"] == 2

    def test_fast_path_without_history_creates_no_span(self):
        out = resolve_standalone_query(
            FakeListChatModel(responses=["nunca debería usarse"]),
            "consulta directa",
            [],
        )
        assert out == "consulta directa"
        assert _span_names(_search()) == []


# --------------------------------------------------------------------------- #
# D — Retrocompatibilidad: sin historial la firma sigue siendo la misma
# --------------------------------------------------------------------------- #


class TestContractIntact:
    def test_pipeline_and_engine_signatures(self):
        import inspect

        def params(fn):
            return [p for p in inspect.signature(fn).parameters if p != "self"]

        assert params(RAGPipeline.answer_query) == ["question", "history"]
        assert params(RAGEngine.query) == ["question", "history"]

    def test_fast_path_returns_original_question(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        _reset_state(monkeypatch)
        calls = []
        monkeypatch.setattr(
            pipeline_module,
            "QUERY_REWRITE_PROMPT",
            type("_No", (), {"__or__": lambda s, o: calls.append(1)})(),
        )
        assert (
            resolve_standalone_query(FakeListChatModel(responses=["x"]), "q", None)
            == "q"
        )
        assert calls == []
