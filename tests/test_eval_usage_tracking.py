"""Volcado a MLflow: uso de tokens, coste del LLM y resultados de la suite.

Sin red: respuestas del LLM simuladas con ``SimpleNamespace`` y MLflow
contra sqlite temporal (mismo patrón que test_evaluate_rag_mlflow).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_rag_mlflow import PRICING, TokenUsageCounter, capture_usage
from scripts.log_test_results import parse_junit


def llm_result(token_usage=None, usage_metadata=None):
    """LLMResult simulado: token_usage de llm_output o usage_metadata del msg."""
    message = SimpleNamespace(usage_metadata=usage_metadata)
    return SimpleNamespace(
        llm_output={"token_usage": token_usage} if token_usage else {},
        generations=[[SimpleNamespace(message=message)]],
    )


# --------------------------------------------------------------------------- #
# A — Contador de uso (callbacks LCEL) y coste
# --------------------------------------------------------------------------- #


def test_counter_aggregates_from_llm_output_token_usage():
    counter = TokenUsageCounter()
    counter.on_llm_end(llm_result(
        token_usage={"prompt_tokens": 1200, "completion_tokens": 300}))
    counter.on_llm_end(llm_result(
        token_usage={"prompt_tokens": 800, "completion_tokens": 200}))
    assert counter.calls == 2
    assert counter.prompt_tokens == 2000
    assert counter.completion_tokens == 500


def test_counter_falls_back_to_message_usage_metadata():
    counter = TokenUsageCounter()
    counter.on_llm_end(llm_result(usage_metadata={
        "input_tokens": 1500, "output_tokens": 400}))
    assert counter.calls == 1
    assert counter.prompt_tokens == 1500
    assert counter.completion_tokens == 400


def test_counter_ignores_responses_without_usage():
    counter = TokenUsageCounter()
    counter.on_llm_end(SimpleNamespace(llm_output={}, generations=[]))
    assert (counter.calls, counter.prompt_tokens, counter.completion_tokens) == (0, 0, 0)


def test_cost_uses_published_groq_pricing():
    counter = TokenUsageCounter()
    counter.on_llm_end(llm_result(
        token_usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}))
    assert counter.cost_usd == pytest.approx(PRICING["prompt"] + PRICING["completion"])
    assert PRICING == {"prompt": 0.15, "completion": 0.60}


def test_capture_usage_sets_global_and_resets():
    import scripts.evaluate_rag_mlflow as mod

    counter = TokenUsageCounter()
    capture_usage(counter)
    assert mod._USAGE_COUNTER is counter
    capture_usage(None)
    assert mod._USAGE_COUNTER is None


# --------------------------------------------------------------------------- #
# B — Métricas de coste en MLflow (sqlite temporal)
# --------------------------------------------------------------------------- #


def test_log_eval_metrics_registers_usage_and_cost(tmp_path, monkeypatch):
    import mlflow

    from scripts.evaluate_rag_mlflow import log_eval_metrics, summarize_generation

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("asesoria-rag")
    rows = [{"scores": {"document_hit": True, "page_hit": True, "retrieved": 8},
             "latency_ms": 100.0, "retrieval_latency_ms": 20.0,
             "generation_latency_ms": 80.0, "answer_chars": 50, "judge": None}]
    summary = summarize_generation(rows)
    summary["usage"] = {"llm_calls": 3, "prompt_tokens": 4000,
                        "completion_tokens": 1000, "cost_usd": 0.0012,
                        "rate_limit_retries": 1}
    run = mlflow.start_run(run_name="rag-eval-usage-test")
    try:
        log_eval_metrics(summary)
    finally:
        mlflow.end_run()

    stored = mlflow.get_run(run.info.run_id).data.metrics
    assert stored["prompt_tokens_total"] == 4000
    assert stored["completion_tokens_total"] == 1000
    assert stored["llm_calls"] == 3
    assert stored["cost_usd_total"] == 0.0012
    assert stored["cost_usd_per_question"] == 0.0012
    assert stored["rate_limit_retries"] == 1
    assert stored["mean_retrieval_latency_ms"] == 20.0
    assert stored["mean_generation_latency_ms"] == 80.0


# --------------------------------------------------------------------------- #
# C — Parseo del informe JUnit de pytest
# --------------------------------------------------------------------------- #


def _write_xml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_junit_sums_multiple_testsuites(tmp_path):
    xml = _write_xml(tmp_path / "multi.xml", """<?xml version="1.0"?>
<testsuites>
  <testsuite name="a" tests="3" failures="1" errors="0" skipped="1" time="1.5"/>
  <testsuite name="b" tests="4" failures="0" errors="1" skipped="0" time="2.5"/>
</testsuites>""")
    totals = parse_junit(xml)
    assert totals == {"tests": 7, "failures": 1, "errors": 1, "skipped": 1,
                      "time": 4.0, "passed": 4}


def test_parse_junit_accepts_single_testsuite_root(tmp_path):
    xml = _write_xml(tmp_path / "single.xml", """<?xml version="1.0"?>
<testsuite tests="10" failures="0" errors="0" skipped="2" time="9.25"/>""")
    totals = parse_junit(xml)
    assert totals["passed"] == 8
    assert totals["time"] == 9.25


def test_parse_junit_rejects_empty_file(tmp_path):
    xml = _write_xml(tmp_path / "empty.xml", "<testsuites/>")
    with pytest.raises(ValueError, match="testsuite"):
        parse_junit(xml)
