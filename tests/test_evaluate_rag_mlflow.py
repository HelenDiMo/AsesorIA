"""Fase 9 — evaluación RAG con MLflow: scoring, agregación, juez y registro.

Todo sin red: motor y LLM simulados; MLflow contra sqlite temporal.
"""
import json
import sys
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from scripts.evaluate_rag_mlflow import (
    evaluate_engine,
    log_eval_metrics,
    make_judge,
    query_with_backoff,
    score_generation,
    summarize_generation,
)


def doc(doc_id="expected", page=7, end=8, text="fragmento"):
    return Document(
        page_content=text, metadata={"doc_id": doc_id, "page": page, "page_end": end}
    )


def summary_fixture():
    rows = [
        {"scores": {"document_hit": True, "page_hit": True, "retrieved": 8},
         "latency_ms": 100.0, "answer_chars": 50,
         "judge": {"faithfulness": 4, "relevance": 5}},
        {"scores": {"document_hit": False, "page_hit": None, "retrieved": 8},
         "latency_ms": 300.0, "answer_chars": 150,
         "judge": {"faithfulness": 2, "relevance": None}},
        {"scores": {"document_hit": None, "page_hit": None, "retrieved": 8},
         "latency_ms": 200.0, "answer_chars": 100, "judge": None},
    ]
    return summarize_generation(rows)


class FakeEngine:
    def __init__(self, answers):
        self.answers = list(answers)
        self.questions = []

    def query(self, question):
        self.questions.append(question)
        return self.answers.pop(0)


# --------------------------------------------------------------------------- #
# A — Scoring de evidencia (denominadores como evaluate_retrieval)
# --------------------------------------------------------------------------- #


def test_score_generation_hit_page_and_missing_labels():
    q = {"expected_doc_id": "expected", "expected_page": 8}
    assert score_generation(q, [doc()]) == {
        "document_hit": True, "page_hit": True, "retrieved": 1}
    assert score_generation(q, [doc(doc_id="other")])["document_hit"] is False
    assert score_generation(q, [doc(doc_id="other")])["page_hit"] is False
    assert score_generation(q, [doc(end=7)])["page_hit"] is False
    assert score_generation({}, [doc()]) == {
        "document_hit": None, "page_hit": None, "retrieved": 1}


def test_summarize_excludes_unlabeled_and_computes_means():
    summary = summary_fixture()
    assert summary["questions"] == 3
    assert summary["document_hit"] == {"hits": 1, "evaluated": 2, "rate": 0.5}
    assert summary["page_hit"] == {"hits": 1, "evaluated": 1, "rate": 1.0}
    assert summary["mean_latency_ms"] == 200.0
    assert summary["mean_answer_chars"] == 100
    assert summary["faithfulness_mean"] == 3.0
    assert summary["relevance_mean"] == 5.0
    assert summarize_generation([])["page_hit"]["rate"] is None


# --------------------------------------------------------------------------- #
# B — Ejecución del motor y juez
# --------------------------------------------------------------------------- #


def test_evaluate_engine_rows_without_judge():
    engine = FakeEngine([
        {"answer": "R1", "raw_documents": [doc()], "latency_ms": 120.0},
        {"answer": "R2", "raw_documents": [doc(doc_id="other")], "latency_ms": 80.0},
    ])
    questions = [
        {"id": "q1", "question": "P1", "expected_doc_id": "expected",
         "expected_page": 8},
        {"id": "q2", "question": "P2", "expected_doc_id": "expected",
         "expected_page": 8},
    ]
    rows = evaluate_engine(engine, questions)
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert engine.questions == ["P1", "P2"]
    assert rows[0]["scores"]["document_hit"] is True
    assert rows[1]["scores"]["document_hit"] is False
    assert rows[0]["answer_chars"] == 2
    assert rows[0]["judge"] is None


def test_evaluate_engine_invokes_judge_with_context():
    seen = []

    def judge(question, answer, context, ground_truth):
        seen.append((question, answer, context, ground_truth))
        return {"faithfulness": 5, "relevance": 4}

    engine = FakeEngine([
        {"answer": "R", "raw_documents": [doc(text="evidencia")], "latency_ms": 1.0}
    ])
    rows = evaluate_engine(
        engine,
        [{"id": "q1", "question": "P1", "note": "verificada"}],
        judge_fn=judge,
    )
    assert rows[0]["judge"] == {"faithfulness": 5, "relevance": 4}
    assert seen == [("P1", "R", "evidencia", "verificada")]


def test_make_judge_clamps_scores_and_fails_soft():
    class LLM:
        def __init__(self, payloads):
            self.payloads = list(payloads)

        def invoke(self, prompt):
            return SimpleNamespace(content=self.payloads.pop(0))

    judge = make_judge(LLM(['{"faithfulness": 9, "relevance": 0}']), retries=0)
    assert judge("q", "a", "c", "g") == {"faithfulness": 5, "relevance": 1}

    judge = make_judge(LLM(["sin json", "tampoco"]), retries=1)
    assert judge("q", "a", "c", "g") == {
        "faithfulness": None, "relevance": None}


# --------------------------------------------------------------------------- #
# B2 — Backoff ante rate limit (429); otros fallos se propagan
# --------------------------------------------------------------------------- #


def test_query_with_backoff_retries_only_rate_limit(monkeypatch):
    from scripts import evaluate_rag_mlflow as mod

    sleeps = []
    monkeypatch.setattr(mod, "time", SimpleNamespace(sleep=sleeps.append))

    class Flaky:
        def __init__(self):
            self.calls = 0

        def query(self, question):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Error code: 429 - rate limit reached")
            return {"answer": "ok", "raw_documents": [], "latency_ms": 1.0}

    flaky = Flaky()
    out = query_with_backoff(flaky, "q")
    assert out["answer"] == "ok"
    assert flaky.calls == 2
    assert sleeps == [30.0]

    class Broken:
        def query(self, question):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        query_with_backoff(Broken(), "q")
    assert sleeps == [30.0], "un fallo distinto de 429 no debe reintentar"

    class Always429:
        def query(self, question):
            raise RuntimeError("429")

    with pytest.raises(RuntimeError, match="429"):
        query_with_backoff(Always429(), "q")
    assert len(sleeps) == 4, "1 (Flaky) + 3 backoffs de Always429"


def test_make_judge_backs_off_on_rate_limit(monkeypatch):
    from scripts import evaluate_rag_mlflow as mod

    sleeps = []
    monkeypatch.setattr(mod, "time", SimpleNamespace(sleep=sleeps.append))

    class LLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 rate limit")
            return SimpleNamespace(
                content='{"faithfulness": 4, "relevance": 3}')

    judge = make_judge(LLM(), retries=2)
    assert judge("q", "a", "c", "g") == {"faithfulness": 4, "relevance": 3}
    assert sleeps == [30.0]


def test_extract_judge_scores_tolerates_formats():
    from scripts.evaluate_rag_mlflow import extract_judge_scores

    assert extract_judge_scores('{"faithfulness": 4, "relevance": 3}') == {
        "faithfulness": 4, "relevance": 3}
    assert extract_judge_scores(
        "```json\n{'faithfulness': 5, 'relevance': 2}\n```") == {
        "faithfulness": 5, "relevance": 2}
    assert extract_judge_scores(
        "Puntuación: faithfulness: 1, relevance: 10") == {
        "faithfulness": 1, "relevance": 5}
    with pytest.raises(ValueError, match="no encontradas"):
        extract_judge_scores("sin puntuación numérica")


def test_make_judge_uses_repair_prompt_on_bad_format():
    prompts = []

    class LLM:
        def invoke(self, prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return SimpleNamespace(content="respuesta sin puntuación")
            return SimpleNamespace(
                content='{"faithfulness": 3, "relevance": 4}')

    judge = make_judge(LLM(), retries=2)
    assert judge("q", "a", "c", "g") == {"faithfulness": 3, "relevance": 4}
    assert len(prompts) == 2
    assert "EXACTAMENTE" in prompts[1]
    assert "EXACTAMENTE" not in prompts[0]


# --------------------------------------------------------------------------- #
# C — Registro en MLflow (sqlite temporal)
# --------------------------------------------------------------------------- #


def test_log_eval_metrics_registers_run(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("asesoria-rag")
    run = mlflow.start_run(run_name="rag-eval-test")
    try:
        log_eval_metrics(summary_fixture())
    finally:
        mlflow.end_run()

    stored = mlflow.get_run(run.info.run_id).data.metrics
    assert stored["questions"] == 3
    assert stored["document_hit_rate"] == 0.5
    assert stored["page_hit_rate"] == 1.0
    assert stored["mean_latency_ms"] == 200.0
    assert stored["faithfulness_mean"] == 3.0
    assert stored["relevance_mean"] == 5.0


# --------------------------------------------------------------------------- #
# D — CLI end-to-end sin red (motor simulado, MLflow sqlite)
# --------------------------------------------------------------------------- #


def _write_fixtures(tmp_path, monkeypatch, engine_answers):
    manifest = {
        "status": "verified",
        "directory": str(tmp_path / "chroma"),
        "model": "intfloat/multilingual-e5-base",
        "collections": {"corpus_eval": {"id1": "digest"}},
        "report": {
            "documents": [{"doc_id": "manual", "pages": 50}],
            "max_input_tokens": 512,
            "passage_prefix": "passage: ",
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps({"questions": [
        {"id": "q1", "question": "¿Cuánto se aplica?", "expected_doc_id": "manual",
         "expected_page": 10, "note": "verificada"},
        {"id": "q2", "question": "¿Cuándo toca?", "note": "sin referencia"},
    ]}), encoding="utf-8")

    from scripts import evaluate_rag_mlflow as mod

    fake = FakeEngine(engine_answers)
    monkeypatch.setattr(mod, "build_engine", lambda settings: fake)
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    )
    monkeypatch.setattr(sys, "argv", [
        "evaluate_rag_mlflow", "--manifest", str(manifest_path),
        "--benchmark", str(benchmark_path), "--collection", "corpus_eval",
    ])
    return manifest_path, fake


def test_main_runs_full_eval_without_network(tmp_path, monkeypatch):
    manifest_path, fake = _write_fixtures(tmp_path, monkeypatch, [
        {"answer": "A1", "latency_ms": 100.0, "raw_documents": [
            Document(page_content="x", metadata={"doc_id": "manual", "page": 10,
                                                 "page_end": 11})]},
        {"answer": "A2", "latency_ms": 50.0, "raw_documents": []},
    ])

    from scripts.evaluate_rag_mlflow import main

    main()

    assert fake.questions == ["¿Cuánto se aplica?", "¿Cuándo toca?"]
    results_path = manifest_path.parent / "evaluations"
    produced = next(results_path.iterdir())
    report = json.loads((produced / "results.json").read_text(encoding="utf-8"))
    assert report["collection"] == "corpus_eval"
    assert report["judge"] is False
    assert report["summary"]["document_hit"] == {"hits": 1, "evaluated": 1,
                                                 "rate": 1.0}
    assert report["summary"]["page_hit"]["rate"] == 1.0
    assert report["summary"]["mean_latency_ms"] == 75.0

    import os

    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name("asesoria-rag")
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 1, "debe crearse exactamente un run de evaluación"
    metrics = runs[0].data.metrics
    assert metrics["document_hit_rate"] == 1.0
    assert metrics["mean_latency_ms"] == 75.0


def test_main_refuses_without_tracking_uri(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_rag_mlflow", "--manifest", "x.json", "--collection", "c"])
    from scripts.evaluate_rag_mlflow import main

    try:
        main()
    except SystemExit as exc:
        assert "MLFLOW_TRACKING_URI" in str(exc.code)
    else:
        raise AssertionError("SystemExit esperado sin MLFLOW_TRACKING_URI")
