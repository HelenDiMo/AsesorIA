"""Evaluación E2E de la cadena RAG registrada en MLflow (generación).

Complementa a ``scripts.evaluate_retrieval.py`` (solo retrieval, sin MLflow):
aquí se ejecuta el motor COMPLETO (retriever + prompt + LLM + rewriter) sobre
el benchmark revisado y se registran en un run de MLflow:

* ``document_hit_rate`` / ``page_hit_rate`` — la evidencia esperada del
  benchmark aparece entre los documentos recuperados (denominadores
  explícitos: los casos sin referencia no cuentan como fallos).
* ``mean_latency_ms`` — latencia total media por consulta (incluye rewriting
  cuando procede, tal y como lo reporta ``RAGEngine.query``).
* ``faithfulness_mean`` / ``relevance_mean`` (con ``--judge``) — LLM-as-judge
  sobre la respuesta generada, puntuación 1-5.

Cada consulta queda además trazada como span ``rag.query`` si
``MLFLOW_TRACKING_URI`` está activo (src/common/tracing.py).

Uso:
    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    python -m scripts.evaluate_rag_mlflow \
        --manifest chroma_db/corpus_runs/<id>/manifest.json \
        --collection corpus_384_48 [--limit N] [--judge]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

from src.common.settings import PROJECT_ROOT, Settings
from scripts.evaluate_retrieval import load_questions

JUDGE_SYSTEM = (
    "Eres evaluador de un asistente fiscal para autónomos en España. "
    "Puntúa de 1 a 5 (enteros). Responde SOLO con JSON: "
    '{"faithfulness": <1-5>, "relevance": <1-5>}.'
)
JUDGE_USER = (
    "PREGUNTA: {question}\n\n"
    "FRAGMENTOS RECUPERADOS:\n{context}\n\n"
    "RESPUESTA DEL ASISTENTE:\n{answer}\n\n"
    "REFERENCIA VERIFICADA:\n{ground_truth}\n\n"
    "faithfulness: ¿la respuesta se sostiene SOLO en los fragmentos "
    "(sin inventar cifras ni afirmaciones)? relevance: ¿responde a la "
    "pregunta y aprovecha la referencia verificada?"
)


def score_generation(question: Dict[str, Any], raw_documents: List[Any]) -> Dict[str, Any]:
    """Aciertos de evidencia sobre los documentos recuperados por el motor.

    ``None`` significa que el benchmark no tiene referencia para ese campo
    (igual que ``scripts.evaluate_retrieval.score_question``): no cuenta
    como fallo en los agregados.
    """
    expected_doc = question.get("expected_doc_id")
    expected_page = question.get("expected_page")
    matching = [
        d for d in raw_documents
        if expected_doc is not None and (d.metadata or {}).get("doc_id") == expected_doc
    ]
    page_hit = (
        None if expected_page is None else
        any(
            (d.metadata or {}).get("page", -1)
            <= expected_page
            <= (d.metadata or {}).get("page_end", -1)
            for d in matching
        )
    )
    return {
        "document_hit": bool(matching) if expected_doc is not None else None,
        "page_hit": page_hit,
        "retrieved": len(raw_documents),
    }


def summarize_generation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tasas con denominadores explícitos; los casos sin referencia se excluyen."""
    summary: Dict[str, Any] = {"questions": len(rows)}
    for metric in ("document_hit", "page_hit"):
        values = [r["scores"][metric] for r in rows if r["scores"][metric] is not None]
        summary[metric] = {
            "hits": sum(values),
            "evaluated": len(values),
            "rate": sum(values) / len(values) if values else None,
        }
    summary["mean_latency_ms"] = mean(r["latency_ms"] for r in rows) if rows else 0.0
    summary["mean_answer_chars"] = mean(r["answer_chars"] for r in rows) if rows else 0
    judged = [r["judge"] for r in rows if r.get("judge") is not None]
    for metric in ("faithfulness", "relevance"):
        values = [j[metric] for j in judged if j.get(metric) is not None]
        if values:
            summary[f"{metric}_mean"] = mean(values)
    return summary


JudgeFn = Callable[[str, str, str, str], Dict[str, int]]

RETRY_WAIT_S = 30.0
RATE_LIMIT_ATTEMPTS = 4


def is_rate_limit(exc: Exception) -> bool:
    """429 transitorio de la API del LLM (el único fallo con reintento)."""
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def query_with_backoff(engine: Any, question: str) -> Dict[str, Any]:
    """Consulta con espera creciente SOLO ante rate limit; otros fallos propagan."""
    last: Optional[Exception] = None
    for attempt in range(RATE_LIMIT_ATTEMPTS):
        try:
            return engine.query(question)
        except Exception as exc:  # noqa: BLE001 — se reintenta únicamente 429
            if not is_rate_limit(exc) or attempt == RATE_LIMIT_ATTEMPTS - 1:
                raise
            last = exc
            wait = RETRY_WAIT_S * (attempt + 1)
            print(f"[reintento] rate limit; esperando {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise last  # inalcanzable: el bucle siempre retorna o relanza


def evaluate_engine(
    engine: Any,
    questions: List[Dict[str, Any]],
    judge_fn: Optional[JudgeFn] = None,
) -> List[Dict[str, Any]]:
    """Ejecuta el motor por pregunta; ``judge_fn`` (opcional) puntúa 1-5."""
    rows: List[Dict[str, Any]] = []
    for question in questions:
        result = query_with_backoff(engine, question["question"])
        row = {
            "id": question["id"],
            "question": question["question"],
            "scores": score_generation(question, result.get("raw_documents", [])),
            "latency_ms": result.get("latency_ms", 0.0),
            "answer_chars": len(result.get("answer", "")),
            "answer": result.get("answer", ""),
            "judge": None,
        }
        if judge_fn is not None:
            context = "\n---\n".join(
                d.page_content for d in result.get("raw_documents", [])
            )
            row["judge"] = judge_fn(
                question["question"], result.get("answer", ""),
                context, question.get("note", ""),
            )
        rows.append(row)
        print(f"Evaluada {question['id']}", flush=True)
    return rows


def extract_judge_scores(text: str) -> Dict[str, int]:
    """Extrae faithfulness/relevance 1-5 tolerando markdown, eco o comillas simples."""
    values: Dict[str, int] = {}
    for key in ("faithfulness", "relevance"):
        match = re.search(rf"{key}\D{{0,5}}?(\d{{1,2}})", text, re.IGNORECASE)
        if match:
            values[key] = max(1, min(5, int(match.group(1))))
    if len(values) != 2:
        raise ValueError(f"puntuaciones no encontradas en: {text[:120]!r}")
    return values


def make_judge(llm: Any, retries: int = 2) -> JudgeFn:
    """LLM-as-judge con parse tolerante; un fallo devuelve ``None`` por campo.

    Reintenta ante formato inválido (con recordatorio de contrato) y con
    backoff SOLO ante rate limit; los campos nunca se inventan.
    """
    repair = (
        "\nRecuerda: responde EXACTAMENTE un JSON con las claves "
        '"faithfulness" y "relevance", valores enteros del 1 al 5.'
    )

    def judge(question: str, answer: str, context: str, ground_truth: str) -> Dict[str, int]:
        last_error: Optional[Exception] = None
        text = ""
        prompt = JUDGE_USER.format(
            question=question, context=context[:6000], answer=answer,
            ground_truth=ground_truth[:2000],
        )
        for attempt in range(retries + 1):
            try:
                raw = llm.invoke(prompt + (repair if attempt else ""))
                text = raw.content if hasattr(raw, "content") else str(raw)
                return extract_judge_scores(text)
            except Exception as exc:  # noqa: BLE001 — el juez es best effort
                last_error = exc
                if attempt < retries and is_rate_limit(exc):
                    wait = RETRY_WAIT_S * (attempt + 1)
                    print(f"[judge] rate limit; esperando {wait:.0f}s", flush=True)
                    time.sleep(wait)
        print(f"[judge] sin puntuación: {last_error} | texto: {text[:160]!r}",
              flush=True)
        return {"faithfulness": None, "relevance": None}

    return judge


def build_engine(settings: Settings) -> Any:
    """Motor real apuntando a la colección elegida del manifest verificado."""
    from src.indexing.vectorstore import get_vectorstore
    from src.rag.engine import RAGEngine
    from src.rag.pipeline import RAGPipeline, get_llm
    from src.retrieval.retriever import get_retriever

    vectorstore = get_vectorstore(settings)
    retriever = get_retriever(top_k=8, score_threshold=None, vectorstore=vectorstore)
    pipeline = RAGPipeline(retriever=retriever, llm=get_llm())
    return RAGEngine(pipeline=pipeline)


def start_eval_run(args: argparse.Namespace, manifest_sha: str, benchmark_sha: str) -> None:
    """Run de MLflow con tags y métricas agregadas del experimento."""
    import mlflow

    mlflow.start_run(run_name="rag-eval")
    mlflow.set_tags({
        "eval.script": "scripts/evaluate_rag_mlflow",
        "eval.manifest_sha256": manifest_sha,
        "eval.benchmark_sha256": benchmark_sha,
        "eval.collection": args.collection,
        "eval.judge": bool(args.judge),
        "eval.limit": args.limit or 0,
    })


def log_eval_metrics(summary: Dict[str, Any]) -> None:
    import mlflow

    metrics = {
        "questions": summary["questions"],
        "mean_latency_ms": summary["mean_latency_ms"],
        "mean_answer_chars": summary["mean_answer_chars"],
    }
    for metric in ("document_hit", "page_hit"):
        if summary[metric]["rate"] is not None:
            metrics[f"{metric}_rate"] = summary[metric]["rate"]
            metrics[f"{metric}_evaluated"] = summary[metric]["evaluated"]
    for metric in ("faithfulness_mean", "relevance_mean"):
        if metric in summary:
            metrics[metric] = summary[metric]
    mlflow.log_metrics(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path,
                        default=PROJECT_ROOT / "data/eval/benchmark.json")
    parser.add_argument("--collection", type=str, required=True,
                        help="Colección a evaluar (selección explícita; no hay ganadora)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluar solo las primeras N preguntas (0 = todas)")
    parser.add_argument("--judge", action="store_true",
                        help="Puntuar faithfulness/relevance con LLM-as-judge (Groq)")
    parser.add_argument("--output", type=str, default="",
                        help="Directorio de salida (por defecto: <run>/evaluations/<ts>)")
    args = parser.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI", "").strip():
        raise SystemExit(
            "MLFLOW_TRACKING_URI no definido: la evaluación se registra en MLflow "
            "(p. ej. export MLFLOW_TRACKING_URI=sqlite:///mlflow.db)"
        )

    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest["status"] != "verified" or not manifest["collections"]:
        raise ValueError("Se requiere una indexación completa y verificada")
    if args.collection not in manifest["collections"]:
        raise ValueError(
            f"Colección {args.collection!r} no existe en el manifest "
            f"(disponibles: {sorted(manifest['collections'])})"
        )
    documents = {d["doc_id"]: d["pages"] for d in manifest["report"]["documents"]}
    questions, benchmark_sha = load_questions(args.benchmark, documents)
    if args.limit:
        questions = questions[: args.limit]

    settings = Settings(_env_file=None, embedding_model=manifest["model"],
        embedding_max_tokens=manifest["report"]["max_input_tokens"],
        embedding_passage_prefix=manifest["report"]["passage_prefix"],
        chroma_dir=manifest["directory"], chroma_collection=args.collection)
    engine = build_engine(settings)

    judge_fn = None
    if args.judge:
        from src.rag.pipeline import get_llm as _get_llm
        judge_fn = make_judge(_get_llm())

    import mlflow
    from src.common.tracing import DEFAULT_EXPERIMENT

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"].strip())
    mlflow.set_experiment(DEFAULT_EXPERIMENT)
    start_eval_run(
        args,
        manifest_sha=sha256(manifest_raw).hexdigest(),
        benchmark_sha=benchmark_sha,
    )
    try:
        rows = evaluate_engine(engine, questions, judge_fn=judge_fn)
        summary = summarize_generation(rows)
        log_eval_metrics(summary)
    finally:
        mlflow.end_run()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(manifest_raw).hexdigest(),
        "benchmark_sha256": benchmark_sha,
        "collection": args.collection,
        "judge": bool(args.judge),
        "summary": summary,
        "rows": rows,
    }
    output = (
        Path(args.output)
        if args.output
        else args.manifest.parent / "evaluations"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    doc, page = summary["document_hit"], summary["page_hit"]
    print("\n=== Resumen de evaluación (MLflow: run rag-eval) ===")
    print(f"Preguntas        : {summary['questions']}")
    print(f"Documento hit    : {doc['hits']}/{doc['evaluated']}"
          + (f" ({doc['rate']:.0%})" if doc["rate"] is not None else ""))
    print(f"Página hit       : {page['hits']}/{page['evaluated']}"
          + (f" ({page['rate']:.0%})" if page["rate"] is not None else ""))
    print(f"Latencia media   : {summary['mean_latency_ms']:.0f} ms")
    if "faithfulness_mean" in summary:
        print(f"Faithfulness     : {summary['faithfulness_mean']:.2f} / 5")
        print(f"Relevance        : {summary['relevance_mean']:.2f} / 5")
    print(f"Resultados: {output}")


if __name__ == "__main__":
    main()
