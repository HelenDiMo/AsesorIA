"""Evaluación E2E de la cadena RAG registrada en MLflow (generación).

Complementa a ``scripts.evaluate_retrieval.py`` (solo retrieval): aquí se
ejecuta el motor COMPLETO (retriever + prompt + LLM + rewriter) sobre el
benchmark revisado y se registran en un run de MLflow:

* ``document_hit_rate`` / ``page_hit_rate`` — la evidencia esperada del
  benchmark aparece entre los documentos recuperados (denominadores
  explícitos: los casos sin referencia no cuentan como fallos).
* ``mean_latency_ms`` / ``mean_retrieval_latency_ms`` /
  ``mean_generation_latency_ms`` — latencias medias por consulta.
* ``prompt_tokens_total`` / ``completion_tokens_total`` / ``llm_calls`` —
  consumo real de tokens del LLM (capturado con callbacks LCEL; incluye
  reescritura, generación y juez cuando procede).
* ``cost_usd_total`` / ``cost_usd_per_question`` — coste calculado con los
  precios publicados por Groq (tags ``eval.price_*``).
* ``faithfulness_mean`` / ``relevance_mean`` (con ``--judge``) — LLM-as-judge
  sobre la respuesta generada, puntuación 1-5.

El detalle completo (pregunta a pregunta) se sube como artefacto
``results.json`` del run. Cada consulta queda trazada como span ``rag.query``
si ``MLFLOW_TRACKING_URI`` está activo (src/common/tracing.py).

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

from langchain_core.callbacks import BaseCallbackHandler

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

# Groq, openai/gpt-oss-120b: precios publicados por el proveedor (USD / Mtok).
PRICING = {"prompt": 0.15, "completion": 0.60}
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"

RATE_LIMIT_RETRIES = 0

# Contador de uso fijado por ``capture_usage`` y consumido por ``build_engine``
# (permite que los tests sigan inyectando el motor con un solo parámetro).
_USAGE_COUNTER: Optional["TokenUsageCounter"] = None


def capture_usage(counter: Optional["TokenUsageCounter"]) -> None:
    """Fija el contador de tokens que ``build_engine`` vincula al LLM."""
    global _USAGE_COUNTER
    _USAGE_COUNTER = counter


class TokenUsageCounter(BaseCallbackHandler):
    """Acumula tokens de todas las invocaciones del LLM de la ejecución.

    Se inyecta con ``llm.with_config({"callbacks": [counter]})`` (LCEL), por
    lo que captura reescritura, generación y juez sin modificar ``src/``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage: Dict[str, Any] = (
            llm_output.get("token_usage") or llm_output.get("usage") or {}
        )
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        if not prompt and not completion:
            for generation in getattr(response, "generations", None) or []:
                for item in generation:
                    meta = getattr(getattr(item, "message", None), "usage_metadata", None) or {}
                    prompt += int(meta.get("input_tokens") or 0)
                    completion += int(meta.get("output_tokens") or 0)
        if prompt or completion:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self._record_on_active_span(prompt, completion)

    @staticmethod
    def _record_on_active_span(prompt: int, completion: int) -> None:
        """Anota uso y coste en el span activo (paneles Usage/Cost de MLflow).

        Nombres de atributo según la documentación oficial de MLflow
        (``mlflow.chat.tokenUsage`` / ``mlflow.llm.cost``). La observabilidad
        es best effort: un fallo nunca interrumpe la evaluación.
        """
        try:
            import mlflow

            span = mlflow.get_current_active_span()
            if span is None:
                return
            input_cost = prompt * PRICING["prompt"] / 1_000_000
            output_cost = completion * PRICING["completion"] / 1_000_000
            span.set_attribute("mlflow.chat.tokenUsage", {
                "input_tokens": prompt,
                "output_tokens": completion,
                "total_tokens": prompt + completion,
            })
            span.set_attribute("mlflow.llm.cost", {
                "input_cost": round(input_cost, 8),
                "output_cost": round(output_cost, 8),
                "total_cost": round(input_cost + output_cost, 8),
            })
        except Exception:  # noqa: BLE001
            pass

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens * PRICING["prompt"]
            + self.completion_tokens * PRICING["completion"]
        ) / 1_000_000


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
    retrieval = [r["retrieval_latency_ms"] for r in rows
                 if "retrieval_latency_ms" in r]
    generation = [r["generation_latency_ms"] for r in rows
                  if "generation_latency_ms" in r]
    summary["mean_retrieval_latency_ms"] = mean(retrieval) if retrieval else 0.0
    summary["mean_generation_latency_ms"] = mean(generation) if generation else 0.0
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
    global RATE_LIMIT_RETRIES
    last: Optional[Exception] = None
    for attempt in range(RATE_LIMIT_ATTEMPTS):
        try:
            return engine.query(question)
        except Exception as exc:  # noqa: BLE001 — se reintenta únicamente 429
            if not is_rate_limit(exc) or attempt == RATE_LIMIT_ATTEMPTS - 1:
                raise
            last = exc
            RATE_LIMIT_RETRIES += 1
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
        metrics = result.get("metrics") or {}
        row = {
            "id": question["id"],
            "question": question["question"],
            "scores": score_generation(question, result.get("raw_documents", [])),
            "latency_ms": result.get("latency_ms", 0.0),
            "retrieval_latency_ms": round(
                float(metrics.get("retrieval_latency_s", 0.0)) * 1000, 2
            ),
            "generation_latency_ms": round(
                float(metrics.get("generation_latency_s", 0.0)) * 1000, 2
            ),
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
    """Motor real apuntando a la colección elegida del manifest verificado.

    Si ``capture_usage`` fijó un contador previamente, se vinculan callbacks
    de uso de tokens al LLM (reescritura + generación) sin tocar ``src/``.
    """
    from src.indexing.vectorstore import get_vectorstore
    from src.rag.engine import RAGEngine
    from src.rag.pipeline import RAGPipeline, get_llm
    from src.retrieval.retriever import get_retriever

    vectorstore = get_vectorstore(settings)
    retriever = get_retriever(top_k=8, score_threshold=None, vectorstore=vectorstore)
    llm = get_llm()
    if _USAGE_COUNTER is not None:
        llm = llm.with_config({"callbacks": [_USAGE_COUNTER]})
    pipeline = RAGPipeline(retriever=retriever, llm=llm)
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
        "eval.llm_model": os.getenv("GROQ_MODEL") or DEFAULT_LLM_MODEL,
        "eval.price_prompt_usd_per_mtok": PRICING["prompt"],
        "eval.price_completion_usd_per_mtok": PRICING["completion"],
    })


def log_eval_metrics(summary: Dict[str, Any]) -> None:
    import mlflow

    metrics = {
        "questions": summary["questions"],
        "mean_latency_ms": summary["mean_latency_ms"],
        "mean_retrieval_latency_ms": summary.get("mean_retrieval_latency_ms", 0.0),
        "mean_generation_latency_ms": summary.get("mean_generation_latency_ms", 0.0),
        "mean_answer_chars": summary["mean_answer_chars"],
    }
    for metric in ("document_hit", "page_hit"):
        if summary[metric]["rate"] is not None:
            metrics[f"{metric}_rate"] = summary[metric]["rate"]
            metrics[f"{metric}_evaluated"] = summary[metric]["evaluated"]
    for metric in ("faithfulness_mean", "relevance_mean"):
        if metric in summary:
            metrics[metric] = summary[metric]
    usage = summary.get("usage") or {}
    metrics.update({
        "llm_calls": usage.get("llm_calls", 0),
        "prompt_tokens_total": usage.get("prompt_tokens", 0),
        "completion_tokens_total": usage.get("completion_tokens", 0),
        "rate_limit_retries": usage.get("rate_limit_retries", 0),
        "cost_usd_total": usage.get("cost_usd", 0.0),
        "cost_usd_per_question": (
            round(usage.get("cost_usd", 0.0) / summary["questions"], 6)
            if summary["questions"] else 0.0
        ),
    })
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
    counter = TokenUsageCounter()
    capture_usage(counter)
    engine = build_engine(settings)

    judge_fn = None
    if args.judge:
        from src.rag.pipeline import get_llm as _get_llm
        judge_fn = make_judge(_get_llm().with_config({"callbacks": [counter]}))

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
        summary["usage"] = {
            "llm_calls": counter.calls,
            "prompt_tokens": counter.prompt_tokens,
            "completion_tokens": counter.completion_tokens,
            "cost_usd": round(counter.cost_usd, 6),
            "rate_limit_retries": RATE_LIMIT_RETRIES,
        }
        output = (
            Path(args.output)
            if args.output
            else args.manifest.parent / "evaluations"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        output.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(manifest_raw).hexdigest(),
            "benchmark_sha256": benchmark_sha,
            "collection": args.collection,
            "judge": bool(args.judge),
            "pricing": PRICING,
            "summary": summary,
            "rows": rows,
        }
        (output / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_eval_metrics(summary)
        mlflow.log_artifact(str(output / "results.json"))
    finally:
        mlflow.end_run()

    doc, page = summary["document_hit"], summary["page_hit"]
    usage = summary["usage"]
    print("\n=== Resumen de evaluación (MLflow: run rag-eval) ===")
    print(f"Preguntas         : {summary['questions']}")
    print(f"Documento hit     : {doc['hits']}/{doc['evaluated']}"
          + (f" ({doc['rate']:.0%})" if doc["rate"] is not None else ""))
    print(f"Página hit        : {page['hits']}/{page['evaluated']}"
          + (f" ({page['rate']:.0%})" if page["rate"] is not None else ""))
    print(f"Latencia media    : {summary['mean_latency_ms']:.0f} ms"
          f" (retrieval {summary['mean_retrieval_latency_ms']:.0f} ms +"
          f" generación {summary['mean_generation_latency_ms']:.0f} ms)")
    if "faithfulness_mean" in summary:
        print(f"Faithfulness      : {summary['faithfulness_mean']:.2f} / 5")
        print(f"Relevance         : {summary['relevance_mean']:.2f} / 5")
    print(f"Resultados        : {output}")
    print("\n=== Consumo y coste LLM (registrados en MLflow) ===")
    print(f"Modelo            : {os.getenv('GROQ_MODEL') or DEFAULT_LLM_MODEL}")
    print(f"Llamadas LLM      : {usage['llm_calls']} (reescritura + generación"
          + (" + juez)" if args.judge else ")"))
    print(f"Tokens entrada    : {usage['prompt_tokens']}"
          f" (${PRICING['prompt']}/Mtok)")
    print(f"Tokens salida     : {usage['completion_tokens']}"
          f" (${PRICING['completion']}/Mtok)")
    per_question = usage["cost_usd"] / summary["questions"] if summary["questions"] else 0.0
    print(f"Coste total       : ${usage['cost_usd']:.6f}"
          f" (${per_question:.6f} / consulta)")
    print(f"Reintentos 429    : {usage['rate_limit_retries']}")


if __name__ == "__main__":
    main()
