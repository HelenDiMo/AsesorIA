"""Genera el informe estático de MLflow para ``docs/informe_mlflow.md``.

Lee el experimento ``asesoria-rag`` del almacén de tracking (por defecto
``sqlite:///mlflow.db`` o ``MLFLOW_TRACKING_URI``) y produce la página de
documentación con:

* runs de evaluación RAG (aciertos, latencias, juez, tokens y coste)
* runs de la suite de pruebas (``test-suite``), si existen
* notas de método: precios de Groq aplicados y límites de la demo

La página se commitea generada para que el sitio publicado (GitHub Pages)
no dependa de un servidor MLflow. Regenerar tras cada evaluación:

    python -m scripts.export_mlflow_report
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "informe_mlflow.md"
EXPERIMENT = "asesoria-rag"


def fetch_runs(uri: str) -> List[Any]:
    """Runs del experimento, del más reciente al más antiguo."""
    import mlflow

    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise SystemExit(
            f"No existe el experimento {EXPERIMENT!r} en {uri}. "
            "Ejecuta primero una evaluación con MLFLOW_TRACKING_URI definido."
        )
    return client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )


def _fmt_date(start_time_ms: int) -> str:
    return datetime.fromtimestamp(
        start_time_ms / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")


def _hit(metrics: Dict[str, float], rate_key: str, ev_key: str) -> str:
    rate, evaluated = metrics.get(rate_key), metrics.get(ev_key)
    if rate is None or evaluated is None:
        return "—"
    return f"{rate:.0%} ({round(rate * evaluated)}/{int(evaluated)})"


def _num(value: Any) -> str:
    return "—" if value is None else str(int(value))


def _money(value: Any) -> str:
    return "—" if value is None else f"${value:.4f}"


def _judge(tags: Dict[str, str], metrics: Dict[str, float]) -> str:
    if tags.get("eval.judge") != "True":
        return "—"
    faith, rel = metrics.get("faithfulness_mean"), metrics.get("relevance_mean")
    if faith is None or rel is None:
        return "—"
    return f"{faith:.2f} / {rel:.2f}"


def eval_table(runs: List[Any]) -> List[str]:
    rows = [
        r for r in runs
        if r.info.run_name == "rag-eval" and r.info.status == "FINISHED"
        and r.data.metrics.get("questions") is not None
    ]
    if not rows:
        return ["_Sin runs de evaluación finalizados todavía._", ""]
    lines = [
        "| Fecha | Preguntas | Documento | Página | Latencia media | "
        "Tokens (entrada / salida) | Coste | Juez F / R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in rows:
        m, t = run.data.metrics, run.data.tags
        prompt, completion = m.get("prompt_tokens_total"), m.get("completion_tokens_total")
        tokens = "—" if prompt is None else f"{int(prompt)} / {int(completion or 0)}"
        lines.append(
            f"| {_fmt_date(run.info.start_time)} | {_num(m.get('questions'))} "
            f"| {_hit(m, 'document_hit_rate', 'document_hit_evaluated')} "
            f"| {_hit(m, 'page_hit_rate', 'page_hit_evaluated')} "
            f"| {_num(m.get('mean_latency_ms'))} ms | {tokens} "
            f"| {_money(m.get('cost_usd_total'))} | {_judge(t, m)} |"
        )
    return lines + [""]


def tests_table(runs: List[Any]) -> List[str]:
    rows = [
        r for r in runs
        if r.info.run_name == "test-suite" and r.info.status == "FINISHED"
    ]
    if not rows:
        return ["_Sin runs de la suite en MLflow todavía._", ""]
    lines = [
        "| Fecha | Total | Pasadas | Fallidas | Skipped | Duración | Commit |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in rows:
        m, t = run.data.metrics, run.data.tags
        lines.append(
            f"| {_fmt_date(run.info.start_time)} | {_num(m.get('tests_total'))} "
            f"| {_num(m.get('tests_passed'))} | {_num(m.get('tests_failed'))} "
            f"| {_num(m.get('tests_skipped'))} | {m.get('tests_duration_s', 0):.0f} s "
            f"| `{t.get('test.git_commit') or '—'}` |"
        )
    return lines + [""]


def render(runs: List[Any], uri: str) -> str:
    finished = [r for r in runs if r.info.status == "FINISHED"]
    failed = len(runs) - len(finished)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Informe de evaluación y costes (MLflow)",
        "",
        "Resultados de las evaluaciones registradas en MLflow",
        f"(experimento `asesoria-rag`, almacén `{uri}`). Esta página se genera",
        "automáticamente; para ver la UI viva en local:",
        "",
        "```bash",
        "python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001",
        "```",
        "",
        "## Evaluación end-to-end (RAG completo)",
        "",
        "Runs `rag-eval` sobre el benchmark revisado: aciertos de evidencia",
        "(documento/página esperados), latencia media total, consumo de tokens",
        "del LLM y coste estimado. El juez LLM puntúa faithfulness / relevance",
        "de 1 a 5 cuando la evaluación se ejecuta con `--judge`.",
        "",
    ]
    parts += eval_table(finished)
    parts += [
        "## Suite de pruebas",
        "",
        "Run `test-suite` generado desde el informe JUnit de pytest",
        "(`python -m pytest tests/ -q --junitxml=results.xml` +",
        "`python -m scripts.log_test_results`).",
        "",
    ]
    parts += tests_table(finished)
    parts += [
        "## Método y límites",
        "",
        "- Coste = tokens × precio publicado por Groq para",
        "  `openai/gpt-oss-120b`: **$0,15 / M tokens de entrada** y",
        "  **$0,60 / M tokens de salida** (tags `eval.price_*` del run y",
        "  atributo `mlflow.llm.cost` en cada span de las trazas).",
        "- El detalle pregunta a pregunta queda como artefacto",
        "  `results.json` de cada run (MLflow → run → Artifacts).",
        "- La demo pública usa el free tier de Groq (≈ 200.000 tokens/día,",
        "  unas 46 consultas/día); los costes de esta tabla corresponden a",
        "  las evaluaciones, no a la demo desplegada.",
        f"- Página generada: {generated} · runs finalizados: {len(finished)}"
        + (f" · excluidos por error: {failed}" if failed else "") + ".",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Markdown de salida (docs/informe_mlflow.md)")
    args = parser.parse_args()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip() or "sqlite:///mlflow.db"
    runs = fetch_runs(uri)
    args.output.write_text(render(runs, uri), encoding="utf-8")
    print(f"Informe escrito: {args.output} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
