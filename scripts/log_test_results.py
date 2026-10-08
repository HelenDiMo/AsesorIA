"""Registra la suite de pruebas (pytest) como run de MLflow.

Lee el informe JUnit XML generado por pytest y vuelca en el run:

* ``tests_total`` / ``tests_passed`` / ``tests_failed`` / ``tests_errors`` /
  ``tests_skipped`` / ``duration_s``
* tags: intérprete de Python, plataforma, commit git (mejor esfuerzo)
* artefacto: el propio XML del informe

Uso:
    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    python -m pytest tests/ -q --junitxml=results.xml
    python -m scripts.log_test_results --junit results.xml
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
from xml.etree import ElementTree


def parse_junit(path: Path) -> Dict[str, Any]:
    """Suma atributos de todos los <testsuite> (raíz testsuite o testsuites)."""
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    if not suites:
        raise ValueError(f"Sin <testsuite> en {path}")
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    for suite in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            totals[key] += int(suite.get(key, 0) or 0)
        totals["time"] += float(suite.get("time", 0.0) or 0.0)
    totals["passed"] = (
        totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    )
    return totals


def git_commit() -> str:
    """Commit actual (mejor esfuerzo; vacío si no hay repo o git)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — el SHA es opcional
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True,
                        help="Informe JUnit XML de pytest (--junitxml)")
    parser.add_argument("--name", type=str, default="test-suite",
                        help="Nombre del run de MLflow")
    args = parser.parse_args()

    if not args.junit.is_file():
        raise SystemExit(f"No existe el informe: {args.junit}")
    if not os.environ.get("MLFLOW_TRACKING_URI", "").strip():
        raise SystemExit(
            "MLFLOW_TRACKING_URI no definido: la suite se registra en MLflow "
            "(p. ej. export MLFLOW_TRACKING_URI=sqlite:///mlflow.db)"
        )

    totals = parse_junit(args.junit)

    import mlflow
    from src.common.tracing import DEFAULT_EXPERIMENT

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"].strip())
    mlflow.set_experiment(DEFAULT_EXPERIMENT)
    with mlflow.start_run(run_name=args.name) as run:
        mlflow.set_tags({
            "test.script": "scripts/log_test_results",
            "test.python": platform.python_version(),
            "test.platform": sys.platform,
            "test.git_commit": git_commit(),
        })
        mlflow.log_metrics({
            "tests_total": totals["tests"],
            "tests_passed": totals["passed"],
            "tests_failed": totals["failures"],
            "tests_errors": totals["errors"],
            "tests_skipped": totals["skipped"],
            "tests_duration_s": round(totals["time"], 3),
        })
        mlflow.log_artifact(str(args.junit))

    print("\n=== Suite de pruebas (MLflow: run " + args.name + ") ===")
    print(f"Total             : {totals['tests']}")
    print(f"Pasadas           : {totals['passed']}")
    print(f"Fallidas          : {totals['failures']}")
    print(f"Errores           : {totals['errors']}")
    print(f"Skipped           : {totals['skipped']}")
    print(f"Duración (pytest) : {totals['time']:.1f} s")
    print(f"Run MLflow        : {run.info.run_id}")


if __name__ == "__main__":
    main()
