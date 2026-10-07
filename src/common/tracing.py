"""Tracing opcional con MLflow para la cadena RAG (observabilidad).

Contrato:

* Sin ``MLFLOW_TRACKING_URI`` definido → **cero efecto**: no se importa
  mlflow, no se crean spans y el coste por consulta es una comprobación
  de variable de entorno.
* Con ``MLFLOW_TRACKING_URI`` definido → cada consulta genera el span
  raíz ``rag.query`` (pregunta, turnos de historial, salidas y latencias)
  y, cuando hay historial, un span anidado ``rag.rewrite`` con la consulta
  autónoma resultante.
* La observabilidad nunca interrumpe el serving: cualquier fallo de mlflow
  se registra como warning y la consulta continúa sin trazar.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
DEFAULT_EXPERIMENT = "asesoria-rag"

# Estado por proceso (setup idempotente):
#   configured: ya se intentó configurar (evita reintentos en cada span)
#   usable:     la configuración terminó bien (mlflow importable y seteado)
_state: Dict[str, bool] = {"configured": False, "usable": False}


def tracing_enabled() -> bool:
    """True solo si MLFLOW_TRACKING_URI está definido (off por defecto)."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def setup_tracing() -> None:
    """Configura uri y experimento una sola vez; idempotente y a prueba de
    fallos: si mlflow no puede configurarse se avisa y se sigue sin trazar.
    """
    if _state["configured"] or not tracing_enabled():
        return
    _state["configured"] = True
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"].strip())
        mlflow.set_experiment(os.environ.get(EXPERIMENT_ENV) or DEFAULT_EXPERIMENT)
        _state["usable"] = True
    except Exception:
        logger.warning("MLflow no disponible; tracing desactivado", exc_info=True)


def _safe(fn, *args, **kwargs):
    """Ejecuta una llamada de anotación sin propagar errores (best effort)."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.warning("MLflow: anotación de span no registrada", exc_info=True)
        return None


def record(
    span: Any,
    outputs: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Adjunta salidas/atributos a un span; ``span`` puede ser ``None``."""
    if span is None:
        return
    if outputs is not None:
        _safe(span.set_outputs, outputs)
    if attributes:
        _safe(span.set_attributes, attributes)


@contextmanager
def trace_span(
    name: str, inputs: Optional[Dict[str, Any]] = None
) -> Iterator[Optional[Any]]:
    """Abre el span ``name`` si el tracing está habilitado; si no, cede ``None``.

    El cuerpo se ejecuta SIEMPRE: si mlflow no puede abrir el span, el cuerpo
    continúa sin trazar en lugar de romper. Las excepciones del cuerpo se
    propagan (y el span queda cerrado con ese error).
    """
    if not tracing_enabled():
        yield None
        return
    setup_tracing()
    if not _state["usable"]:
        yield None
        return
    import mlflow

    try:
        span_cm = mlflow.start_span(name=name)
        span = span_cm.__enter__()
    except Exception:
        logger.warning("MLflow: no se pudo iniciar el span %r", name, exc_info=True)
        yield None
        return
    if inputs:
        _safe(span.set_inputs, inputs)
    try:
        yield span
    finally:
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            span_cm.__exit__(exc_info[0], exc_info[1], exc_info[2])
        else:
            span_cm.__exit__(None, None, None)
