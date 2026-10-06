"""Development MOCK — simulated answers to exercise the UI without a backend.

⚠️ This module is NOT the real RAG: it fetches no documents and uses no LLM.
Its only purpose is to develop and demo the interface (and every state)
before `src/rag/engine.py` exists.

Deterministic: the same question always produces the same answer.
Keyword scenarios (see the ui/ README):

- "iva" + "deducible"      → answer + 2 fully documented sources
- "303"                    → answer + 3 sources (several documents)
- "alta" + "autónomo"      → answer + 1 source
- "cripto" / "bitcoin"     → grounded=False, sources=[]
- "metadata" / "incompleto"→ sources with incomplete metadata
- "error" / "fallo"        → raises RuntimeError (UI error state)
- anything else            → generic answer + 1 source
"""

from __future__ import annotations

import asyncio
import random
from typing import List

try:
    from .contracts import RAGResponse, Source
except ImportError:
    from contracts import RAGResponse, Source

# Simulated latency just to make the processing state visible.
SIMULATED_LATENCY_MS = (400, 900)


def _simulate_latency() -> float:
    return random.uniform(*SIMULATED_LATENCY_MS) / 1000.0


async def _delay() -> None:
    await asyncio.sleep(_simulate_latency())


def _make_source(
    document: str,
    content: str,
    page: int | None = None,
    section: str | None = None,
    score: float | None = None,
    metadata: dict | None = None,
) -> Source:
    return Source(
        document=document,
        content=content,
        page=page,
        section=section,
        score=score,
        metadata=metadata,
    )


class MockRAG:
    """Simulated, deterministic backend for UI development and demo."""

    async def ask(self, question: str, documents: List[str]) -> RAGResponse:
        """Returns a simulated RAGResponse based on the incoming question."""
        await _delay()

        q = question.strip().lower()

        # 1️⃣ Normal answer with two sources (IVA deducible)
        if "iva" in q and "deducible" in q:
            return RAGResponse(
                answer=(
                    "El IVA soportado en la adquisición de bienes y servicios "
                    "vinculados con la actividad económica es deducible siempre "
                    "que se cumplan los requisitos de exigibilidad y se posea "
                    "la correspondiente factura.\n\n"
                    "En la práctica, guarda facturas con IVA siempre que el "
                    "gasto esté relacionado con tu actividad y no esté "
                    "excluido expresamente por la normativa."
                ),
                sources=[
                    _make_source(
                        document="manual-practico-iva-2025.pdf",
                        content=(
                            "El IVA soportado es deducible cuando se refiere a "
                            "operaciones gravadas y se cumple el requisito de "
                            "exigibilidad."
                        ),
                        page=42,
                        section="Deducción del IVA soportado",
                        score=0.94,
                    ),
                    _make_source(
                        document="BOE-A-1992-28740-consolidado-37-1992.pdf",
                        content=(
                            "Se considerará deducible el IVA soportado que "
                            "recurra en bienes y servicios utilizados para "
                            "realizar operaciones sujetas al impuesto."
                        ),
                        page=18,
                        section="Artículo 95",
                        score=0.89,
                    ),
                ],
                grounded=True,
                latency_ms=_simulate_latency() * 1000,
            )

        # 2️⃣ Several sources: three different documents (modelo 303)
        if "303" in q or "autoliquidaci" in q:
            return RAGResponse(
                answer=(
                    "El modelo 303 es la autoliquidación del IVA: en él se "
                    "declaran las cuotas devengadas y las soportadas, "
                    "resultando el importe a ingresar o a compensar.\n\n"
                    "El periodo de liquidación habitual es trimestral, con "
                    "presentación dentro de los 20 primeros días naturales "
                    "del mes siguiente al cierre de cada trimestre (mesa "
                    "especial de febrero, mayo, septiembre y enero)."
                ),
                sources=[
                    _make_source(
                        document="modelo-303.pdf",
                        content=(
                            "Este modelo se utiliza para la liquidación del "
                            "impuesto sobre el valor añadido, consignando las "
                            "cuotas devengadas y soportadas."
                        ),
                        page=1,
                        section="Autoliquidación de IVA",
                        score=0.96,
                    ),
                    _make_source(
                        document="manual-practico-iva-2025.pdf",
                        content=(
                            "Los periodos de liquidación ordinarios son "
                            "trimestrales y se liquidan en los veinte primeros "
                            "días naturales del mes siguiente al cierre."
                        ),
                        page=57,
                        section="Periodos de liquidación",
                        score=0.91,
                    ),
                    _make_source(
                        document="Obligaciones_formales__contables_y_registrales.pdf",
                        content=(
                            "Los sujetos pasivos están obligados a presentar "
                            "las autoliquidaciones correspondientes en el "
                            "plazo reglamentariamente establecido."
                        ),
                        page=9,
                        section="Obligaciones de presentación",
                        score=0.85,
                    ),
                ],
                grounded=True,
                latency_ms=_simulate_latency() * 1000,
            )

        # 3️⃣ Answer with a single source (autónomo registration)
        if "alta" in q and "aut" in q:
            return RAGResponse(
                answer=(
                    "Para darse de alta como trabajador autónomo es necesario "
                    "presentar el modelo 036 o 037 ante la Agencia Tributaria "
                    "y causar alta en el Régimen Especial de Trabajadores "
                    "Autónomos (RETA) en la Seguridad Social."
                ),
                sources=[
                    _make_source(
                        document="Alta_en_trabajo_autonomo.pdf",
                        content=(
                            "El alta se realiza mediante la presentación del "
                            "modelo 036 o 037 y la correspondiente afiliación "
                            "al RETA."
                        ),
                        page=5,
                        section="Procedimiento de alta",
                        score=0.97,
                    ),
                ],
                grounded=True,
                latency_ms=_simulate_latency() * 1000,
            )

        # 4️⃣ Not enough information (grounded=False, no sources)
        if "cripto" in q or "bitcoin" in q or "blockchain" in q:
            return RAGResponse(
                answer="",
                sources=[],
                grounded=False,
                no_answer_reason=(
                    "la documentación cargada no incluye información "
                    "sobre criptomonedas"
                ),
                latency_ms=_simulate_latency() * 1000,
            )

        # 5️⃣ Controlled "backend" error (to exercise the error state)
        if "error" in q or "fallo" in q or "excepción" in q or "excepcion" in q:
            raise RuntimeError("Simulated RAG engine error (development mock)")

        # 6️⃣ Sources with incomplete metadata (UI robustness)
        if "metadata" in q or "incompleto" in q or "incompleta" in q:
            return RAGResponse(
                answer=(
                    "Este escenario de prueba devuelve fuentes con metadatos "
                    "incompletos para verificar que la interfaz no se rompe."
                ),
                sources=[
                    _make_source(
                        document="documento_sin_pagina.pdf",
                        content="Este fragmento no tiene número de página asociado.",
                        page=None,
                        section="Introducción",
                        score=None,
                    ),
                    _make_source(
                        document="fragmento_sin_seccion.pdf",
                        content="Este fragmento no especifica la sección de origen.",
                        page=12,
                        section=None,
                        score=0.75,
                    ),
                    _make_source(
                        document="documento_sin_metadatos.pdf",
                        content="Fragmento sin página, sección ni relevancia.",
                        page=None,
                        section=None,
                        score=None,
                    ),
                ],
                grounded=True,
                latency_ms=_simulate_latency() * 1000,
            )

        # 7️⃣ Default generic answer (1 source)
        return RAGResponse(
            answer=(
                "Según la documentación disponible, puedo ofrecerte una "
                "respuesta general sobre tu consulta. Para un detalle más "
                "preciso, prueba a formular la pregunta con el impuesto o "
                "trámite concreto (por ejemplo, IVA, IRPF, modelo 303…)."
            ),
            sources=[
                _make_source(
                    document="consulta_general.pdf",
                    content=(
                        "Respuesta de muestra del modo demostración: la "
                        "interfaz funciona, pero el motor RAG real aún no "
                        "está conectado."
                    ),
                    page=1,
                    section="Nota de demostración",
                    score=0.6,
                ),
            ],
            grounded=True,
            latency_ms=_simulate_latency() * 1000,
        )
