"""Development MOCK — simulated answers to exercise the UI without a backend.

⚠️ This module is NOT the real RAG: it fetches no documents and uses no LLM.
Its only purpose is to develop and demo the interface (and every state)
before `src/rag/engine.py` exists.

Deterministic: the same question always produces the same answer.
Keyword scenarios (see the ui/ README):

- "iva" + "deducible"      → answer + 2 fully documented sources
- "303"                    → answer + 3 sources (several documents)
- "alta" + "autónomo"      → answer + 1 source
- "gastos" + "deducible"   → answer + 2 sources (IRPF expense deduction)
- "irpf"                   → answer + 2 sources (self-employed IRPF)
- "iva" (general)          → answer + 2 sources (what IVA is, devengo/support)
- "obligaciones"           → answer + 2 sources (fiscal duties overview)
- "ayuda" / "guía"         → orientation answer, no sources (UX fallback)
- "cripto" / "bitcoin"     → grounded=False, sources=[]
- "metadata" / "incompleto"→ sources with incomplete metadata
- "error" / "fallo"        → raises RuntimeError (UI error state)
- anything else            → generic answer + 1 source

Every simulated answer is identified as demo (a short 🧠 footer), and the
demo never invents a relevance score: ``score`` stays ``None`` in every
simulated source, so the interface only shows «relevancia» when the real
backend provides a value with validated semantics (cosine similarity).

Responses are immediate: the mock adds NO artificial latency, so the
processing state only appears while there is real work to do.
"""

from __future__ import annotations

from typing import List

try:
    from .contracts import RAGResponse, Source
except ImportError:
    from contracts import RAGResponse, Source

_DEMO_FOOTER = "\n\n*🧠 Demo: respuesta simulada por el modo demostración.*"


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


def _visible_names(documents: List[str], labels: List[str] | None) -> List[str]:
    """Original file names of the session (never storage paths)."""
    pool = labels if labels is not None else documents
    names: List[str] = []
    for item in pool or []:
        text = str(item)
        if "/" in text or "\\" in text:
            continue  # storage path → not displayable
        names.append(text)
    return names


def _bind_to_uploaded(sources: List[Source], names: List[str]) -> List[Source]:
    """Binds every simulated source to a document the user actually loaded.

    Demo rule: never cite a document that is not in the session (the mock
    keeps its simulated content/pages but the file must exist). With no
    names available (unit tests), the simulated names are kept as-is.
    """
    if not names:
        return sources
    used: set = set()
    for src in sources:
        if src.document in names:
            used.add(src.document)
            continue
        candidates = [n for n in names if n not in used] or names
        src.document = candidates[0]
        used.add(src.document)
    return sources


class MockRAG:
    """Simulated, deterministic backend for UI development and demo."""

    async def ask(
        self,
        question: str,
        documents: List[str],
        labels: List[str] | None = None,
    ) -> RAGResponse:
        """Returns a simulated RAGResponse coherent with the loaded documents.

        Args:
            question: the user's question.
            documents: session documents (paths are ignored by the mock).
            labels: original visible file names; every simulated source is
                bound to one of them so the demo never cites a file the
                user has not uploaded.
        """
        response = self._simulate(question)
        response.sources = _bind_to_uploaded(
            response.sources, _visible_names(documents, labels)
        )
        # Product-realistic structure, but always identified as demo.
        if response.grounded and response.answer:
            response.answer = response.answer.rstrip() + _DEMO_FOOTER
        return response

    def _simulate(self, question: str) -> RAGResponse:
        """Keyword scenarios (see the module docstring); immediate, no delay."""
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
                    ),
                ],
                grounded=True,
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
                    ),
                ],
                grounded=True,
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
                    ),
                ],
                grounded=True,
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
            )

        # 7️⃣ Deductible business expenses (IRPF orientation)
        if "gasto" in q and "deduc" in q:
            return RAGResponse(
                answer=(
                    "Dentro del IRPF son deducibles los gastos necesarios "
                    "para la actividad: cuota de autónomos, suministros, "
                    "seguros profesionales, mantenimiento de equipos, "
                    "servicios contables y, con límites, el uso de la "
                    "vivienda y el vehículo.\n\n"
                    "Requisito práctico: que el gasto esté vinculado a la "
                    "actividad, figure en tus libros y se justifique con "
                    "factura."
                ),
                sources=[
                    _make_source(
                        document="Obligaciones_formales__contables_y_registrales.pdf",
                        content=(
                            "Son deducibles los gastos necesarios para la "
                            "obtención de los ingresos, siempre que queden "
                            "justificados documentalmente."
                        ),
                        page=7,
                        section="Gastos deducibles de la actividad",
                    ),
                    _make_source(
                        document="manual-practico-iva-2025.pdf",
                        content=(
                            "Las facturas justificativas deben conservarse "
                            "durante el plazo de prescripción del impuesto."
                        ),
                        page=31,
                        section="Justificación de gastos",
                    ),
                ],
                grounded=True,
            )

        # 8️⃣ IRPF for the self-employed (payments on account)
        if "irpf" in q:
            return RAGResponse(
                answer=(
                    "El IRPF es un impuesto personal sobre la renta. Como "
                    "autónomo realizas pagos fraccionados a cuenta (modelo "
                    "130 con carácter general, o 131 en estimación objetiva) "
                    "y presentas la declaración anual.\n\n"
                    "En la práctica: declara todos los ingresos de la "
                    "actividad, aplica los gastos deducibles necesarios para "
                    "obtenerlos y revisa cada trimestre el resultado del "
                    "pago fraccionado."
                ),
                sources=[
                    _make_source(
                        document="Obligaciones_formales__contables_y_registrales.pdf",
                        content=(
                            "Los contribuyentes en estimación directa ingresan "
                            "a cuenta mediante el modelo 130; en estimación "
                            "objetiva, mediante el modelo 131."
                        ),
                        page=14,
                        section="Pagos fraccionados del IRPF",
                    ),
                    _make_source(
                        document="declaracion_renta.pdf",
                        content=(
                            "La declaración anual del IRPF se presenta entre "
                            "junio y julio, por medios telemáticos."
                        ),
                        page=63,
                        section="Declaración anual",
                    ),
                ],
                grounded=True,
            )

        # 9️⃣ General IVA orientation (devengado / soportado)
        if "iva" in q:
            return RAGResponse(
                answer=(
                    "El IVA es un impuesto indirecto al consumo: el "
                    "empresario recauda la cuota devengada (IVA cobrado en "
                    "tus facturas) y la cuota soportada (IVA pagado en tus "
                    "gastos), y la diferencia es el importe a ingresar o a "
                    "compensar en el modelo 303.\n\n"
                    "Para tu actividad: emite factura con IVA en las "
                    "operaciones sujetas y deduce el IVA soportado de los "
                    "gastos vinculados a la actividad, conservando siempre "
                    "la factura original."
                ),
                sources=[
                    _make_source(
                        document="manual-practico-iva-2025.pdf",
                        content=(
                            "El IVA soportado es deducible cuando se refiere "
                            "a operaciones gravadas y se cumple el requisito "
                            "de exigibilidad."
                        ),
                        page=42,
                        section="Deducción del IVA soportado",
                    ),
                    _make_source(
                        document="modelo-303.pdf",
                        content=(
                            "En este modelo se consignan las cuotas "
                            "devengadas y las soportadas, con resultado a "
                            "ingresar o a compensar."
                        ),
                        page=1,
                        section="Autoliquidación de IVA",
                    ),
                ],
                grounded=True,
            )

        # 🔟 Fiscal obligations overview
        if "obligacion" in q:
            return RAGResponse(
                answer=(
                    "Como autónomo tus obligaciones fiscales principales "
                    "son: darte de alta (modelo 036/037), llevar libros de "
                    "contabilidad, emitir y conservar facturas, presentar "
                    "los pagos fraccionados del IRPF (modelos 130/131) y, "
                    "si estás en IVA, presentar el modelo 303 trimestral y "
                    "el resumen anual 390.\n\n"
                    "También hay obligaciones censales y, si contratas "
                    "servicios con retención, obligaciones de retener e "
                    "ingresar a cuenta."
                ),
                sources=[
                    _make_source(
                        document="Obligaciones_formales__contables_y_registrales.pdf",
                        content=(
                            "Los obligados deben llevar libros de contabilidad "
                            "y conservar facturas y justificantes."
                        ),
                        page=3,
                        section="Obligaciones formales",
                    ),
                    _make_source(
                        document="modelo-303.pdf",
                        content=(
                            "La autoliquidación del IVA se presenta dentro "
                            "de los veinte primeros días naturales del mes "
                            "siguiente al cierre del periodo."
                        ),
                        page=2,
                        section="Plazos de presentación",
                    ),
                ],
                grounded=True,
            )

        # 1️⃣1️⃣ Help / orientation (fallback if it ever reaches the mock)
        if "ayuda" in q or "guia" in q or "guía" in q:
            return RAGResponse(
                answer=(
                    "Puedo ayudarte a consultar la documentación que tengas "
                    "cargada. Por ejemplo: alta de autónomo, IVA y modelo "
                    "303, IRPF, gastos deducibles y obligaciones fiscales.\n\n"
                    "Escribe tu pregunta con tus palabras y la contrasto "
                    "con la documentación disponible."
                ),
                sources=[],
                grounded=True,
            )

        # 1️⃣2️⃣ Default generic answer (1 source)
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
                        "Fragmento de muestra del modo demostración: "
                        "contenido simulado para ejercitar la interfaz."
                    ),
                    page=1,
                    section="Nota de demostración",
                ),
            ],
            grounded=True,
        )
