"""
Módulo de orquestación del pipeline RAG para AsesorIA.
Conecta el retriever vectorial con las directrices de grounding estricto y el LLM,
asegurando trazabilidad documental (citas y fuentes).
"""

import os
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

from src.common.tracing import record, trace_span
from src.retrieval.prompts import CHAT_QA_PROMPT, QUERY_REWRITE_PROMPT


def get_llm(provider: str | None = None, model: str | None = None):
    """Instancia directa del cliente oficial de Groq."""
    import os
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    # Usa llama-3.1-8b-instant (disponible siempre en free tier) o llama-3.1-70b-versatile
    model_name = model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"

    return ChatGroq(
        model_name=model_name,
        groq_api_key=api_key,
        temperature=0.0,
    )


def _format_page_reference(metadata: Dict[str, Any]) -> str:
    """Formatea la referencia de página considerando rangos (page y page_end)."""
    page_start = metadata.get("page")
    page_end = metadata.get("page_end")

    if page_start is None:
        return "Pág. N/A"

    if page_end is not None and page_end != page_start:
        return f"Págs. {page_start}–{page_end}"

    return f"Pág. {page_start}"


def format_docs(docs: List[Document]) -> str:
    """
    Formatea los fragmentos recuperados en un bloque de texto legible para el LLM,
    incluyendo de forma explícita los metadatos de origen (fuente y página/rango).
    """
    if not docs:
        return "No se encontraron documentos relevantes."

    formatted_blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Documento desconocido")
        page_str = _format_page_reference(doc.metadata)
        block = f"[Fragmento {i}] - Fuente: {source} ({page_str})\n{doc.page_content}"
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


def format_history_transcript(history: Optional[List[Any]]) -> str:
    """Transcripción texto del historial para el reescritor de consultas.

    Solo contexto conversacional: entradas cronológicas con role en
    {user, assistant} y contenido utilizable; el resto se descarta.
    """
    lines: List[str] = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        label = "Usuario" if role == "user" else "Asistente"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def resolve_standalone_query(
    llm: BaseChatModel,
    question: str,
    history: Optional[List[Any]] = None,
) -> str:
    """question + history → consulta autónoma para el retrieval.

    Sin historial (None/vacío/sin turnos utilizables) devuelve la pregunta
    original sin invocar al LLM. Con historial, el reescritor resuelve las
    referencias usando SOLO la conversación provista; si el resultado viene
    vacío se conserva la pregunta original (consulta conservadora, nunca
    inventada). Un fallo del LLM se propaga sin enmascarar.

    Con MLFLOW_TRACKING_URI definido, cada reescritura real genera el span
    ``rag.rewrite`` (consulta autónoma resultante); los fast-paths sin LLM
    no generan spans.
    """
    if not history:
        return question
    transcript = format_history_transcript(history)
    if not transcript:
        return question
    with trace_span(
        "rag.rewrite",
        {"question": question, "history_turns": len(history)},
    ) as span:
        chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        rewritten = str(
            chain.invoke({"history": transcript, "question": question})
        ).strip()
        standalone = rewritten or question
        record(span, outputs={"standalone_query": standalone})
        return standalone


class RAGPipeline:
    """
    Orquestador principal de AsesorIA.
    Integra el retriever, el prompt anti-alucinaciones y el LLM mediante LCEL.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: Optional[BaseChatModel] = None,
    ):
        self.retriever = retriever
        # Si no se pasa un LLM explícito, se instancia con get_llm
        self.llm = llm if llm is not None else get_llm()
        self._build_chain()

    def _build_chain(self):
        """Construye la cadena LCEL para generación de respuestas."""
        self.generation_chain = (
            {
                "context": lambda x: format_docs(x["documents"]),
                "question": lambda x: x["question"],
            }
            | CHAT_QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def answer_query(
        self, question: str, history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el ciclo RAG completo para una pregunta dada,
        midiendo la latencia de retrieval y de generación.

        ``history`` (opcional, cronológico, text-only, request-scoped): turnos
        previos de la conversación actual. Se usa ÚNICAMENTE para resolver la
        consulta autónoma antes del retrieval (query rewriting); nunca entra en
        el prompt de generación ni sustituye a los documentos recuperados
        (los documentos siguen siendo la única fuente factual).

        Retorna un diccionario con:
        - 'answer': Respuesta generada por el LLM.
        - 'source_documents': Lista de fragmentos recuperados con metadatos.
        - 'metrics': Diccionario con latencias en segundos.
        """

        import time

        # 0. Resolución contextual: question + history → consulta autónoma
        #    (sin historial: identidad, sin llamada al LLM)
        standalone_query = resolve_standalone_query(self.llm, question, history)

        # 1. Medir latencia de recuperación (ChromaDB)
        t_retrieval_start = time.perf_counter()

        # 1. Recuperar fragmentos relevantes con la consulta resuelta
        retrieved_docs = self.retriever.invoke(standalone_query)

        t_retrieval = round(time.perf_counter() - t_retrieval_start, 3)

        # 2. Medir latencia de generación (Groq LLM)
        t_generation_start = time.perf_counter()

        # 2. Generar respuesta condicionada al contexto inyectado
        #    (solo documentos recuperados + consulta; el historial no participa)
        generated_answer = self.generation_chain.invoke(
            {"documents": retrieved_docs, "question": standalone_query}
        )

        t_generation = round(time.perf_counter() - t_generation_start, 3)
        total_latency = round(t_retrieval + t_generation, 3)

        return {
            "answer": generated_answer,
            "source_documents": retrieved_docs,
            "metrics": {
                "retrieval_latency_s": t_retrieval,
                "generation_latency_s": t_generation,
                "total_latency_s": total_latency,
            },
        }
