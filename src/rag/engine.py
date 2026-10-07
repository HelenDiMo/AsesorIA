"""
Módulo motor (Engine) para AsesorIA.
Proporciona la interfaz de alto nivel para interactuar con el sistema RAG,
gestionando la instanciación única (Singleton/Factory) y el formato estándar de salida.
"""

from typing import Any, Dict, Optional
from src.common.tracing import record, setup_tracing, trace_span
from src.rag.pipeline import RAGPipeline, get_llm
from src.retrieval.retriever import get_retriever


class RAGEngine:
    """
    Fachada principal para el consumo del pipeline RAG.
    Diseñada para ser consumida de forma directa por endpoints de FastAPI,
    interfaces de usuario (Streamlit) o scripts de evaluación.
    """

    def __init__(
        self,
        top_k: int = 8,
        score_threshold: Optional[float] = None,
        pipeline: Optional[RAGPipeline] = None,
    ):
        setup_tracing()  # no-op sin MLFLOW_TRACKING_URI (src/common/tracing.py)
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            retriever = get_retriever(top_k=top_k, score_threshold=score_threshold)
            llm = get_llm()
            self.pipeline = RAGPipeline(retriever=retriever, llm=llm)

    def query(self, question: str, history: Optional[list] = None) -> Dict[str, Any]:
        """
        Procesa una consulta de usuario a través del pipeline.

        ``history`` (opcional): turnos previos de la conversación actual
        (cronológicos, text-only, request-scoped). Si se proporciona, el
        pipeline resuelve las referencias de la pregunta (query rewriting)
        antes del retrieval. ``history`` ausente mantiene el comportamiento
        clásico pregunta-a-la-vez.

        Retorna un diccionario estructurado:
        - question (str): Pregunta original formulada.
        - answer (str): Respuesta generada por el modelo.
        - sources (list[dict]): Lista estructurada de metadatos de las fuentes usadas.
        - metrics (dict): Tiempos de latencia (retrieval, generation, total).
        - raw_documents (list): Documentos LangChain originales recuperados.

        Si MLFLOW_TRACKING_URI está definido, la consulta queda registrada en
        el span ``rag.query`` (src/common/tracing.py); sin esa variable el
        comportamiento y el coste son idénticos a los previos.
        """
        clean_question = question.strip()
        with trace_span(
            "rag.query",
            {"question": clean_question, "history_turns": len(history or [])},
        ) as span:
            if not clean_question:
                payload = {
                    "question": question,
                    "answer": "La consulta no puede estar vacía.",
                    "sources": [],
                    "metrics": {
                        "retrieval_latency_s": 0.0,
                        "generation_latency_s": 0.0,
                        "total_latency_s": 0.0,
                    },
                    "raw_documents": [],
                }
                record(span, outputs={"sources": 0, "answer_chars": 0})
                return payload

            result = self.pipeline.answer_query(clean_question, history=history)
            source_docs = result.get("source_documents", [])
            metrics = result.get("metrics", {})

            # Formateo amigable de fuentes para la API/UI
            formatted_sources = []
            for doc in source_docs:
                metadata = doc.metadata or {}
                source = metadata.get("source", "Documento desconocido")
                page_start = metadata.get("page")
                page_end = metadata.get("page_end", page_start)

                if page_start is not None:
                    if page_end is not None and page_end != page_start:
                        page_label = f"Págs. {page_start}–{page_end}"
                    else:
                        page_label = f"Pág. {page_start}"
                else:
                    page_label = "Pág. N/A"

                formatted_sources.append(
                    {
                        "source": source,
                        "page": page_label,
                        "snippet": doc.page_content[:200]
                        + ("..." if len(doc.page_content) > 200 else ""),
                    }
                )

            payload = {
                "question": clean_question,
                "answer": result.get("answer", ""),
                "sources": formatted_sources,
                "metrics": metrics,
                "latency_ms": round(metrics.get("total_latency_s", 0.0) * 1000, 2),
                "raw_documents": source_docs,
            }
            record(
                span,
                outputs={
                    "sources": len(formatted_sources),
                    "raw_documents": len(source_docs),
                    "answer_chars": len(payload["answer"]),
                },
                attributes={**metrics, "latency_ms": payload["latency_ms"]},
            )
            return payload


# Instancia singleton para reutilización global
_engine_instance: Optional[RAGEngine] = None


def get_rag_engine(
    top_k: int = 8, score_threshold: Optional[float] = None
) -> RAGEngine:
    """Retorna una instancia reutilizable del RAGEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(top_k=top_k, score_threshold=score_threshold)
    return _engine_instance
