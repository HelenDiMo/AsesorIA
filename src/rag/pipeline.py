"""
Módulo de orquestación del pipeline RAG para AsesorIA.
Conecta el retriever vectorial con las directrices de grounding estricto y el LLM,
asegurando trazabilidad documental (citas y fuentes).
"""

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from src.common.settings import get_settings
from src.retrieval.prompts import CHAT_QA_PROMPT


def get_llm() -> BaseChatModel:
    """
    Instancia el LLM según la configuración del entorno (Settings).
    Soporta 'anthropic' y 'ollama'.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()

    if provider == "anthropic":
        from langchain_community.chat_models import ChatAnthropic

        model_name = settings.llm_model or "claude-3-5-sonnet-latest"
        return ChatAnthropic(
            model=model_name,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        model_name = settings.llm_model or "qwen2.5:7b"
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model_name,
            temperature=settings.llm_temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        model_name = settings.llm_model or "gpt-4o-mini"
        return ChatOpenAI(
            model=model_name,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    else:
        raise ValueError(
            f"Proveedor de LLM no soportado: '{provider}'. "
            "Valores válidos en .env: 'anthropic', 'ollama'."
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
        # Si no se pasa un LLM explícito (como en producción/UI), se instancia desde settings
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

    def answer_query(self, question: str) -> Dict[str, Any]:
        """
        Ejecuta el ciclo RAG completo para una pregunta dada.

        Retorna un diccionario con:
        - 'answer': Respuesta generada por el LLM.
        - 'source_documents': Lista de documentos/fragmentos recuperados con metadatos.
        """
        # 1. Recuperar fragmentos relevantes mediante el retriever
        retrieved_docs = self.retriever.invoke(question)

        # 2. Generar respuesta condicionada al contexto inyectado
        generated_answer = self.generation_chain.invoke(
            {"documents": retrieved_docs, "question": question}
        )

        return {
            "answer": generated_answer,
            "source_documents": retrieved_docs,
        }
