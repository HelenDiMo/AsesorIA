"""
Módulo de orquestación del pipeline RAG para AsesorIA.
Conecta el retriever vectorial con las directrices de grounding estricto y el LLM,
asegurando trazabilidad documental (citas y fuentes).
"""

from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from src.retrieval.prompts import CHAT_QA_PROMPT


def format_docs(docs: List[Document]) -> str:
    """
    Formatea los fragmentos recuperados en un bloque de texto legible para el LLM,
    incluyendo de forma explícita los metadatos de origen (fuente y página).
    """
    if not docs:
        return "No se encontraron documentos relevantes."

    formatted_blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Documento desconocido")
        page = doc.metadata.get("page", "N/A")
        block = f"[Fragmento {i}] - Fuente: {source} (Pág. {page})\n{doc.page_content}"
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


class RAGPipeline:
    """
    Orquestador principal de AsesorIA.
    Integra el retriever, el prompt anti-alucinaciones y el LLM mediante LCEL.
    """

    def __init__(self, retriever: VectorStoreRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.llm = llm
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