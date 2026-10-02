"""Retriever público: vectores explícitos -> documentos originales de LangChain."""
from __future__ import annotations

from copy import deepcopy
import math

from chromadb.api.types import validate_where
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig

from src.indexing.vectorstore import VectorStore, get_vectorstore


class VectorRetriever(Runnable[str, list[Document]]):
    """Solo corpus público hasta acordar contexto de sesión fiable con el pipeline."""

    def __init__(self, vectorstore: VectorStore, *, top_k: int = 4,
                 score_threshold: float | None = 0.5, filter_dict: dict | None = None):
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k debe ser un entero positivo")
        if score_threshold is not None and (
            isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float))
            or not math.isfinite(score_threshold) or not -1 <= score_threshold <= 1
        ):
            raise ValueError("score_threshold debe ser None o similitud cosine entre -1 y 1")
        if filter_dict is not None and not isinstance(filter_dict, dict):
            raise TypeError("filter_dict debe ser un diccionario Chroma")
        if filter_dict:
            validate_where(filter_dict)
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._filter = deepcopy(filter_dict)

    def invoke(self, question: str, config: RunnableConfig | None = None, **kwargs) -> list[Document]:
        """Una sola vectorización. config no se interpreta como autorización."""
        if not isinstance(question, str):
            raise TypeError("question debe ser una cadena")
        if not question.strip():
            raise ValueError("question no puede estar vacío")
        vector = self.vectorstore.embedding_service.embed_query(question)
        public = {"source_scope": {"$eq": "public"}}
        where = {"$and": [public, deepcopy(self._filter)]} if self._filter else public
        result = self.vectorstore.query(vector, top_k=self.top_k, filter_dict=where)
        documents = []
        for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0],
                                            result["distances"][0], strict=True):
            # Defensa adicional: no devolver registros sin un scope público explícito.
            if not metadata or metadata.get("source_scope") != "public":
                continue
            # Chroma cosine: distance = 1 - similarity. NO es una probabilidad.
            similarity = 1.0 - distance
            if self.score_threshold is not None and similarity < self.score_threshold:
                continue
            metadata = dict(metadata)
            # Alias de integración, sin modificar ChunkMetadata ni la colección.
            metadata.setdefault("source", metadata.get("source_url") or metadata["doc_id"])
            documents.append(Document(page_content=text, metadata=metadata))
        return documents


def get_retriever(top_k: int = 4, score_threshold: float | None = 0.5,
                  filter_dict: dict | None = None, *,
                  vectorstore: VectorStore | None = None) -> VectorRetriever:
    """k y threshold provisionales; None desactiva únicamente el umbral."""
    return VectorRetriever(vectorstore if vectorstore is not None else get_vectorstore(),
                           top_k=top_k, score_threshold=score_threshold, filter_dict=filter_dict)
