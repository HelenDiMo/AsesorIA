"""Retriever vinculado a una sesión opcional: vectores explícitos -> documentos originales de LangChain."""
from __future__ import annotations

from copy import deepcopy
import math

from chromadb.api.types import validate_where
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig

from src.common.settings import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
from src.indexing.vectorstore import VectorStore, get_vectorstore

# Banda de relajación del umbral: si el umbral estricto no deja NINGÚN
# documento autorizado, se admiten los candidatos con similitud >= 0.795.
# En el corpus real, consultas conversacionales legítimas miden 0.80-0.81
# (con 0.82 estricto devolvían 0 documentos y la demo mostraba «sin
# información» en preguntas que sí están en el corpus); observaciones fuera
# de corpus quedan por debajo del suelo (<= 0.787) y siguen sin documentos.
# La generación conserva la política anti-alucinación por encima de esto.
RELAXED_SCORE_FLOOR = 0.795


class VectorRetriever(Runnable[str, list[Document]]):
    """Públicos y privados de una sesión ya validada por la aplicación."""

    def __init__(self, vectorstore: VectorStore, *, top_k: int = DEFAULT_TOP_K,
                 score_threshold: float | None = DEFAULT_SCORE_THRESHOLD, filter_dict: dict | None = None,
                 session_id: str | None = None):
        if session_id is not None:
            if not isinstance(session_id, str):
                raise TypeError("session_id debe ser una cadena o None")
            if not session_id.strip():
                raise ValueError("session_id no puede estar vacío")
        self._session_id = session_id
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

    @property
    def session_id(self) -> str | None:
        """Identidad de esta instancia; para otra sesión, crear otro retriever."""
        return self._session_id

    def _access_filter(self) -> dict:
        public = {"source_scope": {"$eq": "public"}}
        if self.session_id is None:
            return public
        return {"$or": [public, {"$and": [
            {"source_scope": {"$eq": "private"}},
            {"session_id": {"$eq": self.session_id}},
        ]}]}

    def _authorized(self, metadata: dict | None) -> bool:
        if not metadata:
            return False
        return metadata.get("source_scope") == "public" or (
            self.session_id is not None
            and metadata.get("source_scope") == "private"
            and metadata.get("session_id") == self.session_id
        )

    def invoke(self, question: str, config: RunnableConfig | None = None, **kwargs) -> list[Document]:
        """Una sola vectorización. config no se interpreta como autorización."""
        if not isinstance(question, str):
            raise TypeError("question debe ser una cadena")
        if not question.strip():
            raise ValueError("question no puede estar vacío")
        vector = self.vectorstore.embedding_service.embed_query(question)
        access = self._access_filter()
        # El filtro externo solo restringe el conjunto autorizado.
        where = {"$and": [access, deepcopy(self._filter)]} if self._filter else access
        result = self.vectorstore.query(vector, top_k=self.top_k, filter_dict=where)
        hits: list[tuple[Document, float]] = []
        for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0],
                                            result["distances"][0], strict=True):
            # Defensa adicional aunque el backend devolviera registros no autorizados.
            if not self._authorized(metadata):
                continue
            # Chroma cosine: distance = 1 - similarity. NO es una probabilidad.
            similarity = 1.0 - distance
            metadata = dict(metadata)
            # Alias de integración, sin modificar ChunkMetadata ni la colección.
            metadata.setdefault("source", metadata.get("source_url") or metadata["doc_id"])
            hits.append((Document(page_content=text, metadata=metadata), similarity))
        kept = [hit for hit in hits
                if self.score_threshold is None or hit[1] >= self.score_threshold]
        # Umbral estricto vacío -> banda de relajación (ver RELAXED_SCORE_FLOOR).
        if not kept and hits and hits[0][1] >= RELAXED_SCORE_FLOOR:
            kept = [hit for hit in hits if hit[1] >= RELAXED_SCORE_FLOOR]
        return [doc for doc, _ in kept]


def get_retriever(top_k: int = DEFAULT_TOP_K, score_threshold: float | None = DEFAULT_SCORE_THRESHOLD,
                  filter_dict: dict | None = None, *,
                  vectorstore: VectorStore | None = None,
                  session_id: str | None = None) -> VectorRetriever:
    """Sesión validada por el backend; sin sesión, solo públicos.

    No compartir esta instancia ni su pipeline entre sesiones.
    Defaults V1 evaluados en nuestro corpus; None desactiva únicamente el umbral.
    """
    return VectorRetriever(vectorstore if vectorstore is not None else get_vectorstore(),
                           top_k=top_k, score_threshold=score_threshold, filter_dict=filter_dict,
                           session_id=session_id)
