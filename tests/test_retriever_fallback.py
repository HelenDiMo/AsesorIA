"""Banda de relajación del umbral (RELAXED_SCORE_FLOOR).

Si el umbral estricto no deja ningún documento autorizado, se admiten los
candidatos del suelo hacia arriba; por debajo del suelo se mantiene la
ausencia de documentos. Cubre consultas conversacionales del corpus real
(0.80-0.81) que con 0.82 estricto devolvían 0 documentos.
"""
from unittest.mock import Mock

from src.retrieval.retriever import RELAXED_SCORE_FLOOR, get_retriever


def _store(similarities):
    store = Mock()
    n = len(similarities)
    store.query.return_value = {
        'documents': [[f'texto-{i}' for i in range(n)]],
        'metadatas': [[{'source_scope': 'public', 'doc_id': f'doc-{i}'} for i in range(n)]],
        'distances': [[1.0 - s for s in similarities]],
    }
    return store


def test_floor_is_the_documented_value():
    assert RELAXED_SCORE_FLOOR == 0.795


def test_strict_threshold_emptied_relaxed_band_returns_candidates():
    # Caso real: «¿Qué modelo necesito para el alta?» (top-8 en 0.807-0.810).
    store = _store([0.8096, 0.8073])
    docs = get_retriever(score_threshold=0.82, vectorstore=store).invoke('Pregunta')
    assert [doc.metadata['doc_id'] for doc in docs] == ['doc-0', 'doc-1']


def test_candidates_below_floor_stay_empty():
    # Observado fuera de corpus: Madrid 0.7795, boda 0.7869.
    store = _store([0.7869, 0.7795, 0.75])
    assert get_retriever(score_threshold=0.82, vectorstore=store).invoke('Pregunta') == []


def test_strict_pass_short_circuits_the_band():
    store = _store([0.83, 0.80])
    docs = get_retriever(score_threshold=0.82, vectorstore=store).invoke('Pregunta')
    assert [doc.metadata['doc_id'] for doc in docs] == ['doc-0']


def test_no_candidates_never_rescues():
    store = _store([])
    assert get_retriever(score_threshold=0.82, vectorstore=store).invoke('Pregunta') == []


def test_unauthorized_candidates_are_not_rescued():
    store = Mock()
    store.query.return_value = {'documents': [['privado']],
                                'metadatas': [[{'source_scope': 'private', 'doc_id': 'p',
                                                'session_id': 'otra'}]],
                                'distances': [[0.15]]}
    assert get_retriever(score_threshold=0.82, vectorstore=store).invoke('Pregunta') == []
