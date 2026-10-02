"""Corrección del contrato invoke; sin LLM ni evaluación de relevancia fiscal."""
from unittest.mock import Mock

from langchain_core.documents import Document
import pytest

from src.retrieval.retriever import get_retriever
from tests.test_vectorstore import make_chunk, service, store


def seed(store, service):
    service.embed_documents.side_effect = lambda texts: [[1, 0], [0.6, 0.8], [-1, 0]]
    store.add_chunks([make_chunk('one'), make_chunk('two', page=9, page_end=9),
                      make_chunk('three', source_url='')])


def test_invoke_original_documents_top_k_and_single_raw_query(store, service):
    seed(store, service)
    retriever = get_retriever(top_k=2, score_threshold=None, vectorstore=store)
    docs = retriever.invoke('¿Qué gastos puedo deducir?')
    service.embed_query.assert_called_once_with('¿Qué gastos puedo deducir?')
    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert [doc.metadata['doc_id'] for doc in docs] == ['one', 'two']
    assert docs[0].page_content == 'Texto original fiscal'
    assert docs[0].metadata['source'] == 'https://example.test/manual'
    assert (docs[0].metadata['page'], docs[0].metadata['page_end']) == (7, 8)
    assert docs[0].metadata['section_label'] == 'Deducciones'


@pytest.mark.parametrize('threshold,expected', [(None, 3), (0.5, 2), (0.7, 1), (-1, 3)])
def test_threshold_is_cosine_similarity_not_distance(store, service, threshold, expected):
    seed(store, service)
    docs = get_retriever(5, threshold, vectorstore=store).invoke('Pregunta')
    assert len(docs) == expected
    if threshold is None:
        assert docs[-1].metadata['source'] == 'three'


def test_inclusive_threshold_boundary_and_empty_result():
    store = Mock()
    store.query.return_value = {'documents': [['original']], 'metadatas': [[{'source_scope': 'public',
         'doc_id': 'doc', 'page': 1, 'page_end': 2}]], 'distances': [[0.5]]}
    assert len(get_retriever(score_threshold=0.5, vectorstore=store).invoke('Pregunta')) == 1
    assert get_retriever(score_threshold=0.51, vectorstore=store).invoke('Pregunta') == []


def test_metadata_filter_and_defensive_copy(store, service):
    seed(store, service)
    filter_dict = {'$and': [{'doc_id': {'$eq': 'two'}}, {'page_end': {'$gte': 9}}]}
    retriever = get_retriever(filter_dict=filter_dict, vectorstore=store)
    filter_dict.clear()
    assert [doc.metadata['doc_id'] for doc in retriever.invoke('Pregunta')] == ['two']


def test_public_policy_cannot_be_bypassed_by_filter(store, service):
    store.add_chunks([make_chunk('public'), make_chunk('private_a', source_scope='private', session_id='a'),
                      make_chunk('private_b', source_scope='private', session_id='b')])
    for where in (None, {'$or': [{'source_scope': 'public'}, {'source_scope': 'private'}]}):
        docs = get_retriever(filter_dict=where, vectorstore=store).invoke('Pregunta')
        assert [doc.metadata['doc_id'] for doc in docs] == ['public']
    assert get_retriever(filter_dict={'session_id': 'a'}, vectorstore=store).invoke('Pregunta') == []
    assert get_retriever(filter_dict={'source_scope': 'private'}, vectorstore=store).invoke('Pregunta') == []
    # This proves public-only exclusion, NOT authenticated private-session access.


def test_empty_collection_returns_no_documents(store):
    assert get_retriever(vectorstore=store).invoke('Pregunta') == []


@pytest.mark.parametrize('kwargs', [{'top_k': 0}, {'top_k': True}, {'top_k': 1.5},
    {'score_threshold': float('nan')}, {'score_threshold': 1.1}, {'score_threshold': True}])
def test_invalid_search_configuration(store, kwargs):
    with pytest.raises(ValueError):
        get_retriever(vectorstore=store, **kwargs)


def test_invalid_filter(store):
    with pytest.raises(ValueError):
        get_retriever(filter_dict={'page': {'$invalid': 2}}, vectorstore=store)


def test_runnable_composition_contract(store, service):
    store.add_chunks([make_chunk()])
    chain = get_retriever(vectorstore=store) | (lambda docs: docs[0].page_content)
    assert chain.invoke('Pregunta') == 'Texto original fiscal'


@pytest.mark.parametrize('question,error', [('', ValueError), (' \n', ValueError), (None, TypeError)])
def test_invalid_question_does_not_embed(store, service, question, error):
    with pytest.raises(error):
        get_retriever(vectorstore=store).invoke(question)
    service.embed_query.assert_not_called()


def test_missing_or_private_scope_is_not_returned_even_from_backend():
    store = Mock()
    store.query.return_value = {'documents': [['unknown', 'private']],
        'metadatas': [[{'doc_id': 'a'}, {'doc_id': 'b', 'source_scope': 'private'}]],
        'distances': [[0.0, 0.0]]}
    assert get_retriever(vectorstore=store).invoke('Pregunta') == []


@pytest.mark.parametrize('has_documents', [True, False])
def test_real_pipeline_connection(store, service, has_documents):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from src.rag.pipeline import RAGPipeline, format_docs

    chunk = make_chunk()
    if has_documents:
        store.add_chunks([chunk])
    pipeline = RAGPipeline(retriever=get_retriever(vectorstore=store),
                           llm=FakeListChatModel(responses=['Respuesta simulada']))
    with_prompt = Mock(wraps=pipeline.generation_chain.invoke)
    pipeline.generation_chain = Mock(invoke=with_prompt)
    result = pipeline.answer_query('Pregunta sin prefijo')
    assert result['answer'] == 'Respuesta simulada'
    service.embed_query.assert_called_once_with('Pregunta sin prefijo')
    assert with_prompt.call_args.args[0]['question'] == 'Pregunta sin prefijo'
    assert with_prompt.call_args.args[0]['documents'] == result['source_documents']
    if has_documents:
        doc = result['source_documents'][0]
        assert doc.page_content == chunk.text
        assert doc.metadata['source'] == chunk.metadata.source_url
        assert (doc.metadata['page'], doc.metadata['page_end']) == (7, 8)
        context = format_docs(result['source_documents'])
        assert 'Págs. 7–8' in context
        assert chunk.text in context
        assert chunk.embedding_text not in context
    else:
        assert result['source_documents'] == []
        assert format_docs([]) == 'No se encontraron documentos relevantes.'
