"""Conversational RAG: history → query rewriting → retrieval → respuesta grounded.

Tests deterministas de Backend/RAG (scope `src/rag/**`): el rewriter, el
retriever y la generación se stubbean/espias — sin llamadas al LLM real.
Cobertura: retrocompatibilidad, history None/vacío, preguntas standalone y
contextuales, argumento EXACTO recibido por el retriever, grounding (el
historial nunca es fuente factual), caso negativo de aislamiento y la cadena
Q1–Q5 del proyecto.
"""

import inspect

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from src.rag import pipeline as pipeline_module
from src.rag.engine import RAGEngine
from src.rag.pipeline import (
    RAGPipeline,
    format_history_transcript,
    resolve_standalone_query,
)
from src.retrieval.prompts import CHAT_QA_PROMPT, QUERY_REWRITE_PROMPT

# Cadena conversacional del proyecto (docs/acceptance_criteria + handoff §10)
Q1 = "¿Cuándo hay que darse de alta como autónomo?"
Q2 = "¿Y si además tengo un trabajo por cuenta ajena?"
Q3 = "¿Entonces tengo que hacerlo presencial o puedo hacerlo por internet?"
Q4 = "¿Qué modelo tengo que presentar?"
Q5 = "¿Y cuándo tengo que presentarlo?"


def turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _docs() -> list[Document]:
    return [
        Document(
            page_content="Fragmento fiscal recuperado.",
            metadata={"source": "manual.pdf", "page": 3},
        )
    ]


class SpyRetriever:
    """Registra la query EXACTA que recibe el retriever."""

    def __init__(self, docs=None):
        self.calls: list[str] = []
        self.docs = _docs() if docs is None else docs

    def invoke(self, query):
        self.calls.append(query)
        return self.docs


class SpyGeneration:
    """Sustituye a generation_chain y registra su input."""

    def __init__(self, answer: str = "Respuesta grounded."):
        self.inputs: list[dict] = []
        self.answer = answer

    def invoke(self, inputs):
        self.inputs.append(inputs)
        return self.answer


class _NoRewrite:
    """Sentinela: si el pipeline intenta construir la cadena de reescritura,
    el test falla → demuestra que sin historial NO se invoca al LLM."""

    def __or__(self, other):
        raise AssertionError("reescritor invocado sin historial")


def make_pipeline(monkeypatch, answer: str = "Respuesta grounded."):
    """RAGPipeline con retriever y generación espiados (sin red)."""
    retriever = SpyRetriever()
    pipeline = RAGPipeline(
        retriever=retriever, llm=FakeListChatModel(responses=["stub"])
    )
    monkeypatch.setattr(pipeline, "generation_chain", SpyGeneration(answer))
    return pipeline, retriever, pipeline.generation_chain


def stub_rewriter(monkeypatch, mapping=None, calls=None):
    """Rewriter determinista (question → standalone) con contrato real:
    sin historial no registra ni reescribe (mismo fast-path que producción)."""
    mapping = mapping or {}

    def fake(llm, question, history):
        if not history:
            return question
        if calls is not None:
            calls.append({"question": question, "history": list(history)})
        return mapping.get(question, question)

    monkeypatch.setattr(pipeline_module, "resolve_standalone_query", fake)


def forbid_rewrite(monkeypatch):
    """Falla si se intenta cualquier reescritura (prueba de fast-path)."""
    monkeypatch.setattr(pipeline_module, "QUERY_REWRITE_PROMPT", _NoRewrite())


# --------------------------------------------------------------------------- #
# A — Retrocompatibilidad y firmas
# --------------------------------------------------------------------------- #


class TestBackwardCompatibility:
    def test_answer_query_signature_keeps_history_optional(self):
        sig = inspect.signature(RAGPipeline.answer_query)
        assert "question" in sig.parameters
        assert sig.parameters["history"].default is None

    def test_engine_query_signature_keeps_history_optional(self):
        sig = inspect.signature(RAGEngine.query)
        assert sig.parameters["history"].default is None

    def test_answer_query_without_history_still_works(self, monkeypatch):
        pipeline, retriever, generation = make_pipeline(monkeypatch)
        result = pipeline.answer_query("¿Cuándo se presenta el Modelo 303?")

        assert result["answer"] == "Respuesta grounded."
        assert retriever.calls == ["¿Cuándo se presenta el Modelo 303?"]
        assert generation.inputs[0]["question"] == "¿Cuándo se presenta el Modelo 303?"


# --------------------------------------------------------------------------- #
# B / C — history=None y history=[] equivalen a ausencia de contexto
# --------------------------------------------------------------------------- #


class TestNoHistory:
    def test_history_none_never_invokes_rewriter(self, monkeypatch):
        forbid_rewrite(monkeypatch)
        pipeline, retriever, _ = make_pipeline(monkeypatch)

        pipeline.answer_query("¿Qué gastos son deducibles?", history=None)

        assert retriever.calls == ["¿Qué gastos son deducibles?"]

    def test_history_empty_list_equivalent_to_absence(self, monkeypatch):
        forbid_rewrite(monkeypatch)
        pipeline, retriever, _ = make_pipeline(monkeypatch)

        pipeline.answer_query("¿Qué gastos son deducibles?", history=[])

        assert retriever.calls == ["¿Qué gastos son deducibles?"]

    def test_resolve_with_empty_history_returns_original_without_llm(self, monkeypatch):
        forbid_rewrite(monkeypatch)
        question = "Pregunta autónoma"
        assert (
            resolve_standalone_query(
                FakeListChatModel(responses=["x"]), question, []
            )
            == question
        )


# --------------------------------------------------------------------------- #
# D — Pregunta standalone no se deforma
# --------------------------------------------------------------------------- #


class TestStandaloneQuestion:
    def test_rewriter_prompt_keeps_standalone_questions_intact(self):
        messages = QUERY_REWRITE_PROMPT.format_messages(
            history="Usuario: ¿Cuándo se presenta el Modelo 303?",
            question="¿Cuándo se presenta el Modelo 303?",
        )
        assert len(messages) == 2
        system = messages[0].content
        assert "prácticamente intacta" in system
        assert "No inventes" in system
        assert "SOLO cuando la conversación previa lo permita" in system

    def test_standalone_question_reaches_retriever_unchanged(self, monkeypatch):
        stub_rewriter(monkeypatch)  # rewriter que decide no alterar nada
        pipeline, retriever, _ = make_pipeline(monkeypatch)
        history = [turn("user", Q1), turn("assistant", "El alta es obligatoria.")]

        pipeline.answer_query(Q1, history=history)

        assert retriever.calls == [Q1]  # byte-idéntica


# --------------------------------------------------------------------------- #
# E / F — Pregunta anafórica + el retriever recibe la standalone query
# --------------------------------------------------------------------------- #


class TestContextualRewriting:
    STANDALONE_Q3 = "darse de alta como autónomo presencialmente o por internet"

    def test_anaphoric_question_is_resolved_with_history(self, monkeypatch):
        calls = []
        stub_rewriter(monkeypatch, {Q3: self.STANDALONE_Q3}, calls=calls)
        pipeline, retriever, _ = make_pipeline(monkeypatch)
        history = [
            turn("user", Q1),
            turn("assistant", "El alta como autónomo es obligatoria."),
            turn("user", Q2),
            turn("assistant", "Compatible con trabajo por cuenta ajena."),
        ]

        pipeline.answer_query(Q3, history=history)

        # F: argumento EXACTO del retriever
        assert retriever.calls == [self.STANDALONE_Q3]
        assert Q3 not in retriever.calls
        # el rewriter recibió la pregunta actual y el historial completo
        assert calls == [{"question": Q3, "history": history}]

    def test_generation_receives_resolved_query_not_raw_history(self, monkeypatch):
        stub_rewriter(monkeypatch, {Q3: self.STANDALONE_Q3})
        pipeline, _, generation = make_pipeline(monkeypatch)
        history = [turn("user", Q1), turn("assistant", "Respuesta previa.")]

        pipeline.answer_query(Q3, history=history)

        inputs = generation.inputs[0]
        assert set(inputs) == {"documents", "question"}
        assert inputs["question"] == self.STANDALONE_Q3


# --------------------------------------------------------------------------- #
# G — Grounding: el historial NO es fuente factual
# --------------------------------------------------------------------------- #


class TestGrounding:
    def test_generation_prompt_has_no_history_slot(self):
        assert set(CHAT_QA_PROMPT.input_variables) == {"context", "question"}

    def test_history_never_reaches_generation_input(self, monkeypatch):
        stub_rewriter(
            monkeypatch, {Q5: "plazo para presentar el modelo de alta"}
        )
        pipeline, retriever, generation = make_pipeline(monkeypatch)
        history = [
            turn("user", Q4),
            turn("assistant", "El modelo aplicable es el D017."),
            turn("user", "¿Y con un 37% de actividad?"),
            turn("assistant", "Cifra no incluida en el corpus."),
        ]

        pipeline.answer_query(Q5, history=history)

        serialized = str(generation.inputs[0])
        assert "37%" not in serialized  # hecho del historial jamás se inyecta
        assert "D017" not in serialized
        assert generation.inputs[0]["documents"] == retriever.docs
        assert "history" not in generation.inputs[0]

    def test_empty_rewrite_falls_back_to_original_query(self):
        # LLM devuelve blanco → consulta original (conservadora, no inventada)
        question = Q3
        rewritten = resolve_standalone_query(
            FakeListChatModel(responses=["   "]),
            question,
            [turn("user", Q1)],
        )
        assert rewritten == question

    def test_rewrite_failure_propagates_without_inventing_query(self):
        def boom(_):
            raise RuntimeError("llm no disponible")

        llm = RunnableLambda(boom)
        with pytest.raises(RuntimeError, match="llm no disponible"):
            resolve_standalone_query(
                llm, Q3, [turn("user", Q1)]
            )


# --------------------------------------------------------------------------- #
# H — Referencia no resoluble: no inventar contexto
# --------------------------------------------------------------------------- #


class TestUnsupportedContext:
    def test_unresolvable_reference_keeps_original_query(self, monkeypatch):
        # historial que NO permite resolver "lo" de Q3 → se conserva Q3
        unrelated = [turn("user", "¿Qué es el IVA?"), turn("assistant", "Impuesto.")]
        stub_rewriter(monkeypatch)  # default: devuelve la pregunta tal cual
        pipeline, retriever, _ = make_pipeline(monkeypatch)

        pipeline.answer_query(Q3, history=unrelated)

        assert retriever.calls == [Q3]
        assert "autónomo" not in retriever.calls[0]


# --------------------------------------------------------------------------- #
# I — Caso negativo de aislamiento: conversación nueva + Q3
# --------------------------------------------------------------------------- #


class TestNegativeIsolation:
    def test_q3_alone_never_imports_previous_conversation(self, monkeypatch):
        forbid_rewrite(monkeypatch)  # sin historial → ni rewriter ni LLM
        pipeline, retriever, _ = make_pipeline(monkeypatch)

        pipeline.answer_query(Q3)

        assert retriever.calls == [Q3]
        assert "autónomo" not in retriever.calls[0]
        assert "cuenta ajena" not in retriever.calls[0]


# --------------------------------------------------------------------------- #
# Q1–Q5 — Cadena contextual completa (mockeada)
# --------------------------------------------------------------------------- #


class TestQ1Q5Chain:
    RESOLVED = {
        Q1: Q1,
        Q2: "darse de alta como autónomo teniendo trabajo por cuenta ajena",
        Q3: "darse de alta como autónomo presencialmente o por internet",
        Q4: "modelo de alta de autónomo a presentar",
        Q5: "cuándo presentar el modelo de alta de autónomo",
    }

    def test_context_is_established_extended_and_kept(self, monkeypatch):
        rewrite_calls: list[dict] = []
        stub_rewriter(monkeypatch, self.RESOLVED, calls=rewrite_calls)
        pipeline, retriever, generation = make_pipeline(monkeypatch)

        history: list[dict] = []
        for question in (Q1, Q2, Q3, Q4, Q5):
            pipeline.answer_query(question, history=list(history))
            history.append(turn("user", question))
            history.append(turn("assistant", generation.answer))

        # Retrieval encadena las consultas resueltas (Q1 sin historial = original)
        assert retriever.calls == [
            self.RESOLVED[Q1],
            self.RESOLVED[Q2],
            self.RESOLVED[Q3],
            self.RESOLVED[Q4],
            self.RESOLVED[Q5],
        ]
        # El rewriter nunca se invocó en Q1 (sin historial) y en cada turno
        # posterior recibió TODO el contexto acumulado (2, 4, 6, 8 turnos)
        assert [len(c["history"]) for c in rewrite_calls] == [2, 4, 6, 8]
        assert rewrite_calls[0]["question"] == Q2  # contexto ampliado
        assert rewrite_calls[1]["question"] == Q3  # referencia resuelta
        assert rewrite_calls[2]["question"] == Q4  # contexto mantenido
        assert rewrite_calls[3]["question"] == Q5  # contexto mantenido
        # Q1 aparece en el historial que vio el rewriter de Q3
        assert any(t["content"] == Q1 for t in rewrite_calls[1]["history"])

    def test_q3_in_new_conversation_has_no_context(self, monkeypatch):
        forbid_rewrite(monkeypatch)
        pipeline, retriever, _ = make_pipeline(monkeypatch)

        pipeline.answer_query(Q3)  # conversación NUEVA, sin historial

        assert retriever.calls == [Q3]


# --------------------------------------------------------------------------- #
# Historial: contrato text-only, cronológico, request-scoped
# --------------------------------------------------------------------------- #


class TestHistoryHandling:
    def test_transcript_is_chronological_and_text_only(self):
        history = [
            turn("user", Q1),
            turn("assistant", "Obligatorio."),
            {"role": "system", "content": "ignorado"},  # rol no soportado
            {"role": "user", "content": ""},  # vacío → fuera
            "no-dict",  # no dict → fuera
        ]
        transcript = format_history_transcript(history)
        assert transcript == f"Usuario: {Q1}\nAsistente: Obligatorio."

    def test_transcript_with_none_or_empty_is_blank(self):
        assert format_history_transcript(None) == ""
        assert format_history_transcript([]) == ""

    def test_resolve_with_unusable_turns_returns_original(self, monkeypatch):
        forbid_rewrite(monkeypatch)
        question = "Pregunta"
        history = [{"role": "system", "content": "x"}, "junk"]
        assert (
            resolve_standalone_query(
                FakeListChatModel(responses=["x"]), question, history
            )
            == question
        )

    def test_answer_query_does_not_mutate_history(self, monkeypatch):
        stub_rewriter(monkeypatch)
        pipeline, _, _ = make_pipeline(monkeypatch)
        history = [turn("user", Q1)]
        snapshot = [dict(t) for t in history]

        pipeline.answer_query(Q3, history=history)

        assert history == snapshot


# --------------------------------------------------------------------------- #
# Transporte end-to-end (adapter real → engine → pipeline) — seam §10
# --------------------------------------------------------------------------- #


class StubPipeline:
    """Pipeline de engine: registra la llamada real de answer_query."""

    def __init__(self):
        self.calls: list[dict] = []

    def answer_query(self, question, history=None):
        self.calls.append({"question": question, "history": history})
        return {"answer": "ok", "source_documents": []}


class TestEngineAndTransport:
    def test_engine_forwards_history_to_pipeline(self):
        stub = StubPipeline()
        engine = RAGEngine(pipeline=stub)

        engine.query("hola", history=[turn("user", Q1)])

        assert stub.calls == [{"question": "hola", "history": [turn("user", Q1)]}]

    def test_engine_without_history_passes_none(self):
        stub = StubPipeline()
        engine = RAGEngine(pipeline=stub)

        engine.query("hola")

        assert stub.calls == [{"question": "hola", "history": None}]

    def test_engine_empty_question_skips_pipeline(self):
        stub = StubPipeline()
        engine = RAGEngine(pipeline=stub)

        result = engine.query("   ", history=[turn("user", Q1)])

        assert stub.calls == []
        assert result["answer"] == "La consulta no puede estar vacía."

    @pytest.mark.asyncio
    async def test_adapter_transports_history_into_engine(self):
        from ui.rag_adapter import RagAdapter

        stub = StubPipeline()
        engine = RAGEngine(pipeline=stub)
        adapter = RagAdapter(backend=engine)
        history = [turn("user", Q1), turn("assistant", "Respuesta previa.")]

        resp = await adapter.ask(Q5, [], history=history)

        # el seam del adapter activó history= porque query() lo acepta
        assert stub.calls == [{"question": Q5, "history": history}]
        assert resp.answer == "ok"
