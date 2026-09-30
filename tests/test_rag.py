"""
Pruebas unitarias para validar las plantillas de prompts de AsesorIA.
"""

import pytest
from src.retrieval.prompts import CHAT_QA_PROMPT, STANDALONE_QA_PROMPT, SYSTEM_PROMPT


def test_system_prompt_contains_critical_rules():
    """Verifica que el prompt de sistema contenga las directrices clave."""
    assert "Grounding estricto" in SYSTEM_PROMPT
    assert "Mitigación de alucinaciones" in SYSTEM_PROMPT
    assert "Trazabilidad y citas" in SYSTEM_PROMPT
    assert "No dispongo de suficiente información" in SYSTEM_PROMPT


def test_chat_qa_prompt_formatting():
    """Verifica la correcta inyección de variables en ChatPromptTemplate."""
    dummy_context = "El Modelo 303 de IVA se presenta de forma trimestral."
    dummy_question = "¿Cuándo se presenta el IVA?"

    formatted_messages = CHAT_QA_PROMPT.format_messages(
        context=dummy_context, question=dummy_question
    )

    # Debe generar 2 mensajes: system y human
    assert len(formatted_messages) == 2
    assert formatted_messages[0].type == "system"
    assert formatted_messages[1].type == "human"

    # El contenido debe incluir el contexto y la pregunta inyectados
    assert dummy_context in formatted_messages[1].content
    assert dummy_question in formatted_messages[1].content


def test_standalone_qa_prompt_formatting():
    """Verifica la correcta inyección de variables en el prompt estándar."""
    dummy_context = "Los gastos de manutención tienen un límite de 26,67 € diarios."
    dummy_question = "¿Cuánto puedo desgravar de comidas?"

    formatted_text = STANDALONE_QA_PROMPT.format(
        context=dummy_context, question=dummy_question
    )

    assert dummy_context in formatted_text
    assert dummy_question in formatted_text
    assert "RESPUESTA:" in formatted_text
