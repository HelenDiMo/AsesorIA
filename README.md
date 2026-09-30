# AsesorIA
AsesorIA — Sistema corporativo de preguntas y respuestas mediante RAG para la consulta fiscal, contable y normativa de autónomos en España, garantizando trazabilidad y mitigación de alucinaciones.

## Frontend (ui/)

Chainlit web interface: document upload, chat, answers and **source
traceability** (document, page, section and snippet).

```bash
pip install -r requirements.txt
cd ui
chainlit run app.py        # http://localhost:8000
```

- Frontend tests: `python -m pytest tests/ -q`
- Documentation: [`ui/README.md`](ui/README.md) — UX states, visual system,
  integration contract and handoff to connect the RAG backend.
- Status: **frontend integrated and tested; RAG backend pending integration**
  (the UI runs on a demo mock, labeled as such, until the real engine is
  connected in `ui/rag_adapter.py → create_backend()`).
