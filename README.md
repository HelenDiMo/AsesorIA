# AsesorIA

**[Español](#español) · [English](#english)**

---

## Español

AsesorIA — Sistema corporativo de preguntas y respuestas mediante RAG para la
consulta fiscal, contable y normativa de autónomos en España, garantizando
trazabilidad y mitigación de alucinaciones.

### Frontend (ui/)

Interfaz web Chainlit: carga de documentación, chat, respuestas y
**trazabilidad de fuentes** (documento, página, sección y fragmento exacto),
con panel de fuentes desplegable y tema azul corporativo (claro/oscuro).

```bash
pip install -r requirements.txt
cd ui
chainlit run app.py        # http://localhost:8000
```

- Tests del frontend: `python -m pytest tests/ -q`
- Documentación: [`ui/README.md`](ui/README.md) — estados de UX, sistema
  visual, contrato de integración y guía para conectar el backend RAG.
- Estado: **frontend integrado y probado; backend RAG pendiente de
  integración** (la UI funciona sobre un mock de demostración, etiquetado
  como tal, hasta conectar el motor real en
  `ui/rag_adapter.py → create_backend()`).




---

## English

AsesorIA — Corporate RAG question-answering system for tax, accounting and
regulatory queries by freelancers (*autónomos*) in Spain, ensuring
traceability and mitigating hallucinations.

### Frontend (ui/)

Chainlit web interface: document upload, chat, answers and **source
traceability** (document, page, section and exact snippet), with a
collapsible sources panel and corporate blue theme (light/dark).

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

### ⚡ Backend Integration & Execution Notes

- **Pre-indexed Corpus Access:** Users can query the base legal and tax corpus (RETA, LGSS, IRPF, IVA) immediately upon startup without needing to upload files first.
- **Latency Metric Display:** Answers rendered by `RAGEngine` include an end-to-end latency metric in milliseconds (`⏱️ Latencia: X ms`) at the bottom of the message.
- **Windows Environment Prerequisite:** Chainlit stores message side-elements in a local directory. If running on Windows, ensure the `.files/` folder exists in the project root:

  ```bash
  mkdir -p .files
  ```
- **Visualization Dependencies:** Interactive chart widgets require plotly. Install it within your virtual environment:

   ```bash
  pip install plotly
  ```


