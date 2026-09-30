# Asesor Fiscal IA - UI Module

Chainlit user interface for the RAG system. **Contains no retrieval,
embedding, prompt or LLM logic**: every query goes through the adapter.

## File structure

```
ui/
├── __init__.py            # Makes `ui` an importable package for the tests
├── app.py                 # UI orchestrator (states, validation, flow)
├── contracts.py           # Data contracts (RAGResponse, Source)
├── rag_adapter.py         # Single UI ↔ backend boundary (swap the mock here)
├── mock_rag.py            # Deterministic mock for development and demo
├── formatters.py          # All visible copy (answers, sources, states)
├── README.md              # This file
├── chainlit.md            # Welcome markdown (fallback)
├── chainlit_es.md         # Welcome markdown in Spanish (language `es`)
├── .chainlit/config.toml  # Name, language, theme, CSS/JS, file upload
├── public/
│   ├── theme.json         # Color tokens (light + dark) — ONLY place
│   ├── custom.css         # Scrollbar, selection, focus ring
│   ├── custom.js          # Custom editor placeholder
│   ├── logo_light.svg     # Header logo (light theme)
│   ├── logo_dark.svg      # Header logo (dark theme)
│   └── avatar.svg         # Avatar for every message and step
└── ...

tests/
├── test_frontend_contract.py  # Contract, formatters, mock, adapter
├── test_app_helpers.py        # File validation and app.py helpers
└── test_integration_fake_backend.py  # Adapter against a fake backend (7 cases)
```

## How to run

```bash
pip install chainlit==2.12.0

cd ui
chainlit run app.py            # http://localhost:8000
```

**Always run from `ui/`**: `APP_ROOT` is the working directory, so `public/`
and `.chainlit/` resolve inside `ui/`.

## Interface states

| State | When | Copy (summary) |
|---|---|---|
| Welcome | `on_chat_start` | Identity + 3 steps + grounding limit + demo-mode badge |
| Documentation | A file is attached | `📄 Documentación disponible · N documentos` + list |
| No documents | Question with no files | `📄 Todavía no hay documentación cargada` + upload button |
| Processing | During the query | Step `Buscando fuentes` with `Recuperado: N fragmentos` |
| Answer | `grounded=True` with sources | Answer + `📄 Fuentes utilizadas · N` |
| No information | `grounded=False` or no sources | `ℹ️ No he encontrado información suficiente` + reason |
| Error | Backend exception | `⚠️ No he podido conectar…` / `⚠️ Se ha producido un error técnico` |
| Empty question | Message without content | Friendly reminder with an example |

**No information ≠ error**: the first is an expected outcome of grounding;
the second is a failure and uses different messages.

## Source traceability

Every answer with sources is followed by an independent block:

```
**📄 Fuentes utilizadas · 2**

**1. manual-iva.pdf** · pág. 42 · Deducción del IVA · relevancia 0,94
> "El IVA soportado es deducible cuando…"

**2. BOE-...pdf** · pág. 18 · Artículo 95 · relevancia 0,89
> "Se considerará deducible el IVA…"
```

- `page`, `section` and `score` are optional: when they arrive as `null` the
  UI shows `página no disponible` / `sección no disponible` and the score is
  omitted.
- If the sources list is empty, **no block is emitted** (the UI never breaks).

## Expected backend contract

```json
{
    "answer": "Texto de la respuesta...",
    "sources": [
        {
            "document": "Nombre del documento.pdf",
            "content": "Fragmento recuperado...",
            "page": 42,
            "section": "Sección relevante",
            "score": 0.94,
            "metadata": {}
        }
    ],
    "grounded": true,
    "no_answer_reason": null,
    "latency_ms": 520.0
}
```

Fields: `document` and `content` always present; `page`, `section`, `score`
and `metadata` optional (`null` tolerated). Not enough information:
`{"answer": "", "sources": [], "grounded": false, "no_answer_reason": "..."}`.

## Using the mock

`mock_rag.py` is deterministic (same question → same answer):

| Keywords | Result |
|---|---|
| `iva` + `deducible` | Answer + 2 sources with full metadata |
| `303` / `autoliquidación` | Answer + 3 sources (several documents) |
| `alta` + `autónomo` | Answer + 1 source |
| `cripto` / `bitcoin` | `grounded=False`, no sources (no information) |
| `metadata` / `incompleto` | Sources with `null` metadata (UI robustness) |
| `error` / `fallo` | Raises an exception → UI error state |
| anything else | Generic answer + 1 source |

The 🧠 badge on the welcome screen tells the user they are in demo mode
(`RagAdapter.is_mock`). Once the real engine is connected it becomes `False`
and the badge disappears.

## Integration with the RAG

### Integration point

`ui/rag_adapter.py` is the **only** point of contact. The chain is:

```
app.py  →  rag_adapter.py  →  engine.py (RAG team)   [or MockRAG, today]
```

`app.py` knows nothing about LangChain, Chroma, prompts or the LLM: only the
contract.

### 1. Which function the frontend will call

```python
async def query(question: str, documents: list[str]) -> dict | RAGResponse
# ask(...) and a synchronous implementation are also accepted
```

The adapter invokes it with `backend.query(question, documents)` (or `ask`),
expects a sync or `await`-able result and normalizes it with
`RAGResponse.from_any(...)`.

### 2. Parameters received

| Parameter | Type | Description |
|---|---|---|
| `question` | `str` | The user's natural-language question (already validated: non-empty). |
| `documents` | `list[str]` | Names/paths of the documents loaded in the session. The engine may filter or ignore them. |

### 3. Returned structure (REQUEST / RESPONSE)

REQUEST (what the adapter sends to the engine):

```json
{
  "question": "¿Qué gastos son deducibles de un autónomo?",
  "documents": ["Manual_Renta.pdf", "modelo-303.pdf"]
}
```

RESPONSE (what the engine must return, dict or object equivalent):

```json
{
  "answer": "Los gastos necesarios para la actividad son deducibles…",
  "sources": [
    {
      "document": "Manual_Renta.pdf",
      "page": 42,
      "section": "Gastos deducibles",
      "content": "Son deducibles los gastos de suministros…",
      "score": 0.94,
      "metadata": {}
    }
  ],
  "grounded": true,
  "no_answer_reason": null,
  "latency_ms": 320
}
```

### 4. How to represent sources

Each element of `sources` provides traceability; the UI renders
document · page · section · relevance · snippet:

- `document` (**required**): name shown to the user. If a path arrives, the
  UI shows only the filename (never UUIDs or folders).
- `content` (**required**): the exact retrieved snippet (rendered as a quote).
  When missing, the UI shows *(fragmento no disponible)*.
- `page`, `section`, `score` (optional): `null` → the UI writes
  *página/sección no disponible* and omits the relevance.
- `metadata` (optional): free dict; if it contains `source`/`page`/`section`/
  `score`, the UI uses them as fallback.

With `sources: []` the UI does **not** emit the sources block (provenance is
never invented).

### 5. How to signal `grounded=False`

```json
{ "answer": "", "sources": [], "grounded": false,
  "no_answer_reason": "sin resultados relevantes en la documentación" }
```

The UI renders it as *«No he encontrado información suficiente»* (a
responsible state, **not** a technical error). `no_answer_reason` is shown as
*Motivo: …*. `grounded` also accepts `"true"`/`"false"`; when omitted with a
`no_answer_reason` present it is interpreted as `false`.

### 6. How to communicate errors

**Raise an exception** (`raise`). The UI catches the failure inside the step
and shows a friendly message (`⚠️ Se ha producido un error técnico` or
`⚠️ No he podido conectar con el motor de consulta`), **never** a stack trace
or internal details. Returning an `answer` containing the error text is
wrong: it would be shown to the user as if it were the answer.

### Connecting `engine.py` (2 steps, only `rag_adapter.py`)

```python
def create_backend():
    from src.rag.engine import RagEngine   # ← RAG team import
    return RagEngine()
```

Once done, adapter instances report `is_mock = False` and the welcome screen
drops the 🧠 demo badge. **`app.py`, `formatters.py` and the UI need no
changes.**

### Guaranteed normalization (tolerance)

The adapter converts to the contract without ever raising, even if the
engine delivers:

`sources: null` · a single source instead of a list · `page`/`score` as text ·
`document` as a path · LangChain-style objects (`page_content`, `metadata`) ·
`answer: null` · `grounded: "false"` · a `None` response · an already
normalized response (`RAGResponse`).

## Interface customization

| What | Where |
|---|---|
| Palette (light/dark), radii, sidebar | `public/theme.json` (HSL tokens) |
| Global styles | `public/custom.css` |
| Editor placeholder | `public/custom.js` |
| Name, language, default theme, layout | `.chainlit/config.toml` |
| Brand logo and avatar | `public/logo_*.svg`, `public/avatar.svg` + `default_avatar_file_url` |
| Copy (all visible text) | `formatters.py` |

### Important Chainlit 2.12 restriction

- The **name of a `cl.Step`** is used as the avatar identifier
  (`/avatars/<name>`) and that endpoint rejects accents and symbols → use
  only `[a-zA-Z0-9_ .-]` (e.g. `"Buscando fuentes"`). Accented text goes in
  the step `output`, which does accept it.
- `cl.user_session` only exposes `get(key, default)` / `set(key, value)`.
- There is no `@cl.on_action`: use `cl.Action(name, payload, label, tooltip,
  icon)` + `@cl.action_callback("name")`.
- Exceptions must **never** escape an `async with cl.Step(...)`: the step
  itself would send `str(exc)` to the client. Catch them inside the block.
- Spontaneous file upload: `[features.spontaneous_file_upload]` in the
  config; files arrive in `on_message` as `message.elements` (with the
  original `.name` and the storage `.path`).
- `language = "ex"` must match an existing translation file (`es`, not
  `es-ES`) and `chainlit_es.md`; otherwise Chainlit logs a warning at
  startup. Both files are included → clean startup.
- Known browser console notice:
  `Missing Description or aria-describedby for DialogContent` — comes from
  Chainlit 2.12's internal bundle (Radix dialog), **not** from project code.
  It is cosmetic: it breaks neither the upload dialog nor the accessibility
  of the rest of the interface. It will disappear when Chainlit fixes it.

## Supported file types

PDF, TXT and Markdown (`.pdf`, `.txt`, `.md`), up to 50 MB per file and 5 per
message. Unknown extensions, empty or unreadable files produce a specific
error message (never a silent failure).

## Tests

```bash
python -m pytest tests/ -q      # 97 tests
```

- `tests/test_frontend_contract.py` — contract (`Source`/`RAGResponse`), every
  formatter/copy string, mock scenarios and the adapter.
- `tests/test_app_helpers.py` — file validation and `app.py` helpers.
- `tests/test_integration_fake_backend.py` — fake backend covering the 7
  contract cases (2 sources, 1 source, no sources, `grounded=False`,
  incomplete metadata, error, latency) plus tolerance variants.

No network dependencies or external services.

## UX decisions

1. Ephemeral sessions: conversations and documents are never persisted.
2. All copy in Spanish, professional tone, no technical jargon for the user.
3. Sources always visible (no accordion) right under the answer.
4. Stack traces, keys and internal paths never reach the interface.
5. No authentication and no calls to external services.

---

# Frontend handoff

Guide to connecting `src/rag/engine.py` without asking how the UI consumes
it.

## Integration point

| What | Where |
|---|---|
| Only file to touch | `ui/rag_adapter.py` → `create_backend()` |
| Data contract | `ui/contracts.py` (`Source`, `RAGResponse`) |
| Copy / states | `ui/formatters.py` (do not touch) |
| UI orchestration | `ui/app.py` (do not touch) |

## Expected engine signature

```python
async def query(question: str, documents: list[str]) -> dict:
    ...
```

- Acceptable: `ask(...)` instead of `query(...)`, or a synchronous
  implementation.
- `documents` = names/paths of the session documentation (the engine decides
  whether to use them for filtering).
- Returns a `dict` with the contract keys (an object with the same
  attributes or an already built `RAGResponse` also works).

## Response fields

| Field | Required | Semantics |
|---|---|---|
| `answer` | Yes | Answer text (Markdown allowed). `""` when there is no answer. |
| `sources` | Yes (may be `[]`) | Snippets used, with provenance. |
| `grounded` | Yes | `true` = answer grounded in documentation. `false` = not enough information. |
| `no_answer_reason` | No | Short reason in Spanish (shown when `grounded=false`). |
| `latency_ms` | No | Engine latency. |
| `sources[].document` | Yes | Visible document name (if a path arrives, only the file is shown). |
| `sources[].content` | Yes | The exact retrieved snippet. |
| `sources[].page` | No | Integer. `null` → *página no disponible*. |
| `sources[].section` | No | Text. `null` → *sección no disponible*. |
| `sources[].score` | No | Float 0–1. `null` → relevance not shown. |
| `sources[].metadata` | No | Free dict (fallback for `source`/`page`/`section`/`score`). |

## Expected engine behavior

1. **Grounded answer** → `grounded: true` + `sources` with at least one real
   snippet. The UI paints the `📄 Fuentes utilizadas · N` block.
2. **Not enough information** → `grounded: false`, `answer: ""` and
   `no_answer_reason`. The UI shows the ℹ️ state (not an error).
3. **Real error** → `raise`. The UI shows a friendly `⚠️` with no stack
   trace. Never return the error inside `answer`.

## Replacing the mock with the real engine

```python
# ui/rag_adapter.py
def create_backend():
    from src.rag.engine import RagEngine
    return RagEngine()
```

That is all: the adapter calls `RagEngine().query(...)`, normalizes the
response and `is_mock` flips to `False` (the 🧠 demo badge disappears).
`app.py`, `formatters.py` and the UI stay unchanged.

## Minimum checks after connecting `engine.py`

```bash
python -m pytest tests/ -q      # 97 tests (contract + adapter + formatters)
cd ui && chainlit run app.py --port 8000
```

Manual check (5 minutes):

1. Upload a real PDF → `📄 Documentación disponible` appears with the
   **original filename**.
2. Question answered from that PDF → answer + `📄 Fuentes utilizadas · N`
   block with document, page, section and quote.
3. Question outside the corpus (e.g. cryptocurrencies) →
   *«No he encontrado información suficiente»* state with a reason.
4. Disconnect the engine / force a failure → friendly `⚠️`, no stack trace.
5. Welcome screen without the 🧠 badge (confirms `is_mock = False`).
6. Browser console free of errors and HTTP responses without 4xx/5xx.

If step 2/3 fails because of how `engine.py` returns the data, fix it
**inside `rag_adapter.py`** (normalization), never in the UI.
