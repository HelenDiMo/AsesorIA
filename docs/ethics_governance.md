# Ética y gobernanza

Este documento recoge la gobernanza del proyecto, los principios éticos
aplicados, los controles de seguridad y privacidad y las limitaciones
conocidas que deben conocerse antes de usar AsesorIA. Es la *reflexión
ética* referenciada en los criterios de aceptación (§10 de
`docs/acceptance_criteria.md`).

## Alcance y propósito

AsesorIA es una herramienta de apoyo a la consulta documental fiscal para
autónomos en España. No es, y no debe presentarse como:

* asesoramiento fiscal profesional;
* sustituto de un asesor o de un profesional colegiado;
* una autoridad legal;
* un mecanismo automático para tomar decisiones fiscales vinculantes.

Las respuestas se generan exclusivamente a partir del corpus indexado y
deben verificarse contra la fuente citada (documento y página) antes de
cualquier uso real.

## Gobernanza

### Separación de responsabilidades

El sistema separa corpus, procesamiento documental, recuperación,
generación, interfaz y evaluación. Cada capa tiene su documentación y sus
pruebas propias, y la interfaz solo habla con el backend a través del
contrato del adaptador (`ui/rag_adapter.py`).

### Trazabilidad de decisiones

* Decisiones de retrieval (umbral, banda de relajación, chunking):
  documentadas en `docs/retrieval_*.md` con experimentos y evidencia.
* Benchmark de 40 preguntas con documento/página esperados.
* Evaluación RAG opcional con MLflow (LLM-as-judge) en
  `scripts/evaluate_rag_mlflow.py`.
* Registro de fuentes: `data/sources.csv` y `scripts/download_corpus.sh`
  permiten reconstruir el corpus; los PDF originales no se versionan en
  Git ni viajan en la imagen Docker.

### Revisión de cambios

Los cambios viajan por rama feature y pull request. CI ejecuta la suite
de tests (matriz Python 3.11/3.13), el build del sitio de documentación
y la construcción de la imagen Docker.

## Principios éticos

1. **Honestidad de límites**: abstención explícita cuando el corpus no
   contiene evidencia; copy en español natural, profesional y sin
   promesas de cobertura total; sin métricas de acierto inventadas; el
   modo demostración/mock se etiqueta siempre como tal y los fallos de
   backend no se convierten en respuestas falsas.
2. **Transparencia y trazabilidad**: cada respuesta cita documento y
   página; el panel de fuentes muestra el fragmento recuperado.
3. **Fuentes públicas**: el corpus se compone solo de documentación
   oficial pública (AEAT, BOE), lo que evita incorporar deliberadamente
   datos fiscales reales de contribuyentes en desarrollo y evaluación.
4. **Minimalización de datos**: el logging estructurado tiene prohibido
   registrar `question`, `answer` ni `content`
   (`src/common/logging_config.py`); el tracing de MLflow es opcional y
   local.
5. **Proveedor externo**: la consulta del usuario se envía a un proveedor
   LLM externo (Groq por defecto) para generar la respuesta; la
   conversación se persiste localmente (SQLite de Chainlit). Esta
   transferencia debe conocerse en cualquier evaluación de privacidad.

## Seguridad: controles implementados

* Aislamiento `source_scope` (público/privado) con `session_id` por
  sesión en el retriever.
* Validación de datos y contratos en la frontera interfaz↔backend.
* Separación entre documentos originales y artefactos generados; corpus
  e índice fuera de Git (`.gitignore` / `.dockerignore`).
* Secretos solo en variables de entorno: `.env.example` documenta
  nombres, CI usa claves dummy y el blueprint de Render usa
  `sync: false` / `generateValue`.
* Prueba de prompt injection (inglés) con rechazo explícito del modelo y
  `docs=0` (README §25.8); la prueba complementaria en español está
  pendiente (README §25.9).

Estos son controles técnicos del proyecto, no una certificación formal de
seguridad.

## Limitación documentada: anonimización de PII (trabajo pendiente)

Requisito exigido en `docs/acceptance_criteria.md` §10 («Trabajo
pendiente: anonimización de datos personales»):

* Existe `redact_pii()` en `src/privacy/pii.py` (detección pattern-based
  de IBAN, email, NIF, NIE, CIF y teléfono), con tests en
  `tests/test_pii.py`.
* **No está integrado en el pipeline de ingesta/indexación**: los
  documentos privados que contengan información personal pueden conservar
  dichos datos durante el proceso de indexación.
* La detección es pattern-based y no equivale a una solución
  especializada de DLP/NER; la detección de nombres de personas requiere
  NER y queda como mejora futura.
* Hasta que la redacción esté integrada y validada (tanto en texto como
  en tablas extraídas de PDF), **no debe presentarse la anonimización de
  PII como una funcionalidad disponible del sistema**.
* El aviso de privacidad en la interfaz queda como dependencia pendiente
  de UI.

## Uso aceptable y uso no apto

**Apto**: consulta orientativa sobre la documentación oficial indexada,
con verificación de la cita antes de cualquier uso real.

**No apto**:

* sustituir al asesor o presentar la respuesta como criterio fiscal
  vinculante sin revisión humana;
* indexar documentos con datos personales de terceros mientras la
  anonimización no esté integrada;
* automatizar presentaciones, decisiones o comunicaciones con la
  Administración sin supervisión;
* tratar AsesorIA como fuente legal vigente sin verificar el texto
  original citado.

## Evaluación y revisión continua

* Preguntas sin respuesta y fuera de alcance incluidas en el benchmark
  para validar la abstención.
* Tests específicos: `tests/test_retriever_fallback.py` (umbral y banda
  de relajación), `tests/test_conversational_rag.py` (reescritura de
  consultas y grounding), `tests/test_pii.py`, `tests/test_tracing.py`.
* LLM-as-judge en MLflow: corridas pequeñas, no validación científica
  exhaustiva (README §25.6 y §22.2).
* Limitaciones y follow-ups vigentes de la release candidate: README
  §25.9.

## Referencias

* README §9 (Governance, Ethics, Safety & Security), §10 (Privacidad y
  PII), §25.8 (seguridad) y §25.9 (limitaciones).
* `docs/acceptance_criteria.md` §10 (anonimización de PII pendiente).
* Este documento se publica en el sitio Docusaurus en
  `/docs/ethics_governance`.
