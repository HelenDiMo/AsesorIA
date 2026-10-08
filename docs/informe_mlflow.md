# Informe de evaluación y costes (MLflow)

Resultados de las evaluaciones registradas en MLflow
(experimento `asesoria-rag`, almacén `sqlite:///mlflow.db`). Esta página se genera
automáticamente; para ver la UI viva en local:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

## Evaluación end-to-end (RAG completo)

Runs `rag-eval` sobre el benchmark revisado: aciertos de evidencia
(documento/página esperados), latencia media total, consumo de tokens
del LLM y coste estimado. El juez LLM puntúa faithfulness / relevance
de 1 a 5 cuando la evaluación se ejecuta con `--judge`.

| Fecha | Preguntas | Documento | Página | Latencia media | Tokens (entrada / salida) | Coste | Juez F / R |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-08 12:40 UTC | 1 | 100% (1/1) | 100% (1/1) | 34217 ms | 4338 / 480 | $0.0009 | — |
| 2026-10-08 12:31 UTC | 3 | 100% (3/3) | 100% (3/3) | 31014 ms | 30300 / 8157 | $0.0094 | 4.00 / 4.33 |

## Suite de pruebas

Run `test-suite` generado desde el informe JUnit de pytest
(`python -m pytest tests/ -q --junitxml=results.xml` +
`python -m scripts.log_test_results`).

| Fecha | Total | Pasadas | Fallidas | Skipped | Duración | Commit |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-08 12:30 UTC | 515 | 510 | 0 | 5 | 118 s | `dc8f120` |

## Método y límites

- Coste = tokens × precio publicado por Groq para
  `openai/gpt-oss-120b`: **$0,15 / M tokens de entrada** y
  **$0,60 / M tokens de salida** (tags `eval.price_*` del run y
  atributo `mlflow.llm.cost` en cada span de las trazas).
- El detalle pregunta a pregunta queda como artefacto
  `results.json` de cada run (MLflow → run → Artifacts).
- La demo pública usa el free tier de Groq (≈ 200.000 tokens/día,
  unas 46 consultas/día); los costes de esta tabla corresponden a
  las evaluaciones, no a la demo desplegada.
- Página generada: 2026-10-08 13:28 UTC · runs finalizados: 4.
