# Despliegue en Render

AsesorIA se despliega como **Web Service con runtime Docker**: Render construye
la imagen desde el `Dockerfile` del repositorio (o consume la imagen publicada
en GHCR por el job `publish` de CI). La configuración declarativa está en
`render.yaml` (Blueprint).

## Requisitos

1. Cuenta en Render y el repositorio conectado (New → Blueprint →
   seleccionar la rama con `render.yaml`).
2. `GROQ_API_KEY` real.
3. Plan **Starter o superior**: el plan Free no admite discos persistentes y
   el corpus se perdería en cada redeploy (el servicio arrancaría sin datos).

## Variables de entorno

| Variable | Origen | Valor |
|----------|--------|-------|
| `GROQ_API_KEY` | dashboard (`sync: false`) | clave real, nunca en Git |
| `OAUTH_GOOGLE_CLIENT_ID` / `_SECRET` | dashboard | opcional (login) |
| `CHAINLIT_AUTH_SECRET` | `generateValue: true` | secreto de sesión de Chainlit |
| `CHROMA_DIR` | `render.yaml` | `/app/chroma_db` (paso 2: ver abajo) |
| `CHROMA_COLLECTION` | `render.yaml` | `corpus_384_48` (provisional) |
| `CHAINLIT_SQLITE_PATH` | `render.yaml` | `/app/chroma_db/chainlit.db` |

El disco (`mountPath: /app/chroma_db`) persiste corpus y historial entre
redeploys. `render.yaml` es la única fuente de verdad de estas claves.

## Arranque en dos pasos (bootstrap del corpus)

El índice **no está en Git** (`.gitignore`) ni en la imagen
(`.dockerignore`), por lo que el primer deploy arranca con un disco vacío:

1. **Primer deploy**: el servicio queda en `/health` 200, pero el retriever
   no encuentra colecciones y el asesor responde con el mensaje honesto de
   «no dispongo de suficiente información» (no inventa fuentes).
2. **Indexar en la instancia**: abrir *Shell* en el dashboard de Render y
   ejecutar `python -m scripts.index_corpus --index`. El settings escribe en
   `<PROJECT_ROOT>/chroma_db` = `/app/chroma_db` (el disco montado), sea
   cual sea el directorio de trabajo. Tarda bastante en CPU (≈2 h en local
   con torch-CPU); vigilar memoria (el plan Starter ajusta 1 GB: si el
   proceso queda sin memoria, subir a Standard para indexar). El resultado
   es un run nuevo en `/app/chroma_db/corpus_runs/<id>` con estado
   `verified`.
3. **Apuntar la app al run**: copiar `<id>` del `manifest.json`, cambiar en
   el dashboard `CHROMA_DIR=/app/chroma_db/corpus_runs/<id>` y redeplegar.
   Repetir este paso en cada reindexación.

Limitación conocida (sin mecanismo propio todavía): no existe canal oficial
para subir un índice indexado localmente; publicar el artefacto de índice
(como imagen o paquete) es trabajo futuro, no implementado aquí.

## OAuth en producción

Registrar como redirect URI en Google Cloud Console:

```
https://<tu-app>.onrender.com/auth/oauth/google/callback
```

Sin OAuth la app arranca igual (login desactivado).

## Alternativa: imagen de GHCR (CD)

El job `publish` de CI publica `ghcr.io/<owner>/<repo>:sha-<commit>` y
`:main` en cada push a `main`. En `render.yaml` puede sustituirse el build
por:

```yaml
image: ghcr.io/<owner>/<repo>:sha-<commit>
```

Los paquetes privados de GHCR requieren `imageCredentials` en el Blueprint.
Ventaja: se despliega exactamente la imagen validada por CI (misma tag
`sha-<commit>`), sin volver a construir en Render.

## Riesgos y limitaciones observados

- **Plan Free sin disco**: corpus efímero; solo sirve para demostración de
  UI (modo demo honesto, sin datos).
- **Cold start**: tras el sleep del plan Free/estacional, el primer arranque
  tarda (carga de dependencias; el modelo E5 ya va horneado en la imagen).
- **Indexación in-place ≈2 h** y consumo de memoria en planes pequeños
  (proceso verificado en local, no en Render: cuenta propia pendiente).
- **Build en Render** descarga torch-CPU + dependencias (~1 GB) en cada
  build sin caché: monitorizar el timeout de build la primera vez; la
  alternativa GHCR evita reconstruir.
- **Dominio**: Render aporta TLS en `*.onrender.com`; dominio propio es
  configuración adicional fuera de este archivo.
