# Despliegue en Railway

AsesorIA se despliega en **Railway** como servicio Docker construido desde el
`Dockerfile` del repositorio (rama `main`). Esta es la implementación activa;
el blueprint de Render documentado en [`deploy_render.md`](deploy_render.md)
queda como alternativa.

URL pública: `https://asesoria.up.railway.app`

## Requisitos

1. Cuenta en Railway y proyecto conectado al repositorio (New Project →
   Deploy from GitHub repo → `HelenDiMo/AsesorIA`, rama `main`).
2. `GROQ_API_KEY` real.
3. Recursos del plan de prueba: **1 GB RAM** (la aplicación mide ≈831 MB en
   reposo; con 512 MB el contenedor muere por OOM) y volumen de **500 MB**
   como máximo.

## Configuración del servicio

| Ajuste | Valor |
|--------|-------|
| Builder | Dockerfile (`/Dockerfile`) |
| Región | US East (`iad`) |
| RAM | 1 GB (máximo del plan de prueba) |
| Volumen | `/app/chroma_db` (500 MB) |
| Healthcheck | `/health` |
| Restart policy | On failure (máx. 10 reintentos) |

## Variables de entorno

Se configuran en Settings → Variables; nunca se commitean:

| Variable | Origen | Valor |
|----------|--------|-------|
| `GROQ_API_KEY` | dashboard | clave real, nunca en Git |
| `OAUTH_GOOGLE_CLIENT_ID` / `_SECRET` | dashboard | opcional (login) |
| `CHAINLIT_AUTH_SECRET` | dashboard | secreto de sesión de Chainlit |
| `CHROMA_DIR` | dashboard | `/app/chroma_db/corpus_runs/ae121e0628ac4595b09cce0dbfb72205` |
| `CHROMA_COLLECTION` | dashboard | `corpus_384_48` |
| `CHAINLIT_SQLITE_PATH` | dashboard | `/app/chroma_db/chainlit.db` |
| `CHAINLIT_URL` | dashboard | `https://asesoria.up.railway.app` (obligatoria tras el proxy: sin ella el callback OAuth llega como `http://` y Google devuelve `redirect_uri_mismatch`) |
| `RAILWAY_RUN_UID` | dashboard | `0` — Railway monta el volumen como root; sin esta variable el usuario `app` no puede escribir en `/app/chroma_db` y el arranque muere con `Permission denied` |

## Arranque con corpus (custom start command)

El índice **no está en Git** (`.gitignore`) ni en la imagen
(`.dockerignore`) y Railway no expone shell para reindexar. El volumen se
rellena en el primer arranque mediante un *custom start command* que
descarga el corpus ya indexado desde Hugging Face, lo extrae y solo entonces
arranca la aplicación:

```bash
python -c 'import urllib.request,zipfile,os; d="/app/chroma_db/corpus_runs/ae121e0628ac4595b09cce0dbfb72205"; m=d+"/chroma.sqlite3"; u="https://huggingface.co/juanmanueldlf/asesoria-corpus/resolve/main/corpus.zip"; os.makedirs(d,exist_ok=True); (not os.path.isfile(m)) and (urllib.request.urlretrieve(u,"/tmp/c.zip"), zipfile.ZipFile("/tmp/c.zip").extractall(d)); os.chdir("/app/ui"); os.execvp("chainlit",["chainlit","run","app.py","--host","0.0.0.0","--port",os.environ.get("PORT","8000"),"--headless"])'
```

La descarga solo se ejecuta si falta `chroma.sqlite3`, por lo que es
idempotente en cada redeploy. Si se elimina el volumen, se repite sola. Sin
este paso (o sin volumen) el servicio arranca igual y responde con el modo
honesto de «no dispongo de suficiente información», sin inventar fuentes.

## OAuth en producción

Registrar como redirect URI en Google Cloud Console:

```
https://asesoria.up.railway.app/auth/oauth/google/callback
```

Sin OAuth la app arranca igual (login desactivado).

## CI/CD

- **Railway**: despliega automáticamente desde `main` en cada push (builder
  Dockerfile desde el repositorio).
- **GHCR**: el job `publish` de CI publica
  `ghcr.io/<owner>/<repo>:sha-<commit>` y `:main`; Railway no lo consume
  (construye desde el repo), pero la imagen queda disponible para
  despliegues trazables o para Render (ver `deploy_render.md`).

## Costes y límites observados

- Plan de prueba: crédito de 5 $ durante 30 días; con 1 GB encendido
  (≈10 $/mes equivalentes) rinde ≈1-2 semanas de servicio continuo.
- RAM medida en reposo: ≈831 MB sobre 1 GB disponibles; los picos de
  embedding + generación se acercan al límite.
- La app no puede reindexar en la nube (el proceso tarda ≈2 h y supera los
  1 GB de RAM): el corpus se distribuye como paquete indexado vía Hugging
  Face (el mecanismo que `deploy_render.md` dejaba como trabajo futuro está
  implementado aquí en el *start command*).
