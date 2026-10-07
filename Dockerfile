# AsesorIA — imagen de la aplicación (Chainlit + backend RAG)
#
# Base python:3.13-slim: misma versión mayor con la que se validó el E2E
# real local (3.13.12). CI ejecuta los tests en 3.11 (ver .github/workflows).

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# Torch solo-CPU (dependencia de sentence-transformers): evita la build CUDA
# de varios GB. Debe instalarse ANTES de requirements.txt para que
# sentence-transformers reutilice este torch en lugar de descargar otro.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Usuario no root: los artefactos escritos por la app (SQLite de Chainlit,
# caché de Hugging Face) quedan propiedad de `app`.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface /app/ui/.data \
    && chown -R app:app /app /home/app

COPY --chown=app:app . .

USER app

# Modelo de embeddings por defecto horneado: el arranque no descarga nada
# (otros modelos configurables se descargarían en caliente a HF_HOME).
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-base')"

# Chainlit se ejecuta desde ui/ (APP_ROOT: .chainlit/ y public/ resuelven ahí).
WORKDIR /app/ui
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
