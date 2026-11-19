FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/models \
    DATA_DIR=/data \
    SAMPLES_DIR=/samples

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Download the reranker at build time so the container works offline and starts instantly.
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder as T; T('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/models')"

COPY backend/app ./app
COPY samples /samples

RUN useradd --create-home appuser && mkdir -p /data && chown -R appuser /data /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
