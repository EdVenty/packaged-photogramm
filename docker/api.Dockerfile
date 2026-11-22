# API server: FastAPI + SQLite
FROM python:3.12-slim
WORKDIR /app
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY server/photogramm_api /app/photogramm_api
ENV PG_DATA_DIR=/data PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
# Single process on purpose: SQLite access is serialised in-process.
CMD ["uvicorn", "photogramm_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
