# CPU worker: image preprocessing / underwater colour correction (OpenCV)
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "numpy<3" opencv-python-headless pillow
COPY worker/pgworker /opt/pgworker/pgworker
ENV PYTHONPATH=/opt/pgworker PYTHONUNBUFFERED=1 PG_WORKER_KINDS=cpu
CMD ["python3", "-m", "pgworker"]
