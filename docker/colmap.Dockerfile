# SfM worker: official COLMAP image (CUDA) + the stdlib-only worker agent.
# Pin a specific tag via COLMAP_IMAGE for reproducible builds.
ARG COLMAP_IMAGE=colmap/colmap:latest
FROM ${COLMAP_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
 && rm -rf /var/lib/apt/lists/*
COPY worker/pgworker /opt/pgworker/pgworker
ENV PYTHONPATH=/opt/pgworker PYTHONUNBUFFERED=1 QT_QPA_PLATFORM=offscreen PG_WORKER_KINDS=sfm
CMD ["python3", "-m", "pgworker"]
