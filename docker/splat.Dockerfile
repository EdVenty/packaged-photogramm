# Splat worker: nerfstudio (splatfacto, gsplat) + WaterSplatting plugin.
# Versions follow the WaterSplatting README: CUDA 11.8, torch 2.1.2, nerfstudio 1.1.4.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
ARG WATER_SPLATTING_REPO=https://github.com/water-splatting/water-splatting.git
ARG WATER_SPLATTING_REF=main
ARG INSTALL_TCNN=1
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git ffmpeg libgl1 libglib2.0-0 \
        ninja-build build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --no-cache-dir "pip==24.0" "setuptools<70" wheel \
 && pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118

RUN if [ "$INSTALL_TCNN" = "1" ]; then \
      TCNN_CUDA_ARCHITECTURES="$(echo ${TORCH_CUDA_ARCH_LIST} | tr -d '.')" \
      pip install --no-cache-dir "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"; \
    fi

RUN pip install --no-cache-dir nerfstudio==1.1.4 "numpy<2"

RUN git clone ${WATER_SPLATTING_REPO} /opt/water-splatting \
 && cd /opt/water-splatting && git checkout ${WATER_SPLATTING_REF} \
 && git submodule update --init --recursive \
 && pip install --no-cache-dir --no-use-pep517 -e . \
 && pip install --no-cache-dir "numpy<2"

COPY worker/pgworker /opt/pgworker/pgworker
# gsplat compiles its CUDA kernels on first use; cache them in a volume.
ENV PYTHONPATH=/opt/pgworker PYTHONUNBUFFERED=1 TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    XDG_CACHE_HOME=/cache PG_WORKER_KINDS=splat
CMD ["python3", "-m", "pgworker"]
