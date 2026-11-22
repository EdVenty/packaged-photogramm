# Refractive SfM worker: Refractive COLMAP (MDSKiel/colmap_underwater, IROS 2024),
# built from source following the fork's own docker/Dockerfile.
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.3.1
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG COLMAP_UW_REPO=https://github.com/MDSKiel/colmap_underwater.git
ARG COLMAP_UW_REF=underwater
ARG CUDA_ARCHITECTURES="75;80;86;89"
ENV DEBIAN_FRONTEND=noninteractive QT_XCB_GL_INTEGRATION=xcb_egl

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
        git cmake ninja-build build-essential python3 \
        libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \
        libboost-system-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
        libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev \
        libqt5opengl5-dev libcgal-dev libceres-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git clone ${COLMAP_UW_REPO} /src/colmap_uw \
 && cd /src/colmap_uw && git checkout ${COLMAP_UW_REF} \
 && mkdir build && cd build \
 && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DCMAKE_INSTALL_PREFIX=/colmap \
 && ninja install \
 && cd / && rm -rf /src/colmap_uw

COPY worker/pgworker /opt/pgworker/pgworker
ENV PATH="/colmap/bin:${PATH}" PYTHONPATH=/opt/pgworker PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen PG_WORKER_KINDS=sfm_refrac
CMD ["python3", "-m", "pgworker"]
