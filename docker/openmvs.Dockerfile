# MVS worker: OpenMVS with CUDA, built the same way as upstream docker/buildInDocker.sh.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG OPENMVS_REPO=https://github.com/cdcseacave/openMVS.git
ARG OPENMVS_BRANCH=develop
ARG CGAL_VERSION=v6.0.1
ARG CUDA_ARCHITECTURES="75;80;86;89"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq --no-install-recommends \
        build-essential git cmake ca-certificates python3 python3-dev \
        libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev libglew-dev libglfw3-dev \
        libboost-iostreams-dev libboost-program-options-dev libboost-system-dev \
        libboost-serialization-dev libboost-thread-dev libopencv-dev \
        libgmp-dev libmpfr-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Eigen 3.4 (built against CUDA as upstream does)
RUN git clone --depth 1 --branch 3.4 https://gitlab.com/libeigen/eigen /src/eigen \
 && cmake -S /src/eigen -B /src/eigen_build -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
 && cmake --build /src/eigen_build --target install && rm -rf /src/eigen /src/eigen_build

# CGAL (header-only install)
RUN git clone --depth 1 --branch ${CGAL_VERSION} https://github.com/cgal/cgal /src/cgal \
 && cmake -S /src/cgal -B /src/cgal_build && cmake --build /src/cgal_build --target install \
 && rm -rf /src/cgal /src/cgal_build

RUN git clone --recurse-submodules --branch ${OPENMVS_BRANCH} ${OPENMVS_REPO} /src/openMVS \
 && cmake -S /src/openMVS -B /src/openMVS_build -DCMAKE_BUILD_TYPE=Release \
        -DOpenMVS_USE_CUDA=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include/ \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 \
 && cmake --build /src/openMVS_build -j"$(nproc)" && cmake --install /src/openMVS_build \
 && rm -rf /src/openMVS /src/openMVS_build

COPY worker/pgworker /opt/pgworker/pgworker
ENV PATH="/usr/local/bin/OpenMVS:${PATH}" PYTHONPATH=/opt/pgworker PYTHONUNBUFFERED=1 \
    PG_WORKER_KINDS=mvs
CMD ["python3", "-m", "pgworker"]
