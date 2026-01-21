FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_CJK_FONTS=0

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

ENV VIRTUAL_ENV=/opt/pyeidors_venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Create a stable runtime environment on top of the official FEniCS image.
# This image is intended to be used by mounting the repository into /workspace.
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3-venv \
  && if [ "${INSTALL_CJK_FONTS}" = "1" ]; then apt-get install -y --no-install-recommends fonts-wqy-zenhei; fi \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" --system-site-packages \
  && pip install --upgrade pip setuptools wheel

COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install -r /tmp/requirements-lock.txt

ARG CUQIPY_VERSION=1.3.0
ARG CUQIPY_FENICS_VERSION=0.8.0
RUN pip install "CUQIpy==${CUQIPY_VERSION}" "CUQIpy-FEniCS==${CUQIPY_FENICS_VERSION}"

# GPU-enabled PyTorch by default (cu128). For CPU-only builds, override TORCH_INDEX_URL.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCHAUDIO_VERSION=2.7.1
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

WORKDIR /workspace
ENV PYTHONPATH="/workspace/src:${PYTHONPATH}"

CMD ["bash"]

