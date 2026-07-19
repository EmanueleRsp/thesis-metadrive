# syntax=docker/dockerfile:1.4
FROM python:3.10.20-slim-bookworm@sha256:ff7161e2b8e2a56fc6a62a6099ff8feb72f1a6dbae9860cdcb9a6c65cf4c6be9 AS base

ARG USER_NAME=appuser
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG UV_VERSION=0.11.28
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    USER=${USER_NAME} \
    HOME=/workspace/.container-home \
    OUTPUTS_ROOT=/workspace/outputs \
    DATA_ROOT=/workspace/data \
    SCENARIONET_DATA_ROOT=/workspace/data/scenarionet \
    METADRIVE_DATA_ROOT=/workspace/data/metadrive \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH}

WORKDIR /workspace/thesis-metadrive

# Minimal utilities for interactive dev sessions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gdal-bin \
        libgl1 \
        libgdal-dev \
        libglib2.0-0 \
        tmux \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    existing_group="$(getent group "${HOST_GID}" | cut -d: -f1 || true)"; \
    if [ -z "${existing_group}" ]; then \
        groupadd -g "${HOST_GID}" "${USER_NAME}"; \
        existing_group="${USER_NAME}"; \
    fi; \
    existing_user="$(getent passwd "${HOST_UID}" | cut -d: -f1 || true)"; \
    if [ -z "${existing_user}" ]; then \
        useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${USER_NAME}"; \
    fi

# Install uv and create an isolated project environment.
RUN pip install --no-cache-dir "uv==${UV_VERSION}"
RUN uv venv /opt/venv --python /usr/local/bin/python
COPY pyproject.toml uv.lock ./
COPY third_party ./third_party
RUN uv sync --frozen --no-install-project

# Copy project sources after deps are installed.
COPY . .
RUN mkdir -p /workspace/.container-home /workspace/outputs /workspace/data/scenarionet /workspace/data/metadrive
RUN uv sync --frozen --extra dev

# Keep the shared base free of PyTorch. Dataset preparation uses MetaDrive,
# ScenarioNet, NumPy and PyArrow, but never imports the RL/training stack.

# CPU-only image for ScenarioNet dataset preparation. It deliberately has no
# PyTorch and no CUDA runtime; Waymo conversion remains in Dockerfile.waymo.
FROM base AS pipeline

RUN chown -R "${HOST_UID}:${HOST_GID}" \
    /workspace/.container-home /workspace/outputs /workspace/data

CMD ["bash"]

# The main development image adds the machine-specific PyTorch backend only in
# this target. The pipeline target above reuses the base layers without
# downloading PyTorch or CUDA runtime libraries.
FROM base AS dev

ARG TORCH_VERSION=2.9.1
ARG TORCH_BACKEND=cu128
ARG UV_HTTP_TIMEOUT=300

ENV UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}

# Torch is intentionally installed after uv sync because it comes from the
# dedicated CUDA wheel index and is excluded from the portable project lock.
RUN case "${TORCH_BACKEND}" in cpu|cu126|cu128) ;; *) echo "Unsupported TORCH_BACKEND=${TORCH_BACKEND}" >&2; exit 2;; esac \
    && uv pip install --no-config --python /opt/venv \
        --index-url "https://download.pytorch.org/whl/${TORCH_BACKEND}" \
        "torch==${TORCH_VERSION}"
RUN bash scripts/validate_torch_environment.sh "${TORCH_VERSION}" "${TORCH_BACKEND}"
RUN chown -R "${HOST_UID}:${HOST_GID}" \
    /workspace/.container-home /workspace/outputs /workspace/data

CMD ["bash"]
