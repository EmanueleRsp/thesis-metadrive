# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG USER_NAME=appuser
ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH}

WORKDIR /workspace/thesis-metadrive

# Minimal utilities for interactive dev sessions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
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

# Install uv and sync locked Python dependencies first for better layer caching.
# `torch` is provided by the NVIDIA base image, so we keep the project env
# aligned with the lockfile while skipping a redundant torch reinstall.
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
COPY --from=third_party metadrive /workspace/third-party/metadrive
RUN uv sync --frozen --no-install-project --no-install-package torch

# Copy project sources after deps are installed.
COPY . .
RUN uv sync --frozen --no-install-package torch
RUN set -eux; \
    metadrive_dir="$(python -c "import importlib.util; from pathlib import Path; spec = importlib.util.find_spec('metadrive'); assert spec is not None and spec.origin is not None, 'metadrive package not found after uv sync'; print(Path(spec.origin).resolve().parent)")"; \
    chown -R "${HOST_UID}:${HOST_GID}" "${metadrive_dir}"

CMD ["bash"]
