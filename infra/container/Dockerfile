# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c AS uv

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

ARG PYTHON_VERSION=3.12.13

LABEL org.opencontainers.image.title="A3 OutcomeStack remote training environment" \
      org.opencontainers.image.source="https://github.com/ASkarin/a3-outcome-stack" \
      ai.a3-outcome-stack.base-digest="sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/a3/python \
    UV_PROJECT_ENVIRONMENT=/opt/a3/.venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/a3/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        acl \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        jq \
        libavcodec-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libgl1 \
        libglib2.0-0 \
        libswresample-dev \
        libswscale-dev \
        numactl \
        openssh-client \
        openssh-server \
        pkg-config \
        procps \
        ripgrep \
        rsync \
        sudo \
        tini \
        tmux \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_RETRIES=5 \
    UV_HTTP_TIMEOUT=300 \
    UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

WORKDIR /opt/a3/build
COPY pyproject.toml uv.lock .python-version ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install "${PYTHON_VERSION}" \
    && export GIT_CONFIG_COUNT=1 \
        GIT_CONFIG_KEY_0=http.version \
        GIT_CONFIG_VALUE_0=HTTP/1.1 \
    && sync_args=( \
        --frozen \
        --no-install-project \
        --extra remote-training \
        --group dev \
    ) \
    && for attempt in 1 2 3; do \
        if uv sync "${sync_args[@]}"; then \
            break; \
        fi; \
        if (( attempt == 3 )); then \
            exit 1; \
        fi; \
        sleep "$((attempt * 5))"; \
    done \
    && /opt/a3/.venv/bin/python -c \
        "import importlib.metadata as m; import tensorboard, torch, torchvision; assert m.version('tensorboard') == '2.21.0'; assert torch.__version__ == '2.11.0+cu128'; assert torchvision.__version__ == '0.26.0+cu128'"

ARG SOURCE_COMMIT=unknown
ARG UV_LOCK_SHA256=unknown

LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      ai.a3-outcome-stack.uv-lock-sha256="${UV_LOCK_SHA256}"

ENV PYTHONNOUSERSITE=1

COPY infra/container/bin/ /usr/local/bin/
COPY infra/container/lib/ /usr/local/lib/a3-container/
COPY infra/container/entrypoint.sh /usr/local/sbin/a3-entrypoint
COPY infra/container/profile.sh /etc/profile.d/a3.sh
COPY infra/container/sshd_config /etc/ssh/sshd_config.a3

RUN chmod 0755 \
        /usr/local/bin/a3-artifact-fetch \
        /usr/local/bin/a3-artifact-promote \
        /usr/local/bin/a3-env-doctor \
        /usr/local/bin/a3-gpu-run \
        /usr/local/sbin/a3-entrypoint \
    && chmod 0644 /etc/profile.d/a3.sh /etc/ssh/sshd_config.a3 \
    && mkdir -p /opt/a3/manifest /run/sshd /workspace \
    && /opt/a3/.venv/bin/python /usr/local/lib/a3-container/write_image_manifest.py \
        --output /opt/a3/manifest/environment.json \
        --python-version "${PYTHON_VERSION}" \
        --source-commit "${SOURCE_COMMIT}" \
        --uv-lock-sha256 "${UV_LOCK_SHA256}" \
    && chown -R root:root /opt/a3 /usr/local/lib/a3-container \
    && chmod -R a-w /opt/a3/.venv /opt/a3/python

EXPOSE 22

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/sbin/a3-entrypoint"]
