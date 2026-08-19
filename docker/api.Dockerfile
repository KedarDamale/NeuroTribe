# NeuroTRIBE-HBN — API and worker image.
# Both services share this image; only the command differs.

FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NEUROTRIBE_ROOT=/app

# ffmpeg  -> stimulus probing / frame extraction / synthetic smoke-test clip
# git     -> install and resolve the pinned TRIBE commit for provenance
# curl    -> container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# TRIBE v2 is part of the runnable stack, not a manual host prerequisite.  The
# model package is baked into the image; its much larger pretrained weights are
# fetched once by the `tribe-bootstrap` Compose service into the model-cache
# volume, which survives container and image rebuilds.
ARG TRIBE_REPO=https://github.com/facebookresearch/tribev2
ARG TRIBE_GIT_REF=main
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1+cpu
ARG TORCHVISION_VERSION=0.20.1+cpu
# Pin CPU builds explicitly.  TRIBE accepts torch 2.5.x, but an unpinned
# resolver can choose the much larger generic/CUDA-enabled wheel from PyPI.
RUN pip install --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
    && pip install "git+${TRIBE_REPO}@${TRIBE_GIT_REF}"

# The Docker CLI lets the worker launch the fMRIPrep container on the host
# daemon (the socket is mounted read-write in docker-compose).
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

COPY neurotribe/ ./neurotribe/
COPY apps/api/ ./apps/api/
COPY config/ ./config/
COPY alembic.ini ./
COPY migrations/ ./migrations/
COPY scripts/ ./scripts/

RUN mkdir -p /app/data /app/cache /app/work \
    && useradd --create-home --uid 1000 neurotribe \
    && chown -R neurotribe:neurotribe /app

# Root is retained so the mounted Docker socket is usable; the application
# itself performs no privileged operations.

EXPOSE 8000

# No image-level HEALTHCHECK: this image serves three roles (api, worker, beat)
# and an HTTP check is only meaningful for the api. Each service declares its
# own check in docker-compose.yml.

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# --------------------------------------------------------------------------
FROM base AS dev
RUN pip install -r requirements-dev.txt
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
