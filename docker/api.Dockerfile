# NeuroTRIBE-HBN — API and worker image.
# Both services share this image; only the command differs.

FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NEUROTRIBE_ROOT=/app

# ffmpeg  -> stimulus probing / frame extraction / synthetic smoke-test clip
# git     -> resolving the pinned TRIBE commit for provenance
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
