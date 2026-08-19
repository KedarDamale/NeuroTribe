# GPU-only serving image for a private Hugging Face Inference Endpoint.
# The endpoint receives a stimulus video and returns a float32 NumPy vector of
# TRIBE's mean cortical prediction. It never receives local fMRI data.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    TRIBE_CACHE=/data/tribe \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_DISABLED=true

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install fastapi "uvicorn[standard]" python-multipart numpy pandas \
    && pip install "git+https://github.com/facebookresearch/tribev2@main"

WORKDIR /srv
COPY main.py ./

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
