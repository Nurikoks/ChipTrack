# GPU image (Linux host with NVIDIA Container Toolkit). Weights are mounted at /models
# (download once with scripts/download_weights.py, HF_HOME=/models); the container runs offline.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FORTE_WORK_DIR=/tmp/forte
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /srv
COPY requirements.txt .
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt
COPY app ./app
COPY data/rules_registry.json ./data/rules_registry.json
EXPOSE 8000
# FORTE_API_KEY must be provided at runtime (docker run -e FORTE_API_KEY=...)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
