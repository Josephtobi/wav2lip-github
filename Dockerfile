FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (Python 3.10, ffmpeg, OpenCV libs)
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pull Wav2Lip source (frozen commit for reproducibility)
RUN git clone https://github.com/Rudrabha/Wav2Lip.git && \
    cd Wav2Lip && git checkout 9c551c6

# App code
COPY app.py inference_util.py download_models.sh entrypoint.sh ./
RUN chmod +x download_models.sh entrypoint.sh && ./download_models.sh

# Data dirs
RUN mkdir -p assets outputs

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
