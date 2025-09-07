#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/Wav2Lip/checkpoints
cd /app/Wav2Lip/checkpoints

# Stable baseline checkpoint (recommended)
if [ ! -f wav2lip.pth ]; then
  echo "Downloading wav2lip.pth ..."
  curl -L -o wav2lip.pth \
    "https://huggingface.co/rippertnt/wav2lip/resolve/main/checkpoints/wav2lip.pth?download=true"
fi

# Optional GAN checkpoint (crisper but can flicker)
if [ ! -f wav2lip_gan.pth ]; then
  echo "Downloading wav2lip_gan.pth ..."
  curl -L -o wav2lip_gan.pth \
    "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth"
fi
