# Wav2Lip API – Dockerized FastAPI Service

A production-ready FastAPI service that lip-syncs a fixed base video to user-uploaded audio using Wav2Lip (PyTorch).

## Features

- **Input**: Audio files (.wav, .mp3, .aac, .ogg, etc.)
- **Output**: Lip-synced MP4 video
- **Optimized for**: RunPod GPU pods with CUDA 12.1
- **Dockerized**: Ready for deployment with GitHub Container Registry

## Quick Start

### 1. Add Your Base Video

Place your base video file in the `assets/` directory and name it `base.mp4`:

```bash
cp /path/to/your/video.mp4 assets/base.mp4
```

### 2. Build and Run Locally

```bash
# Build the Docker image
docker build -t wav2lip-api .

# Run with GPU support
docker run --rm --gpus all -p 8000:8000 \
  -v $(pwd)/assets:/app/assets \
  -e BASE_VIDEO=/app/assets/base.mp4 \
  wav2lip-api
```

### 3. Test the API

```bash
curl -X POST "http://localhost:8000/lip-sync" \
  -F "audio=@sample.wav" \
  -o output.mp4
```

## Deployment to RunPod

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial Wav2Lip API setup"
git push origin main
```

The GitHub Actions workflow will automatically build and push the Docker image to GitHub Container Registry.

### 2. Deploy on RunPod

1. Go to [RunPod](https://runpod.io)
2. Create a new **Dedicated GPU Pod** with **HTTP** template
3. Use the image: `ghcr.io/YOUR_USERNAME/video-gen-ai:latest`
4. Set port: `8000`
5. Add environment variables:
   - `BASE_VIDEO=/app/assets/base.mp4` (if mounting)
   - `W2L_CKPT=/app/Wav2Lip/checkpoints/wav2lip.pth` (default)
6. Mount volume `/app/assets` and upload your `base.mp4`

### 3. Health Check

Visit: `https://your-pod-id-8000.proxy.runpod.net/health`

## API Endpoints

- `GET /health` - Health check
- `POST /lip-sync` - Upload audio file and get lip-synced video

## Configuration

Environment variables:

- `BASE_VIDEO`: Path to base video (default: `/app/assets/base.mp4`)
- `W2L_CKPT`: Path to Wav2Lip checkpoint (default: `/app/Wav2Lip/checkpoints/wav2lip.pth`)
- `BATCH_SIZE`: Inference batch size (default: `8`)
- `UVICORN_WORKERS`: Number of workers (default: `1`)

## Requirements

- Python 3.10
- CUDA 12.1
- PyTorch with CUDA support
- FFmpeg
- OpenCV

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `torch.cuda.is_available() == False` | CUDA wheel/base mismatch | Use this Dockerfile (CUDA 12.1 base + cu121 wheels) |
| `ImportError: cv2` | OpenCV missing | Keep opencv-python==4.10.0.84 installed |
| `ffmpeg not found` | Missing system ffmpeg | Dockerfile installs ffmpeg |
| Output is slow | Large base video | Use 720p template; tune BATCH_SIZE |
| RunPod can't pull image | GHCR private | Make GHCR package public, or add registry secret |

## License

This project uses the Wav2Lip model. Please refer to the original [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip) for licensing information.
