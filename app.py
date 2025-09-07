from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pathlib import Path
import tempfile, os

from inference_util import Wav2LipEngine

BASE_VIDEO = os.getenv("BASE_VIDEO", "/app/assets/base.mp4")
CHECKPOINT = os.getenv("W2L_CKPT", "/app/Wav2Lip/checkpoints/wav2lip.pth")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

app = FastAPI(title="Wav2Lip API", version="1.0.0")
engine = None

@app.on_event("startup")
def load_engine():
    global engine
    if not Path(BASE_VIDEO).exists():
        raise RuntimeError(f"Missing base video at {BASE_VIDEO}")
    if not Path(CHECKPOINT).exists():
        raise RuntimeError(f"Missing checkpoint at {CHECKPOINT}")
    engine = Wav2LipEngine(
        base_video=BASE_VIDEO,
        checkpoint=CHECKPOINT,
        batch_size=BATCH_SIZE
    )

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.post("/lip-sync")
async def lip_sync(audio: UploadFile = File(...)):
    allowed = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/aac", "audio/ogg", "audio/mp4", "application/octet-stream"}
    if (audio.content_type or "application/octet-stream") not in allowed:
        raise HTTPException(400, detail=f"Unsupported audio type: {audio.content_type}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename or '')[-1]) as f:
        f.write(await audio.read())
        audio_path = f.name

    out_path = tempfile.mktemp(suffix=".mp4")
    try:
        engine.synthesize(audio_path, out_path)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    out_name = f"lipsynced_{Path(audio.filename or 'audio').stem}.mp4"
    return FileResponse(out_path, media_type="video/mp4", filename=out_name)
