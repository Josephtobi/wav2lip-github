import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import cv2
import librosa

# Add Wav2Lip repo to path
W2L_ROOT = str(Path(__file__).parent / 'Wav2Lip')
if W2L_ROOT not in sys.path:
    sys.path.insert(0, W2L_ROOT)

# Import model
from Wav2Lip.models import Wav2Lip  # type: ignore

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Wav2LipEngine:
    """
    Minimal, production-minded Wav2Lip wrapper.

    - Detects (or approximates) a face box on the base video
    - Computes mels from audio
    - Runs batched inference on GPU
    - Re-composites frames and muxes audio with FFmpeg
    """

    def __init__(
        self,
        base_video: str,
        checkpoint: str,
        batch_size: int = 8,
        fps_override: Optional[int] = None,
        face_detect_scale: float = 1.1,
        face_detect_min_neighbors: int = 5,
        face_box_fallback_ratio: float = 0.45,
    ):
        self.base_video = base_video
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.fps_override = fps_override
        self.face_detect_scale = face_detect_scale
        self.face_detect_min_neighbors = face_detect_min_neighbors
        self.face_box_fallback_ratio = face_box_fallback_ratio

        self.model = self._load_model(checkpoint)
        self.model.eval()

        # Load Haar cascade for face detect (bundled with OpenCV)
        self.haar = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )

        torch.backends.cudnn.benchmark = True

    # ---------- Utils ----------

    @staticmethod
    def _ffmpeg(cmd: List[str]):
        subprocess.check_call(['ffmpeg', '-y', *cmd],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    @staticmethod
    def _to_wav_16k_mono(src_path: str, dst_path: str):
        subprocess.check_call([
            'ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', '16000', dst_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        model = Wav2Lip()
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state)
        return model.to(DEVICE)

    # ---------- Media IO ----------

    def _extract_audio_wav(self, audio_in: str) -> str:
        out_wav = tempfile.mktemp(suffix='.wav')
        self._to_wav_16k_mono(audio_in, out_wav)
        return out_wav

    def _read_all_frames(self, video_path: str) -> Tuple[List[np.ndarray], int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = int(self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 25)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError("No frames extracted from base video.")
        return frames, fps

    # ---------- Audio features ----------

    @staticmethod
    def _mels(wav_path: str, fps: int) -> List[np.ndarray]:
        wav, sr = librosa.load(wav_path, sr=16000)
        hop_length = int(16000 / fps)
        mel = librosa.feature.melspectrogram(
            y=wav, sr=16000, n_fft=1024, hop_length=hop_length,
            n_mels=80, fmin=55, fmax=7600
        )
        mel = np.log(np.maximum(1e-5, mel))
        mel_chunks = []
        mel_step = 16
        idx = 0
        while True:
            start = idx * mel_step
            if start + mel_step > mel.shape[1]:
                break
            mel_chunks.append(mel[:, start:start+mel_step])
            idx += 1
        if len(mel_chunks) == 0:
            raise RuntimeError("Audio too short after preprocessing.")
        return mel_chunks

    # ---------- Face detection / crop box ----------

    def _largest_face_box(self, frames: List[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
        """
        Try to find the largest face across a sampled subset of frames.
        Returns (x1, y1, x2, y2) or None.
        """
        H_sample = max(1, len(frames)//20)
        best = None
        best_area = 0
        for f in frames[::H_sample]:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(
                gray,
                scaleFactor=self.face_detect_scale,
                minNeighbors=self.face_detect_min_neighbors,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                area = w * h
                if area > best_area:
                    best_area = area
                    best = (x, y, x + w, y + h)
        return best

    def _fallback_center_box(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        If detection fails, use a center crop box (ratio of width/height).
        """
        h, w = frame_shape[:2]
        r = self.face_box_fallback_ratio
        bw, bh = int(w * r), int(h * r)
        x1 = (w - bw)//2
        y1 = (h - bh)//2
        return x1, y1, x1 + bw, y1 + bh

    # ---------- Main pipeline ----------

    def synthesize(self, audio_file: str, out_path: str) -> str:
        # 1) Audio → wav 16k mono
        wav16 = self._extract_audio_wav(audio_file)

        # 2) Video → frames (+ fps)
        frames, fps = self._read_all_frames(self.base_video)

        # 3) Face box
        box = self._largest_face_box(frames)
        if box is None:
            box = self._fallback_center_box(frames[0].shape)
        x1, y1, x2, y2 = box

        # 4) Audio → mels aligned to fps
        mels = self._mels(wav16, fps)

        # 5) Batched inference
        gen_frames: List[np.ndarray] = []
        model = self.model

        for i in range(0, len(mels), self.batch_size):
            mel_batch = mels[i:i + self.batch_size]
            img_batch = []
            for j, _mel in enumerate(mel_batch):
                idx = min(i + j, len(frames) - 1)
                f = frames[idx].copy()
                face = f[y1:y2, x1:x2]
                # Resize to 96x96 as expected by Wav2Lip
                face = cv2.resize(face, (96, 96))
                img = face.astype(np.float32) / 255.0
                img_batch.append(img)

            # Create tensors
            img_batch_np = np.stack(img_batch)  # (B, 96, 96, 3)
            img_batch_np = np.transpose(img_batch_np, (0, 3, 1, 2))  # (B, 3, 96, 96)
            img_t = torch.from_numpy(img_batch_np).float().to(DEVICE)

            mel_batch_np = np.float32(mel_batch)
            mel_batch_np = np.expand_dims(mel_batch_np, 1)  # (B, 1, 80, 16)
            mel_t = torch.from_numpy(mel_batch_np).float().to(DEVICE)

            with torch.no_grad():
                pred = model(mel_t, img_t)  # (B, 3, 96, 96)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for k in range(pred.shape[0]):
                new_face = pred[k].astype(np.uint8)
                # Paste back to original frame size at the detected box
                f = frames[min(i + k, len(frames) - 1)].copy()
                new_face_resized = cv2.resize(new_face, (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = new_face_resized
                gen_frames.append(f)

        # 6) Write temporary silent video
        tmp_vid = tempfile.mktemp(suffix='.mp4')
        h, w, _ = gen_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_vid, fourcc, fps, (w, h))
        for f in gen_frames:
            writer.write(f)
        writer.release()

        # 7) Mux audio with video (shortest)
        self._ffmpeg([
            '-i', tmp_vid, '-i', wav16,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'aac', '-shortest', out_path
        ])

        return out_path
