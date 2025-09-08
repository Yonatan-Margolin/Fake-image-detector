# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import io, os, shutil, subprocess, glob, tempfile
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import torch
from torchvision.transforms.functional import to_pil_image  # NEW

# your existing helpers
from app.schemas import PredictResponse
from app.inference import predict_images

CROP_FACES = os.getenv("CROP_FACES", "none").lower()   # 'none' or 'mtcnn'
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "/app/app/weights/model.pt")
THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))     # video aggregation threshold
VIDEO_FPS = float(os.getenv("VIDEO_FPS", "1"))        # frames per second extracted by ffmpeg
_MTCNN = None

def _prepare_image_from_upload(f: UploadFile) -> Image.Image:
    """Load an UploadFile -> PIL.Image and (optionally) crop face via MTCNN."""
    im = _pil_from_upload(f)  # this already converts to RGB

    mtcnn = _get_mtcnn() if CROP_FACES == "mtcnn" else None
    if mtcnn is not None:
        face = mtcnn(im)  # torch.Tensor [3,H,W] or None
        if face is not None:
            # clamp to [0,1] and convert to PIL for the rest of the pipeline
            im = to_pil_image(face.clamp(0, 1))
            return im

    # fallback: no face found or cropping disabled -> use original image
    return im

def _get_mtcnn() -> MTCNN | None:
    global _MTCNN
    if _MTCNN is None and CROP_FACES == "mtcnn":
        _MTCNN = MTCNN(
            keep_all=False,
            select_largest=True,
            image_size=224,
            margin=14,
            post_process=True,
            device="cpu",     # <= keep MTCNN on CPU
        )
    return _MTCNN


app = FastAPI(title="Deepfake Image Detector")

def _pil_from_upload(f: UploadFile) -> Image.Image:
    if not f.content_type or not f.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"{f.filename}: not an image/* upload")
    data = f.file.read()  # UploadFile exposes .file
    return Image.open(io.BytesIO(data)).convert("RGB")

def _frames_from_video_upload(f: UploadFile, fps: float) -> List[Image.Image]:
    """Save an uploaded video, extract frames with ffmpeg, and (optionally) crop faces with MTCNN."""
    if not f.content_type or not f.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail=f"{f.filename}: not a video/* upload")

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Save video to disk
        vid_path = Path(tmp) / f.filename
        with open(vid_path, "wb") as out:
            f.file.seek(0)
            shutil.copyfileobj(f.file, out)

        # 2) Extract frames with ffmpeg
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir(exist_ok=True)
        frame_pat = str(frames_dir / (Path(f.filename).stem + "_%05d.jpg"))

        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",  # cleaner logs
            "-y",
            "-i", str(vid_path),
            "-vf", f"fps={fps}",
            frame_pat,
        ]
        subprocess.run(cmd, check=True)

        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            raise HTTPException(status_code=400, detail=f"No frames extracted from {f.filename}")

        # 3) Optional face-cropping
        mtcnn = _get_mtcnn() if CROP_FACES == "mtcnn" else None

        frames: List[Image.Image] = []
        for p in frame_files:
            # Always ensure RGB + close file handle promptly
            with Image.open(p) as img:
                im = img.convert("RGB")

            if mtcnn is not None:
                face = mtcnn(im)  # torch.Tensor [3,H,W] on device OR None
                if face is None:
                    # If you prefer to keep cadence, replace 'continue' with 'pass' here.
                    continue
                # Move tensor to CPU for PIL conversion and clamp to [0,1]
                im = to_pil_image(face.detach().cpu().clamp(0, 1))

            frames.append(im)

        return frames


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(files: List[UploadFile] = File(...)):
    """Score one or more images."""
    imgs = [_pil_from_upload(f) for f in files]
    probs, labels = predict_images(imgs)
    return PredictResponse(probabilities=probs, labels=labels)

@app.post("/predict/video")
async def predict_video(files: List[UploadFile] = File(...)):
    """Score one or more videos. Aggregates per-frame scores to a single score/label."""
    results = []
    for f in files:
        frames = _frames_from_video_upload(f, fps=VIDEO_FPS)
        probs, _ = predict_images(frames)
        score = float(np.median(probs))                 # aggregation: median
        label = int(score >= THRESHOLD)
        results.append({
            "filename": f.filename,
            "frames_scored": len(probs),
            "score": score,
            "label": label,                             # 1=fake, 0=real (consistent w/ your training)
        })
    return {"videos": results}

@app.post("/predict")
async def predict_mixed(files: List[UploadFile] = File(...)):
    """Accepts both images and videos in one request."""
    images, videos = [], []
    for f in files:
        ct = (f.content_type or "").lower()
        if ct.startswith("image/"):
            images.append(f)
        elif ct.startswith("video/"):
            videos.append(f)
        else:
            raise HTTPException(415, detail=f"{f.filename}: unsupported content-type {ct}")

    payload = {}

    if images:
        imgs = [_prepare_image_from_upload(f) for f in images]
        probs, labels = predict_images(imgs)
        payload["images"] = [
            {"filename": f.filename, "score": float(p), "label": int(l)}
            for f, p, l in zip(images, probs, labels)
        ]

    if videos:
        vids = []
        for f in videos:
            frames = _frames_from_video_upload(f, fps=VIDEO_FPS)
            probs, _ = predict_images(frames)
            score = float(np.median(probs))
            label = int(score >= THRESHOLD)
            vids.append({
                "filename": f.filename,
                "frames_scored": len(probs),
                "score": score,
                "label": label,
            })
        payload["videos"] = vids

    return payload
