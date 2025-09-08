from pathlib import Path
import cv2, torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

# ----- paths -----
ROOT_VIDEOS = Path(r"F:\datasets\CelebDFv2")         # where the MP4s are
CROPS_ROOT  = ROOT_VIDEOS / "crops"
OUT_REAL = CROPS_ROOT / "real"
OUT_FAKE = CROPS_ROOT / "fake"
OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

# ----- sampling settings (tweak later) -----
FPS_SAMPLE = 1                # grab 1 frame per second
MAX_FRAMES_PER_VIDEO = 5      # save up to 5 crops per video
IMG_SIZE = 224                # crop size for training
MAX_VIDEOS = None             # e.g. 300 for a quick test run

def is_fake_dir(p: Path) -> bool:
    s = "/".join(p.parts).lower()
    # Celeb-DF v2 uses names like "Celeb-synthesis" for fakes
    return ("synth" in s) or ("fake" in s)

def save_crop(bgr, box, out_dir: Path, stem: str, idx: int) -> bool:
    x1, y1, x2, y2 = [int(v) for v in box]
    face = bgr[y1:y2, x1:x2]
    if face.size == 0 or min(face.shape[:2]) < 20:
        return False
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(str(out_dir / f"{stem}_{idx:03d}.jpg"), face)
    return True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device)

    vids = list(ROOT_VIDEOS.rglob("*.mp4"))
    if MAX_VIDEOS:
        vids = vids[:MAX_VIDEOS]
    print(f"Found {len(vids)} videos. Writing crops to: {CROPS_ROOT}")

    for v in tqdm(vids):
        out_dir = OUT_FAKE if is_fake_dir(v.parent) else OUT_REAL
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps / FPS_SAMPLE)))
        i = saved = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            if i % step == 0 and saved < MAX_FRAMES_PER_VIDEO:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(Image.fromarray(rgb))
                if boxes is not None and len(boxes):
                    if save_crop(frame, boxes[0], out_dir, v.stem, saved):
                        saved += 1
            if saved >= MAX_FRAMES_PER_VIDEO: break
            i += 1
        cap.release()

if __name__ == "__main__":
    main()
