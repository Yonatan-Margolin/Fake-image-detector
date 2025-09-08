# app/inference.py
import os
import yaml
import torch
import torchvision.transforms as T
from typing import List, Tuple
from PIL import Image

# optional face cropper (largest face); enabled via env CROP_FACES=mtcnn
try:
    from facenet_pytorch import MTCNN  # type: ignore
except Exception:  # pragma: no cover
    MTCNN = None  # facenet-pytorch not installed

# Try SimpleEffB0 first; fall back to EffB0Detector if your file uses that name
try:
    from src.models.effb0 import SimpleEffB0 as _Backbone
except Exception:
    from src.models.effb0 import EffB0Detector as _Backbone  # type: ignore

# ------------------------------
# device / globals
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL: torch.nn.Module | None = None     # lazy singleton
_CFG: dict | None = None                  # last-resort cfg dict
_IMG_SIZE: int = 224                      # resolved when the model loads
_TRANSFORM: T.Compose | None = None       # built after _IMG_SIZE is known
_MTCNN: MTCNN | None = None               # lazy init face detector


def _build_transform(img_size: int) -> T.Compose:
    """Keep this consistent with your training preprocessing."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # If you trained with ImageNet normalization, uncomment:
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _maybe_init_mtcnn() -> None:
    global _MTCNN
    if _MTCNN is None and os.getenv("CROP_FACES", "none").lower() == "mtcnn" and MTCNN is not None:
        # keep_all to pick largest; runs on same device
        _MTCNN = MTCNN(keep_all=True, device=DEVICE)


def _crop_largest_face(img: Image.Image) -> Image.Image:
    """Return a face crop (largest box + 20% margin) if MTCNN is enabled, else original."""
    if _MTCNN is None:
        return img
    import numpy as np

    boxes, _ = _MTCNN.detect(img)  # boxes: [N, 4] (x1,y1,x2,y2)
    if boxes is None or len(boxes) == 0:
        return img

    # pick largest area
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    box = boxes[int(np.argmax(areas))].tolist()
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = img.size
    # add margin
    mx = int(0.2 * (x2 - x1))
    my = int(0.2 * (y2 - y1))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    try:
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return img


# ------------------------------
# model build / load
# ------------------------------
def _build_model_from_cfg(cfg: dict) -> torch.nn.Module:
    m = cfg.get("model", {})
    model = _Backbone(
        pretrained=False,
        in_chans=m.get("in_chans", 3),
        num_classes=m.get("num_classes", 1),  # MUST be 1 for your binary head
        drop_rate=m.get("drop_rate", 0.2),
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )
    model.to(DEVICE)
    return model


def _load_model(weights_path: str, cfg_path: str | None = None) -> Tuple[torch.nn.Module, dict]:
    """Loads checkpoint + cfg, builds the model to match the training head, then
    loads weights with strict=True (avoids silent mismatches)."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"WEIGHTS_PATH not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=DEVICE)
    cfg = ckpt.get("cfg")

    # try explicit CONFIG_PATH, else default
    if cfg is None:
        if cfg_path is None:
            cfg_path = os.getenv("CONFIG_PATH", "/app/configs/effb0.yaml")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            # last-resort defaults if no yaml exists
            cfg = {
                "model": {"in_chans": 3, "num_classes": 1, "drop_rate": 0.2, "drop_path_rate": 0.1},
                "data": {"img_size": 224},
            }

    # resolve img size early (with default)
    global _IMG_SIZE
    _IMG_SIZE = int(cfg.get("data", {}).get("img_size", 224))

    model = _build_model_from_cfg(cfg)

    # state can be under "model" key or at root
    state = ckpt["model"] if "model" in ckpt else ckpt
    # strict load to guarantee the right head is used
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def _get_model() -> torch.nn.Module:
    """Lazy singleton: build once, reuse across requests."""
    global _MODEL, _CFG, _TRANSFORM
    if _MODEL is None:
        weights_path = os.getenv("WEIGHTS_PATH", "/app/app/weights/model.pt")
        cfg_path = os.getenv("CONFIG_PATH")  # may be None
        model, cfg = _load_model(weights_path, cfg_path)
        _TRANSFORM = _build_transform(_IMG_SIZE)
        _MODEL = model
        _CFG = cfg
        _maybe_init_mtcnn()
    return _MODEL


# ------------------------------
# public API
# ------------------------------
@torch.no_grad()
def predict_images(pil_images: List[Image.Image]) -> Tuple[List[float], List[int]]:
    """
    Args:
        pil_images: list of PIL.Image

    Returns:
        (probabilities, labels) where probabilities are in [0,1] and
        labels are 0=real, 1=fake using THRESHOLD (default 0.5).
    """
    if not pil_images:
        return [], []

    model = _get_model()
    assert _TRANSFORM is not None

    # build batch
    tensors = []
    for im in pil_images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        # optional face crop
        im = _crop_largest_face(im)
        tensors.append(_TRANSFORM(im))

    batch = torch.stack(tensors, dim=0).to(DEVICE)

    # forward
    logits = model(batch)                # shape [N, 1] or [N]
    probs = torch.sigmoid(logits).view(-1).detach().cpu().tolist()

    thr = float(os.getenv("THRESHOLD", "0.5"))
    labels = [1 if p >= thr else 0 for p in probs]

    return probs, labels
