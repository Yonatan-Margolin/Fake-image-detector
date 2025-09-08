import sys, os
from pathlib import Path
from PIL import Image
import torch
from facenet_pytorch import MTCNN

in_dir  = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

# one detector on CPU or CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# find images
imgs = [p for p in in_dir.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
for p in imgs:
    try:
        im = Image.open(p).convert('RGB')
        boxes, _ = mtcnn.detect(im)
        if boxes is None or len(boxes)==0:
            continue
        # pick the largest box (area)
        b = max(boxes, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
        x1,y1,x2,y2 = map(int, b)
        face = im.crop((x1,y1,x2,y2))

        # mirror the input folder structure
        rel = p.relative_to(in_dir)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        face.save(dest)
    except Exception as e:
        pass
