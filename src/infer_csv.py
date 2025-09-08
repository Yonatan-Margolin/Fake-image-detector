# src/infer_csv.py
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train import load_cfg, build_model  # reuse your training helpers
from src.data.dataset import DeepfakeCsvDataset  # <-- correct import path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", required=True)          # input CSV with 'filepath' column
    ap.add_argument("--out", required=True)          # where to write predictions CSV
    ap.add_argument("--threshold", type=float, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # device selection (same logic as train.py)
    dev = cfg["train"].get("device", "cuda")
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    device = torch.device(dev)
    print(f"[infer] Using device: {device}")

    # 1) Read the file list directly from the CSV (robust & order-preserving)
    df_in = pd.read_csv(args.csv)
    if "filepath" not in df_in.columns:
        raise ValueError("Input CSV must contain a 'filepath' column.")
    filepaths = df_in["filepath"].tolist()

    # 2) Dataset & loader (no shuffle!)
    ds = DeepfakeCsvDataset(
        csv_path=args.csv,
        img_size=cfg["data"]["img_size"],
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    # 3) Model + weights
    model = build_model(cfg, device)
    ckpt = torch.load(args.weights, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 4) Predict
    probs = []
    with torch.no_grad():
        for x, _ in tqdm(loader, total=len(loader), unit="batch"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.append(p)
    probs = np.concatenate(probs)

    # 5) Write output CSV
    out_df = pd.DataFrame({"filepath": filepaths, "score": probs})
    if args.threshold is not None:
        out_df["pred"] = (out_df["score"] >= float(args.threshold)).astype(int)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out}")

if __name__ == "__main__":
    main()
