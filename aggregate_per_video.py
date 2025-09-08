# aggregate_per_video.py
import argparse
from pathlib import Path
import pandas as pd

def video_id_from_parent(p: Path) -> str:
    # Use the folder containing the crops as the video id
    # e.g. ...\crops\real\<VIDEO_ID>\xxxx.jpg  -> <VIDEO_ID>
    return p.parent.name

def video_id_from_prefix(p: Path, sep="_", parts=1) -> str:
    # Use the filename prefix before the first N separators
    # e.g. 00017_004.jpg with parts=1 -> "00017"
    stem = p.stem
    chunks = stem.split(sep)
    return sep.join(chunks[:parts]) if chunks else stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=r"preds\scored.csv",
                    help="Path to the crop-level scored CSV (from src.infer_csv).")
    ap.add_argument("--out", default=r"preds\video_scores.csv",
                    help="Where to write the per-video scores.")
    ap.add_argument("--mode", choices=["parent","prefix"], default="prefix",
                    help="How to extract a video id from the filepath.")
    ap.add_argument("--prefix-sep", default="_", help="Separator for prefix mode.")
    ap.add_argument("--prefix-parts", type=int, default=1,
                    help="How many parts of the filename prefix to keep.")
    ap.add_argument("--ratio-thr", type=float, default=0.50,
                    help="Threshold on fake-ratio to mark a video as fake.")
    ap.add_argument("--min-crops", type=int, default=1,
                    help="Optional minimum #crops required to score a video.")
    args = ap.parse_args()

    df = pd.read_csv(args.scored)
    # expected cols: filepath, score/prob, pred (binary). We only need 'filepath' and 'pred'
    if "pred" not in df.columns:
        raise ValueError("Expected a 'pred' column (binary 0/1). Re-run infer with threshold to produce it.")

    # Build video id per row
    paths = df["filepath"].apply(lambda s: Path(str(s)))
    if args.mode == "parent":
        df["video"] = paths.apply(video_id_from_parent)
    else:
        df["video"] = paths.apply(lambda p: video_id_from_prefix(p, args.prefix_sep, args.prefix_parts))

    # Aggregate per video
    grp = df.groupby("video")
    agg = grp["pred"].agg(fake_ratio="mean", n_crops="size").reset_index()

    # (optional) filter by min number of crops
    if args.min_crops > 1:
        agg = agg[agg["n_crops"] >= args.min_crops].copy()

    agg["video_pred"] = (agg["fake_ratio"] >= args.ratio_thr).astype(int)

    # Sort for convenience
    agg = agg.sort_values(["video_pred","fake_ratio","n_crops"], ascending=[False, False, False])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f"Wrote {len(agg):,} videos -> {args.out}")

if __name__ == "__main__":
    main()
