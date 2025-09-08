import csv, os, glob, random, pathlib, collections
random.seed(1337)

root = r"F:\datasets\CelebDFv2\crops"

# collect all crops we have so far
paths = glob.glob(os.path.join(root, "real", "*.jpg")) + \
        glob.glob(os.path.join(root, "fake", "*.jpg"))

# group by video id (before the _### frame index)
by_vid = collections.defaultdict(list)
for p in paths:
    stem = pathlib.Path(p).stem.rsplit("_", 1)[0]
    by_vid[stem].append(p)

vids = list(by_vid.keys())
random.shuffle(vids)

# 20% of videos into validation (video-disjoint split)
n_val = max(1, int(0.2 * len(vids)))
val_vids = set(vids[:n_val])

def label(p: str) -> int:
    return 1 if "\\fake\\" in p.lower() else 0   # 1=fake, 0=real

train_rows, val_rows = [], []
for v, items in by_vid.items():
    target = val_rows if v in val_vids else train_rows
    target += [(p, label(p)) for p in items]

# write CSVs
for out, rows in [(os.path.join(root, "train.csv"), train_rows),
                  (os.path.join(root, "val.csv"),   val_rows)]:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        w.writerows(rows)

print("train:", len(train_rows), "val:", len(val_rows))
