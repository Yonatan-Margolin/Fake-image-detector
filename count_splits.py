import os, csv

root = r"F:\datasets\CelebDFv2\crops"

def count(csv_path):
    neg = pos = 0  # neg=real(0), pos=fake(1)
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lbl = int(row["label"])
            if lbl == 1: pos += 1
            else:        neg += 1
    return neg, pos

for split in ("train", "val"):
    csv_path = os.path.join(root, f"{split}.csv")
    neg, pos = count(csv_path)
    print(f"{split}: real={neg}, fake={pos}")
    if split == "train":
        if pos > 0:
            print("train pos_weight (neg/pos) =", neg/pos)
        else:
            print("train pos_weight: undefined (no positives)")
