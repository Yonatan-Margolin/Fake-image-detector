# eval_val_preds.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report

VAL_CSV   = r"F:\datasets\CelebDFv2\crops\val.csv"     # ground-truth (filepath,label)
PRED_CSV  = r"preds\val_preds.csv"                     # your inference output
OUT_DIR   = r"preds"

df_val  = pd.read_csv(VAL_CSV)
df_pred = pd.read_csv(PRED_CSV)
df = df_val.merge(df_pred, on="filepath", how="inner")

y_true  = df["label"].astype(int).values
y_score = df["score"].values

auc = roc_auc_score(y_true, y_score)

grid = np.linspace(0.01, 0.99, 99)
f1s  = []
for t in grid:
    y_hat = (y_score >= t).astype(int)
    f1s.append(f1_score(y_true, y_hat))
best_t = float(grid[int(np.argmax(f1s))])

y_hat = (y_score >= best_t).astype(int)
acc = accuracy_score(y_true, y_hat)
f1  = f1_score(y_true, y_hat)
cm  = confusion_matrix(y_true, y_hat)

print(f"AUC: {auc:.4f}")
print(f"Best threshold (by F1): {best_t:.3f}")
print(f"ACC: {acc:.4f}  F1: {f1:.4f}")
print("Confusion matrix [ [TN FP], [FN TP] ]:\n", cm)
print("\nClassification report:\n", classification_report(y_true, y_hat, digits=4))

df["pred_bin"] = y_hat
fp = df[(df.label==0) & (df.pred_bin==1)]
fn = df[(df.label==1) & (df.pred_bin==0)]
fp.to_csv(f"{OUT_DIR}/false_positives.csv", index=False)
fn.to_csv(f"{OUT_DIR}/false_negatives.csv", index=False)
print(f"\nSaved {len(fp)} false_positives -> {OUT_DIR}/false_positives.csv")
print(f"Saved {len(fn)} false_negatives -> {OUT_DIR}/false_negatives.csv")
