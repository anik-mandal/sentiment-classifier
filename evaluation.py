import os, gc, json, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====== PATHS (your setup) ======
DATA_DIR  = Path(r"D:\sentiment classifier\data")
MODEL_DIR = Path(r"D:\sentiment classifier\models\xlmr")
TEST_CSV  = DATA_DIR / "test.csv"

# ====== LABEL SPACE ======
LABELS = ["negative", "neutral", "positive"]
LAB2ID = {l:i for i,l in enumerate(LABELS)}
ID2LAB = {i:l for l,i in LAB2ID.items()}

# ====== RUNTIME / MEMORY KNOBS ======
MAX_LEN     = 160     # 128–192; lower if memory tight
BATCH_SIZE  = 32      # 16–64; lower if GPU OOM
CHUNK_ROWS  = 4000    # how many rows to stream per CSV chunk (lower if host RAM tight)

# ====== DEVICE / MODEL / TOKENIZER ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}  torch={torch.__version__}  cuda_ver={getattr(torch.version, 'cuda', None)}")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")

@torch.inference_mode()
def predict_batch(texts: List[str]) -> np.ndarray:
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    logits = model(**enc).logits
    preds  = logits.argmax(-1).detach().cpu().numpy()
    del enc, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds

def stream_preds(texts_iter: List[str], batch_size: int) -> np.ndarray:
    preds_all = []
    batch = []
    for t in texts_iter:
        batch.append(t)
        if len(batch) >= batch_size:
            preds_all.append(predict_batch(batch))
            batch = []
    if batch:
        preds_all.append(predict_batch(batch))
    return np.concatenate(preds_all, axis=0) if preds_all else np.array([], dtype=int)

def metrics_overall(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "report": classification_report(
            y_true, y_pred, target_names=LABELS, digits=4, output_dict=False
        ),
        "report_dict": classification_report(
            y_true, y_pred, target_names=LABELS, digits=4, output_dict=True
        ),
    }

def save_confusion_matrix(cm: np.ndarray, out_path: Path, labels: List[str]):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (test)")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    # ---- FIRST PASS (accumulate predictions & truths) ----
    y_true_all, y_pred_all = [], []
    n_seen = 0

    usecols = ["text", "label", "lang"]
    # If lang not present, pandas will ignore that column
    cols_available = pd.read_csv(TEST_CSV, nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in cols_available]

    for chunk in pd.read_csv(TEST_CSV, chunksize=CHUNK_ROWS, usecols=usecols):
        texts = chunk["text"].astype(str).tolist()
        y_true = chunk["label"].map(LAB2ID).values
        y_pred = stream_preds(texts, BATCH_SIZE)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        n_seen += len(y_true)
        print(f"Processed {n_seen} rows")

        del chunk, texts, y_true, y_pred
        gc.collect()

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # ---- OVERALL ----
    overall = metrics_overall(y_true_all, y_pred_all)
    print("\n=== OVERALL REPORT ===")
    print(overall["report"])

    # Confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(LABELS))))
    cm_path = MODEL_DIR / "confusion_matrix.png"
    save_confusion_matrix(cm, cm_path, LABELS)
    print("Saved confusion matrix:", cm_path)

    # Save metrics JSON
    (MODEL_DIR / "metrics_overall.json").write_text(
        json.dumps({
            "accuracy": overall["accuracy"],
            "macro_f1": overall["macro_f1"],
            "micro_f1": overall["micro_f1"],
        }, indent=2),
        encoding="utf-8",
    )

    # ---- BY LANGUAGE (if available) ----
    if "lang" in cols_available:
        print("\n=== BY LANGUAGE ===")
        # read once for grouping (only needed columns)
        test_small = pd.read_csv(TEST_CSV, usecols=["text","label","lang"])
        by_lang_rows = []
        for lg, df in test_small.groupby("lang"):
            y = df["label"].map(LAB2ID).values
            p = stream_preds(df["text"].astype(str).tolist(), BATCH_SIZE)
            rep = classification_report(y, p, target_names=LABELS, digits=4, output_dict=True)
            print(f"\n[lang={lg}] n={len(df)}")
            print(classification_report(y, p, target_names=LABELS, digits=4))
            by_lang_rows.append({
                "lang": lg,
                "n": int(len(df)),
                "accuracy": float(accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, average="macro")),
                "micro_f1": float(f1_score(y, p, average="micro")),
                **{f"f1_{k}": float(v["f1-score"]) for k,v in rep.items() if k in LABELS}
            })
        pd.DataFrame(by_lang_rows).sort_values("n", ascending=False)\
            .to_csv(MODEL_DIR / "metrics_by_lang.csv", index=False, encoding="utf-8")
        print("Saved by-language metrics:", MODEL_DIR / "metrics_by_lang.csv")

    # ---- Save full per-class report to file ----
    (MODEL_DIR / "classification_report.txt").write_text(overall["report"], encoding="utf-8")
    print("Saved overall metrics:", MODEL_DIR / "metrics_overall.json")
    print("Saved classification report:", MODEL_DIR / "classification_report.txt")

if __name__ == "__main__":
    main()
