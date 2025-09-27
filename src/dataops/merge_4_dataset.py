import os, sys
import pandas as pd
from collections import Counter

DATA_DIR = r"D:\sentiment classifier\data"
FILES = [
    os.path.join(DATA_DIR, "corpus_all.csv"),
    os.path.join(DATA_DIR, "sarcasm_1500_gov_project.csv"),
    os.path.join(DATA_DIR, "sarcasm_1500_gov_project_v2_neg.csv"),
    os.path.join(DATA_DIR, "sarcasm_1500_gov_project_v3_neg.csv"),
]
OUT_PATH = os.path.join(DATA_DIR, "critical_training_dataset_v3_nodup_balanced.csv")
RANDOM_STATE = 42

def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv": return pd.read_csv(path)
    if ext in (".xlsx", ".xls"): return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")

def pick_text_label(df):
    cols = {c.lower().strip(): c for c in df.columns}
    t_keys = ["text","comment","comments","sentence","utterance","review","content","feedback"]
    y_keys = ["label","labels","sentiment","polarity","label_fine","class"]
    t = next((cols[k] for k in t_keys if k in cols), None)
    y = next((cols[k] for k in y_keys if k in cols), None)
    if not t or not y: raise ValueError(f"Missing text/label columns: {list(df.columns)}")
    out = df[[t,y]].copy(); out.columns = ["text","label"]; return out

def map_to_3(s):
    m = {
        "negative":"negative","neutral":"neutral","positive":"positive",
        "neg":"negative","neu":"neutral","pos":"positive",
        "negative, sarcasm":"negative",
        "neutral/conditional":"neutral","neutral/mixed reporting":"neutral","mixed":"neutral",
        "slightly positive / neutral":"neutral",
    }
    s1 = s.astype(str).str.strip().str.lower()
    return s1.map(m).fillna(s1)

def load_all(paths):
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Missing: {p}"); sys.exit(1)
        df = read_any(p)
        df = pick_text_label(df)
        df["label"] = map_to_3(df["label"])
        df = df[df["label"].isin(["negative","neutral","positive"])].dropna(subset=["text"])
        frames.append(df)
        print(f"Included: {os.path.basename(p)} -> {len(df)} rows")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["text","label"]).reset_index(drop=True)
    return merged

def rebalance_no_dup(df, seed=RANDOM_STATE):
    counts = Counter(df["label"])
    target = min(counts.values())
    blocks = []
    for lab in ["negative","neutral","positive"]:
        sub = df[df["label"]==lab]
        if len(sub) >= target:
            blocks.append(sub.sample(n=target, random_state=seed, replace=False))
        else:
            blocks.append(sub)
            print(f"Warning: '{lab}' has only {len(sub)} < target {target}; keeping all (no duplication).")
    out = pd.concat(blocks, axis=0, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out

def main():
    merged = load_all(FILES)
    print("Merged counts:", Counter(merged["label"]))
    balanced = rebalance_no_dup(merged)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    balanced.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print("Saved:", OUT_PATH)
    print("Final counts:", Counter(balanced["label"]))
    print("Total rows:", len(balanced))

if __name__ == "__main__":
    main()
