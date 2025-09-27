import os, sys
import pandas as pd
from collections import Counter

DATA_DIR = r"D:\sentiment classifier\data"
V3_CANDIDATES = [
    os.path.join(DATA_DIR, "critical_training_dataset_v3.csv"),
    os.path.join(DATA_DIR, "critical_training_dataset_v3")
]
NEW_NEG_PATH = os.path.join(DATA_DIR, "sarcasm_2000_gov_project_v4_neg.csv")   # all negative
NEW_POS_PATH = os.path.join(DATA_DIR, "sarcasm_1000_gov_project_v5_pos.csv")   # all positive

OUT_V4_RAW   = os.path.join(DATA_DIR, "critical_training_dataset_v4.csv")
OUT_V4_STRAT = os.path.join(DATA_DIR, "critical_training_dataset_v4.csv")
SEED = 42

def pick_v3():
    for p in V3_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("critical_training_dataset_v3(.csv) not found")

def read_any(p):
    ext = os.path.splitext(p)[1].lower()
    if ext == ".csv": return pd.read_csv(p)
    if ext in (".xlsx",".xls"): return pd.read_excel(p)
    raise ValueError(f"Unsupported file type: {p}")

def pick_text_label(df):
    cols = {c.lower().strip(): c for c in df.columns}
    t_keys = ["text","comment","comments","sentence","utterance","review","content","feedback"]
    y_keys = ["label","labels","sentiment","polarity","label_fine","class"]
    t = next((cols[k] for k in t_keys if k in cols), None)
    y = next((cols[k] for k in y_keys if k in cols), None)
    if not t or not y: raise ValueError(f"Missing text/label columns: {list(df.columns)}")
    out = df[[t,y]].copy(); out.columns = ["text","label"]; return out

def map3(s):
    m = {
        "negative":"negative","neutral":"neutral","positive":"positive",
        "neg":"negative","neu":"neutral","pos":"positive",
        "negative, sarcasm":"negative",
        "neutral/conditional":"neutral","neutral/mixed reporting":"neutral","mixed":"neutral",
        "slightly positive / neutral":"neutral",
    }
    s1 = s.astype(str).str.strip().str.lower()
    return s1.map(m).fillna(s1)

def load_file(p):
    df = read_any(p)
    df = pick_text_label(df)
    df["label"] = map3(df["label"])
    df = df[df["label"].isin(["negative","neutral","positive"])].dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    return df

def add_key(df):
    return df.assign(_k = df["text"].str.lower().str.strip() + "||" + df["label"].str.lower())

def main():
    v3p = pick_v3()
    for p in [v3p, NEW_NEG_PATH, NEW_POS_PATH]:
        if not os.path.exists(p): raise FileNotFoundError(p)

    base   = add_key(load_file(v3p))
    newneg = add_key(load_file(NEW_NEG_PATH))   # all negative
    newpos = add_key(load_file(NEW_POS_PATH))   # all positive

    # Merge & dedup (on exact text+label)
    merged = pd.concat([base, newneg, newpos], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text","label"]).reset_index(drop=True)
    merged = add_key(merged)  # ensure _k exists post-dedup

    os.makedirs(os.path.dirname(OUT_V4_RAW), exist_ok=True)
    merged.sample(frac=1, random_state=SEED).to_csv(OUT_V4_RAW, index=False, encoding="utf-8")
    print("Saved raw:", OUT_V4_RAW)
    print("Merged counts:", Counter(merged["label"]))

    # Compute target = smallest class AFTER merge
    counts = Counter(merged["label"])
    target = min(counts.values())

    # Build seed sets using composite key membership (not DataFrame index!)
    merged_keys = set(merged["_k"])
    seed_neg = newneg[newneg["_k"].isin(merged_keys)]           # keep all new negatives that survived dedup
    seed_pos = newpos[newpos["_k"].isin(merged_keys)]           # keep all new positives that survived dedup

    # Pools = merged minus the seeds (by key), per class
    seed_neg_keys = set(seed_neg["_k"])
    seed_pos_keys = set(seed_pos["_k"])

    neg_pool = merged[(merged["label"]=="negative") & (~merged["_k"].isin(seed_neg_keys))]
    pos_pool = merged[(merged["label"]=="positive") & (~merged["_k"].isin(seed_pos_keys))]
    neu_pool = merged[(merged["label"]=="neutral")]

    # How many more do we need to reach target per class?
    need_neg = max(0, target - len(seed_neg))
    need_pos = max(0, target - len(seed_pos))
    need_neu = target

    if len(neg_pool) < need_neg: print(f"Warning: negative pool {len(neg_pool)} < need {need_neg}; will cap at pool size.")
    if len(pos_pool) < need_pos: print(f"Warning: positive pool {len(pos_pool)} < need {need_pos}; will cap at pool size.")
    if len(neu_pool) < need_neu: print(f"Warning: neutral pool {len(neu_pool)} < need {need_neu}; will cap at pool size.")

    neg_take = neg_pool.sample(n=min(len(neg_pool), need_neg), random_state=SEED, replace=False)
    pos_take = pos_pool.sample(n=min(len(pos_pool), need_pos), random_state=SEED, replace=False)
    neu_take = neu_pool.sample(n=min(len(neu_pool), need_neu), random_state=SEED, replace=False)

    neg_final = pd.concat([seed_neg, neg_take], axis=0, ignore_index=True)
    pos_final = pd.concat([seed_pos, pos_take], axis=0, ignore_index=True)
    neu_final = neu_take

    # If any class ended up smaller (pool too small), reduce all to common min (no duplication)
    final_min = min(len(neg_final), len(neu_final), len(pos_final))
    neg_final = neg_final.sample(n=final_min, random_state=SEED, replace=False)
    neu_final = neu_final.sample(n=final_min, random_state=SEED, replace=False)
    pos_final = pos_final.sample(n=final_min, random_state=SEED, replace=False)

    balanced = pd.concat([neg_final, neu_final, pos_final], axis=0, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=SEED).drop(columns=["_k"]).reset_index(drop=True)
    balanced.to_csv(OUT_V4_STRAT, index=False, encoding="utf-8")

    # Report how many new rows made it
    bal_keys = set((balanced["text"].str.lower().str.strip() + "||" + balanced["label"].str.lower()).tolist())
    kept_new_neg = len(seed_neg_keys & bal_keys)
    kept_new_pos = len(seed_pos_keys & bal_keys)

    print("Saved balanced (stratified):", OUT_V4_STRAT)
    print("Balanced counts:", Counter(balanced["label"]))
    print("Total rows (balanced):", len(balanced))
    print(f"Included new negatives: {kept_new_neg} / {len(seed_neg)}")
    print(f"Included new positives: {kept_new_pos} / {len(seed_pos)}")

if __name__ == "__main__":
    main()
