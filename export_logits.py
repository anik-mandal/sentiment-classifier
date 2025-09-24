import argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def export(split, model_dir: Path, data_dir: Path, max_len=192, batch=64):
    df = pd.read_csv(data_dir/f"{split}.csv")
    texts = df["text"].astype(str).tolist()
    tok = AutoTokenizer.from_pretrained(model_dir)
    m   = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = next(m.parameters()).device
    outs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            outs.append(m(**enc).logits.detach().cpu().numpy())
    logits = np.concatenate(outs, axis=0)
    np.save(model_dir/f"logits_{split}.npy", logits)
    print(f"saved {split} logits:", model_dir/f"logits_{split}.npy")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir",  default=r"D:\sentiment classifier\data")
    ap.add_argument("--max_len",   type=int, default=192)
    ap.add_argument("--batch",     type=int, default=64)
    args = ap.parse_args()
    export("val",  Path(args.model_dir), Path(args.data_dir), args.max_len, args.batch)
    export("test", Path(args.model_dir), Path(args.data_dir), args.max_len, args.batch)
    # also store labels once for B
    dfv = pd.read_csv(Path(args.data_dir)/"val.csv"); dft = pd.read_csv(Path(args.data_dir)/"test.csv")
    import numpy as np
    lab2id = {"negative":0,"neutral":1,"positive":2}
    np.save(Path(args.model_dir)/"labels_val.npy",  dfv["label"].map(lab2id).values)
    np.save(Path(args.model_dir)/"labels_test.npy", dft["label"].map(lab2id).values)
