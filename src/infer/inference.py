import os, sys, json, argparse, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = r"D:\sentiment classifier\models\critical_xlmr_v3"  # change if needed

def load_label_map(model_dir):
    lm_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(lm_path):
        with open(lm_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        id2label = {int(k): v for k, v in lm.get("id2label", {}).items()}
        if id2label:
            return id2label
    # fallback
    return {0: "negative", 1: "neutral", 2: "positive"}

def predict_texts(texts, model, tok, device):
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
    return preds, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Single input text")
    ap.add_argument("--file", type=str, help="Path to a UTF-8 text file with one sentence per line")
    args = ap.parse_args()

    if not args.text and not args.file:
        print("Provide --text \"...\" or --file path.txt"); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    id2label = load_label_map(MODEL_DIR)

    if args.text:
        texts = [args.text.strip()]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]

    preds, probs = predict_texts(texts, model, tok, device)
    out = []
    for t, p, pr in zip(texts, preds, probs):
        out.append({
            "text": t,
            "label": id2label.get(int(p), str(p)),
            "probs": [float(x) for x in pr]  # [neg, neu, pos] in model order
        })
    import json as _json
    print(_json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
