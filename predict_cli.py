import sys, json, torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(r"D:\sentiment classifier\models\xlmr")  # your trained model
LABELS = ["negative","neutral","positive"]

def predict_texts(texts):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=1)
    return [{"text": t, "label": LABELS[p], "probs": probs[i].round(4).tolist()} for i,(t,p) in enumerate(zip(texts, preds))]

if __name__ == "__main__":
    if len(sys.argv) == 1:
        texts = [
            "The draft policy is confusing and will increase compliance burden.",
            "बहुत अच्छा कदम है, इससे कारोबार आसान होगा।",
            "এটা তেমন খারাপ না, কিন্তু আরো স্পষ্টতা দরকার।",
        ]
    else:
        # pass a txt file path: one text per line
        path = Path(sys.argv[1])
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = predict_texts(texts)
    print(json.dumps(out, ensure_ascii=False, indent=2))
