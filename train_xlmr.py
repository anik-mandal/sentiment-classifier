import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.trainer_callback import EarlyStoppingCallback

print("python:", sys.executable)
import transformers as tfm
print("transformers:", tfm.__version__, "from:", tfm.__file__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "cuda_ver:", getattr(torch.version, "cuda", None))
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# ----- Paths -----
DATA = Path(r"D:\sentiment classifier\data")
OUT  = Path(r"D:\sentiment classifier\models\xlmr"); OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "xlm-roberta-base"
LABELS = ["negative","neutral","positive"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

# ----- Data -----
def load_csv(p_csv: Path):
    df = pd.read_csv(p_csv)
    df = df[df["label"].isin(LABELS)].copy()
    prefix = []
    if "lang" in df.columns:
        prefix = df["lang"].apply(lambda s: f"<lang_{s}> ")
    else:
        prefix = pd.Series([""]*len(df))
    if "source" in df.columns:
        prefix = prefix + df["source"].apply(lambda s: f"<src_{s}> ")
    df["text"] = (prefix + df["text"].astype(str)).astype(str)
    return Dataset.from_pandas(df[["text","label"]])

ds_train = load_csv(DATA/"train.csv")
ds_val   = load_csv(DATA/"val.csv")
ds_test  = load_csv(DATA/"test.csv")

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
MAX_LEN = 192  # lower to 160 if OOM

def tok_fn(b): return tok(b["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
ds_train = ds_train.map(tok_fn, batched=True).class_encode_column("label")
ds_val   = ds_val.map(tok_fn, batched=True).class_encode_column("label")
ds_test  = ds_test.map(tok_fn, batched=True).class_encode_column("label")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
)

def metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    return {"acc": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}

args = TrainingArguments(
    output_dir=str(OUT),
    seed=42,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=True,

    # Explicit + matching strategies (fixes your previous crash)
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",

    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    report_to=[],  # no TB/W&B unless you want it
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tok,
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
print("\nVAL:", trainer.evaluate(ds_val))
print("TEST:", trainer.evaluate(ds_test))

preds = np.argmax(trainer.predict(ds_test).predictions, axis=1)
ytrue = ds_test["label"]
rep = classification_report(ytrue, preds, target_names=LABELS, digits=4)
(Path(OUT/"report_test.txt")).write_text(rep, encoding="utf-8")
print("\nTEST REPORT\n", rep)

trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("Saved model + tokenizer to:", OUT)
