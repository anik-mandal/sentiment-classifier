import os, json, random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

CSV_PATH = r"D:\sentiment classifier\data\critical_training_dataset_v3.csv"
OUT_DIR  = r"D:\sentiment classifier\models\critical_xlmr_v3"
LOG_DIR  = r"D:\sentiment classifier\logs\critical_xlmr_v3"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED=42
random.seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)
df = df[["text","label"]].copy()
df["label"] = df["label"].str.strip().str.lower()
labels = ["negative","neutral","positive"]
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
df["label_id"] = df["label"].map(label2id)
assert df["label_id"].notnull().all()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tr, va = next(sss.split(df.index, df["label_id"]))
train_df, val_df = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)

MODEL = "xlm-roberta-base"
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3, id2label=id2label, label2id=label2id)

MAX_LEN=256
def encode(df_):
    enc = tok(df_["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    enc = {k: torch.tensor(v) for k,v in enc.items()}
    y = torch.tensor(df_["label_id"].tolist())
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(y)
        def __getitem__(self, i):
            return {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": y[i]}
    return DS()

train_ds, val_ds = encode(train_df), encode(val_df)

def metrics_fn(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": float((y_pred==y_true).mean()),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted")
    }

args = TrainingArguments(
    output_dir=OUT_DIR,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    logging_dir=LOG_DIR,
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    seed=SEED,
    report_to=["none"]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tokenizer=tok),
    compute_metrics=metrics_fn
)

trainer.train()
metrics = trainer.evaluate()
print("Validation:", metrics)
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)
with open(os.path.join(OUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
