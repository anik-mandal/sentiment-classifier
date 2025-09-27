import os, json, random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

CSV_PATH = r"D:\sentiment classifier\data\critical_training_dataset_v4.csv"
OUT_DIR  = r"D:\sentiment classifier\models\critical__xlmr_V4"
LOG_DIR  = r"D:\sentiment classifier\logs\critical__xlmr_V4"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)
df = df[["text","label"]].copy()
df["label"] = df["label"].astype(str).str.strip().str.lower()

labels = ["negative","neutral","positive"]
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
df["label_id"] = df["label"].map(label2id)
assert df["label_id"].notnull().all(), "Found labels outside {negative,neutral,positive}"

# 85/15 split (stratified)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(sss.split(df.index, df["label_id"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df   = df.iloc[val_idx].reset_index(drop=True)

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id
)

MAX_LEN = 256

def encode_dataframe(df_):
    enc = tokenizer(
        df_["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    x = {k: torch.tensor(v) for k, v in enc.items()}
    y = torch.tensor(df_["label_id"].tolist())

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(y)
        def __getitem__(self, i):
            return {"input_ids": x["input_ids"][i],
                    "attention_mask": x["attention_mask"][i],
                    "labels": y[i]}
    return DS()

train_ds = encode_dataframe(train_df)
val_ds   = encode_dataframe(val_df)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }

args = TrainingArguments(
    output_dir=OUT_DIR,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,           # you said 2–3 epochs; start at 3
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    logging_dir=LOG_DIR,
    logging_steps=100,
    save_steps=1000,
    seed=SEED,
    fp16=torch.cuda.is_available(),   # mixed precision only if CUDA
    report_to=["none"],               # no wandb/tensorboard needed
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print("Validation:", metrics)

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
with open(os.path.join(OUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
