import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.trainer_callback import EarlyStoppingCallback
from torch.utils.data import DataLoader, WeightedRandomSampler

print("python:", sys.executable)
import transformers as tfm
print("transformers:", tfm.__version__, "from:", tfm.__file__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "cuda_ver:", getattr(torch.version,"cuda",None))
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# ----- Paths -----
DATA = Path(r"D:\sentiment classifier\data")
OUT  = Path(r"D:\sentiment classifier\models\xlmr_weighted"); OUT.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "xlm-roberta-base"

LABELS = ["negative","neutral","positive"]
lab2id = {l:i for i,l in enumerate(LABELS)}
id2lab = {i:l for l,i in lab2id.items()}

# ----- Load data -----
def load_csv(p_csv: Path):
    df = pd.read_csv(p_csv)
    df = df[df["label"].isin(LABELS)].copy()
    # helpful special tokens
    pref = []
    if "lang" in df.columns:  pref = df["lang"].apply(lambda s: f"<lang_{s}> ")
    else:                     pref = pd.Series([""]*len(df))
    if "source" in df.columns: pref = pref + df["source"].apply(lambda s: f"<src_{s}> ")
    df["text"] = (pref + df["text"].astype(str)).astype(str)
    return df

df_tr = load_csv(DATA/"train.csv")
df_va = load_csv(DATA/"val.csv")
df_te = load_csv(DATA/"test.csv")

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
MAX_LEN = 192  # raise to 256 if you have VRAM

def tok_map(df):
    ds = Dataset.from_pandas(df[["text","label"]])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=MAX_LEN), batched=True)
    ds = ds.class_encode_column("label")  # -> 0..K-1
    return ds

ds_tr = tok_map(df_tr)
ds_va = tok_map(df_va)
ds_te = tok_map(df_te)

# ----- Class weights (from training labels) -----
y_tr = df_tr["label"].map(lab2id).values
classes = np.arange(len(LABELS))
weights_vec = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr).astype("float32")
print("class weights (neg,neu,pos):", weights_vec.tolist())
weights_t = torch.tensor(weights_vec, dtype=torch.float)

# ----- Weighted loss (override compute_loss) -----
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, train_sampler=None, **kwargs):
        self.class_weights = class_weights
        self._external_train_sampler = train_sampler
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_f = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_f(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self._remove_unused_columns(self.train_dataset, description="training")

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            sampler = self._external_train_sampler if self._external_train_sampler is not None else self._get_train_sampler()
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self._external_train_sampler is not None:
            raise ValueError("Passing a custom sampler is not supported for iterable datasets.")

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ----- Balanced sampler (oversample rare classes a bit) -----
# per-example weights inversely proportional to class freq
freq = np.bincount(y_tr, minlength=len(LABELS))
inv_freq = 1.0 / np.maximum(freq, 1)
ex_weights = inv_freq[y_tr]
sampler = WeightedRandomSampler(weights=torch.tensor(ex_weights, dtype=torch.double), num_samples=len(ex_weights), replacement=True)

# ----- Model -----
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=id2lab, label2id=lab2id
)

# ----- Args -----
args = TrainingArguments(
    output_dir=str(OUT),
    seed=42,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    num_train_epochs=4,                         # try 4–5 with early stopping
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=True,

    eval_strategy="epoch",                # keep eval/save aligned
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",

    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    report_to=[],
)

def metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=ds_tr,
    eval_dataset=ds_va,
    tokenizer=tok,
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    class_weights=weights_t,
    train_sampler=sampler,
)

trainer.train()
print("\nVAL:", trainer.evaluate(ds_va))
print("TEST:", trainer.evaluate(ds_te))

# Save
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("Saved model to:", OUT)
