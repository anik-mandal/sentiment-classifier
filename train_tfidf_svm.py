import pandas as pd, joblib, os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---- Paths ----
DATA = Path(r"D:\sentiment classifier\data")
OUT  = Path(r"D:\sentiment classifier\models\sklearn"); OUT.mkdir(parents=True, exist_ok=True)

# ---- Load ----
train = pd.read_csv(DATA/"train.csv")
val   = pd.read_csv(DATA/"val.csv")
test  = pd.read_csv(DATA/"test.csv")

# ---- Helper: add language token (helps a linear model) ----
def with_lang_prefix(df):
    if "lang" in df.columns:
        return (df["lang"].map(lambda s: f"<{s}> ") + df["text"].astype(str)).tolist()
    return df["text"].astype(str).tolist()

Xtr, ytr = with_lang_prefix(train), train["label"].astype(str)
Xva, yva = with_lang_prefix(val),   val["label"].astype(str)
Xte, yte = with_lang_prefix(test),  test["label"].astype(str)

# ---- Model ----
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95)),
    ("svm",  LinearSVC(class_weight="balanced", random_state=42))
])

pipe.fit(Xtr, ytr)

pred_va = pipe.predict(Xva)
pred_te = pipe.predict(Xte)

rep_val  = classification_report(yva, pred_va, digits=4)
rep_test = classification_report(yte, pred_te, digits=4)

print("\nVAL REPORT\n", rep_val)
print("\nTEST REPORT\n", rep_test)

(Path(OUT/"report_val.txt")).write_text(rep_val, encoding="utf-8")
(Path(OUT/"report_test.txt")).write_text(rep_test, encoding="utf-8")
joblib.dump(pipe, OUT/"model.joblib")
print("Saved model to:", OUT/"model.joblib")
