import os
import sys
import json
import time
import math
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ----------------------------
# Constants and configuration
# ----------------------------
MODEL_DIR_PRIMARY = r"D:\sentiment classifier\models\xlmr_weighted"
MODEL_DIR_FALLBACK = r"D:\sentiment classifier\models\xlmr"
RUNTIME_DIR = r"D:\sentiment classifier\runtime"
OPERATING_POINT_PATH = r"D:\sentiment classifier\releases\v0_2_calibrated\operating_point.json"

LABELS = ["negative", "neutral", "positive"]


def ensure_runtime_dir() -> None:
    if not os.path.exists(RUNTIME_DIR):
        os.makedirs(RUNTIME_DIR, exist_ok=True)


def get_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def select_model_dir(prefer_primary: bool = True) -> str:
    def is_valid_model_dir(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        model_files = ["model.safetensors", "pytorch_model.bin"]
        for mf in model_files:
            if os.path.exists(os.path.join(path, mf)):
                return True
        for name in os.listdir(path):
            if name.startswith("checkpoint-"):
                if os.path.exists(os.path.join(path, name, "model.safetensors")) or os.path.exists(
                    os.path.join(path, name, "pytorch_model.bin")
                ):
                    return True
        return False

    primary_ok = is_valid_model_dir(MODEL_DIR_PRIMARY)
    fallback_ok = is_valid_model_dir(MODEL_DIR_FALLBACK)

    if prefer_primary and primary_ok:
        return MODEL_DIR_PRIMARY
    if not prefer_primary and fallback_ok:
        return MODEL_DIR_FALLBACK
    if primary_ok:
        return MODEL_DIR_PRIMARY
    return MODEL_DIR_FALLBACK


def read_bias_vector_if_any(dir_path: str) -> Optional[np.ndarray]:
    bias_json_path = os.path.join(dir_path, "bias_tuned_results.json")
    if not os.path.exists(bias_json_path):
        return None
    try:
        with open(bias_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        val_best = data.get("val_best", {})
        bias = val_best.get("bias")
        if isinstance(bias, list) and len(bias) == len(LABELS):
            return np.array(bias, dtype=np.float32)
    except Exception:
        return None
    return None


def load_model(dir_path: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, Optional[np.ndarray], str]:
    actual_dir = select_model_dir(prefer_primary=(dir_path == MODEL_DIR_PRIMARY))
    tokenizer = AutoTokenizer.from_pretrained(actual_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(actual_dir, local_files_only=True)
    bias_vec = read_bias_vector_if_any(actual_dir)
    return tokenizer, model, bias_vec, actual_dir


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits -= logits.max()
    exp = np.exp(logits)
    total = exp.sum()
    if total <= 0:
        return np.ones_like(logits) / len(logits)
    return (exp / total).astype(np.float32)


def load_operating_point(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
            if "class_order" not in data:
                raise ValueError("operating_point.json missing class_order")
            return data
    # fallback minimal configuration
    return {
        "class_order": LABELS,
        "modes": {
            "balanced": {"temperature": 1.0, "bias": [0.0, 0.0, 0.0], "thresholds": {}},
            "high_precision_negative": {
                "temperature": 0.9,
                "bias": [0.3, 0.0, -0.15],
                "thresholds": {"negative": 0.75}
            },
        },
        "copy": {
            "caption": "Balanced thresholds vs stricter NEG thresholds.",
            "next_actions": {
                "negative": "Flag for review",
                "neutral": "Defer / needs context",
                "positive": "Acknowledge / consider support",
            },
        },
    }


def get_mode_config(mode: str, operating_point: Dict) -> Dict:
    modes = operating_point.get("modes", {})
    if mode in modes:
        return modes[mode]
    return modes.get("balanced", {})


# ----------------------------
# Model helpers
# ----------------------------
@torch.inference_mode()
def predict_batch(
    texts: List[str],
    tok: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    max_len: int,
    bias: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(texts) == 0:
        return (
            np.array([], dtype=np.int64),
            np.zeros((0, len(LABELS)), dtype=np.float32),
            np.zeros((0, len(LABELS)), dtype=np.float32),
        )
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    model = model.to(device)
    logits = model(**enc).logits
    if bias is not None:
        bias_t = torch.tensor(bias, device=logits.device)
        logits = logits + bias_t
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    logits_np = logits.detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds, probs, logits_np


def iter_minibatches(items: List[str], batch_size: int) -> Iterator[List[str]]:
    total = len(items)
    for start in range(0, total, batch_size):
        yield items[start : start + batch_size]


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_dataframe_to_runtime(df: pd.DataFrame, prefix: str) -> str:
    ensure_runtime_dir()
    fname = f"{prefix}_{timestamp_str()}.csv"
    out_path = os.path.join(RUNTIME_DIR, fname)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def run_streaming_prediction(
    texts: List[str],
    tok: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    max_len: int,
    batch_size: int,
    bias: Optional[np.ndarray],
) -> Iterator[pd.DataFrame]:
    columns = ["text", "pred", "p_negative", "p_neutral", "p_positive"]
    running_rows = []
    for chunk in iter_minibatches(texts, batch_size):
        preds, probs, _ = predict_batch(chunk, tok, model, device, max_len, bias=bias)
        for text, pred, prob in zip(chunk, preds, probs):
            row = {
                "text": text,
                "pred": LABELS[int(pred)],
                "p_negative": float(round(float(prob[0]), 3)),
                "p_neutral": float(round(float(prob[1]), 3)),
                "p_positive": float(round(float(prob[2]), 3)),
            }
            running_rows.append(row)
        yield pd.DataFrame(running_rows, columns=columns)


def read_csv_robust(csv_path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(csv_path)


def autodetect_text_column(df: pd.DataFrame) -> Optional[str]:
    for name in df.columns:
        if str(name).strip().lower() == "text":
            return name
    candidate_cols = []
    for name in df.columns:
        series = df[name]
        if pd.api.types.is_numeric_dtype(series):
            continue
        try:
            lengths = series.astype(str).fillna("").map(len)
            mean_len = float(lengths.mean()) if len(lengths) > 0 else 0.0
            candidate_cols.append((name, mean_len))
        except Exception:
            continue
    if not candidate_cols:
        return None
    candidate_cols.sort(key=lambda x: x[1], reverse=True)
    return candidate_cols[0][0]


def extract_pdf_pages(pdf_path: str) -> List[str]:
    return []


# ----------------------------
# Attribution helpers
# ----------------------------
def _tokens_to_words(tokenizer: AutoTokenizer, input_ids: torch.Tensor, attributions: np.ndarray) -> List[Tuple[str, float]]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    words: List[Tuple[str, float]] = []
    current_word = ""
    current_score = 0.0
    count = 0
    for token, score in zip(tokens, attributions):
        if token in {tokenizer.cls_token, tokenizer.sep_token, "<s>", "</s>"}:
            continue
        clean = token.replace("Ġ", " ").replace("▁", " ")
        if clean.startswith(" "):
            if current_word:
                words.append((current_word, current_score / max(count, 1)))
            current_word = clean.strip()
            current_score = score
            count = 1
        else:
            current_word += clean
            current_score += score
            count += 1
    if current_word:
        words.append((current_word, current_score / max(count, 1)))
    return words


def compute_token_attributions(text: str, target_index: int) -> List[Dict[str, float]]:
    if STATE.tokenizer is None or STATE.model is None:
        return []
    tokenizer = STATE.tokenizer
    model = STATE.model
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=STATE.max_length,
            return_attention_mask=True,
        )
        inputs = {k: v.to(STATE.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"].clone().detach()
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        embeddings.retain_grad()
        embeddings.requires_grad_(True)
        outputs = model(inputs_embeds=embeddings, attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits
        target = logits[:, target_index].sum()
        model.zero_grad(set_to_none=True)
        target.backward()
        grads = embeddings.grad.detach().cpu().numpy()[0]
        embeds = embeddings.detach().cpu().numpy()[0]
        scores = (grads * embeds).sum(axis=1)
        tokens = _tokens_to_words(tokenizer, input_ids, scores)
        ranked = sorted(tokens, key=lambda x: abs(x[1]), reverse=True)
        top = []
        for word, score in ranked:
            clean_word = word.strip()
            if not clean_word or clean_word in {"<s>", "</s>"}:
                continue
            if clean_word.lower().startswith("##"):
                clean_word = clean_word[2:]
            top.append({"token": clean_word, "weight": float(score)})
            if len(top) == 5:
                break
        return top
    except Exception:
        return []


# ----------------------------
# API server state
# ----------------------------
class AppState:
    def __init__(self):
        self.prefer_weighted: bool = True
        self.device: torch.device = get_device()
        self.max_length: int = 160
        self.batch_size: int = 32
        self.apply_bias: bool = True
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.bias_vec: Optional[np.ndarray] = None
        self.model_dir: str = select_model_dir(prefer_primary=True)
        self.operating_point: Dict = load_operating_point(OPERATING_POINT_PATH)
        self.class_order: List[str] = self.operating_point.get("class_order", LABELS)


STATE = AppState()


def reload_model(prefer_weighted: bool, apply_bias: bool) -> Tuple[str, str]:
    STATE.prefer_weighted = prefer_weighted
    preferred_dir = MODEL_DIR_PRIMARY if prefer_weighted else MODEL_DIR_FALLBACK
    tokenizer, model, bias_vec, actual_dir = load_model(preferred_dir)
    STATE.tokenizer = tokenizer
    STATE.model = model.to(STATE.device).eval()
    STATE.bias_vec = bias_vec if apply_bias else None
    STATE.model_dir = actual_dir
    STATE.apply_bias = apply_bias
    return preferred_dir, actual_dir


def summarize_state() -> str:
    bias_str = "none"
    if STATE.bias_vec is not None:
        bias_str = np.array2string(STATE.bias_vec, precision=4)
    lines = [
        f"device: {STATE.device}",
        f"selected_model_dir: {STATE.model_dir}",
        f"prefer_weighted: {STATE.prefer_weighted}",
        f"bias_vector: {bias_str}",
    ]
    return "\n".join(lines)


def apply_operating_point(logits: np.ndarray, mode: str) -> Tuple[int, np.ndarray, np.ndarray]:
    op = STATE.operating_point
    class_order = STATE.class_order
    mode_cfg = get_mode_config(mode, op)
    bias = np.array(mode_cfg.get("bias", [0.0, 0.0, 0.0]), dtype=np.float32)
    temperature = float(mode_cfg.get("temperature", 1.0))
    adjusted = logits.copy()
    if STATE.bias_vec is not None and STATE.bias_vec.shape == adjusted.shape:
        adjusted = adjusted + STATE.bias_vec
    if bias.shape == adjusted.shape:
        adjusted = adjusted + bias
    temperature = max(temperature, 1e-3)
    calibrated_logits = adjusted / temperature
    calibrated_probs = softmax_np(calibrated_logits)
    raw_probs = softmax_np(logits)
    top_idx = int(np.argmax(calibrated_probs))
    thresholds = mode_cfg.get("thresholds", {})
    threshold = float(thresholds.get(class_order[top_idx], 0.0))
    if calibrated_probs[top_idx] < threshold and class_order[top_idx] != "neutral":
        neutral_idx = class_order.index("neutral") if "neutral" in class_order else top_idx
        top_idx = neutral_idx
    return top_idx, calibrated_probs, raw_probs


def predict_with_mode(text: str, mode: str, meta: Optional[Dict] = None) -> Dict:
    if STATE.tokenizer is None or STATE.model is None:
        raise RuntimeError("Model not loaded")
    tokenizer = STATE.tokenizer
    model = STATE.model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=STATE.max_length)
    inputs = {k: v.to(STATE.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits.detach().cpu().numpy()[0]
    index, calibrated_probs, raw_probs = apply_operating_point(logits, mode)
    label = STATE.class_order[index]
    top3_idx = np.argsort(calibrated_probs)[::-1][:3]
    top_classes = [
        {"label": STATE.class_order[i], "confidence": float(calibrated_probs[i])}
        for i in top3_idx
    ]
    confidences = {STATE.class_order[i]: float(calibrated_probs[i]) for i in range(len(STATE.class_order))}
    probabilities = {STATE.class_order[i]: float(raw_probs[i]) for i in range(len(STATE.class_order))}
    top_tokens = compute_token_attributions(text, index)
    remarks = generate_remarks(label, confidences[label], top_tokens, mode, meta)
    return {
        "label": label,
        "confidence": float(confidences[label]),
        "confidences": confidences,
        "probabilities": probabilities,
        "top_classes": top_classes,
        "top_tokens": top_tokens,
        "mode": mode,
        "remarks": remarks,
        "operating_point": {
            "class_order": STATE.class_order,
            "mode": mode,
        },
    }


def aggregate_top_tokens(token_lists: List[List[Dict[str, float]]], top_k: int = 5) -> List[Dict[str, float]]:
    scores: Dict[str, float] = {}
    for tokens in token_lists:
        for item in tokens:
            tok = item.get("token")
            weight = float(item.get("weight", 0.0))
            if not tok:
                continue
            scores[tok] = scores.get(tok, 0.0) + abs(weight)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"token": tok, "weight": float(score)} for tok, score in ranked[:top_k]]


def generate_remarks(label: str, confidence: float, top_tokens: List[Dict[str, float]], mode: str, meta: Optional[Dict]) -> Dict[str, str]:
    caption = STATE.operating_point.get("copy", {}).get("caption", "Balanced thresholds vs stricter NEG thresholds.")
    next_actions = STATE.operating_point.get("copy", {}).get(
        "next_actions",
        {
            "negative": "Flag for review",
            "neutral": "Defer / needs context",
            "positive": "Acknowledge / consider support",
        },
    )
    summary = f"This comment is likely {label} with calibrated confidence {confidence * 100:.1f}%."
    tokens = ", ".join([t["token"] for t in top_tokens[:2]]) or "n/a"
    rationale = f"Driven by tokens: {tokens}."
    doc_line = ""
    if meta:
        source = meta.get("source")
        if source == "pdf":
            pages = meta.get("pages")
            if pages:
                doc_line = f"Derived from PDF pages {pages}."
        elif source == "csv":
            count = meta.get("count")
            if count:
                doc_line = f"Aggregate across {count} CSV rows."
    next_action_line = f"Next action: {next_actions.get(label, 'Monitor')}"
    pieces = [summary, rationale]
    if doc_line:
        pieces.append(doc_line)
    pieces.append(next_action_line)
    copy_text = "\n".join(pieces)
    return {
        "summary": summary,
        "rationale": rationale,
        "doc": doc_line,
        "next_action": next_action_line,
        "caption": caption,
        "copy_text": copy_text,
    }


def aggregate_predictions(texts: List[str], mode: str, meta: Optional[Dict]) -> Dict:
    tokenizer = STATE.tokenizer
    model = STATE.model
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded")
    preds, probs, logits = predict_batch(texts, tokenizer, model, STATE.device, STATE.max_length, bias=None)
    calibrated_list = []
    raw_prob_list = []
    top_tokens_all: List[List[Dict[str, float]]] = []
    class_idx_counts = np.zeros(len(STATE.class_order), dtype=np.float32)
    for text, logit in zip(texts, logits):
        index, calibrated, raw = apply_operating_point(logit, mode)
        calibrated_list.append(calibrated)
        raw_prob_list.append(raw)
        class_idx_counts[index] += 1
        top_tokens_all.append(compute_token_attributions(text, index))
    calibrated_mean = np.mean(calibrated_list, axis=0)
    raw_mean = np.mean(raw_prob_list, axis=0)
    top_idx = int(np.argmax(calibrated_mean))
    label = STATE.class_order[top_idx]
    top_classes = [
        {"label": STATE.class_order[i], "confidence": float(calibrated_mean[i])}
        for i in np.argsort(calibrated_mean)[::-1][:3]
    ]
    confidences = {STATE.class_order[i]: float(calibrated_mean[i]) for i in range(len(STATE.class_order))}
    probabilities = {STATE.class_order[i]: float(raw_mean[i]) for i in range(len(STATE.class_order))}
    top_tokens = aggregate_top_tokens(top_tokens_all)
    merged_meta = dict(meta or {})
    merged_meta.setdefault("count", len(texts))
    remarks = generate_remarks(label, confidences[label], top_tokens, mode, merged_meta)
    return {
        "label": label,
        "confidence": float(confidences[label]),
        "confidences": confidences,
        "probabilities": probabilities,
        "top_classes": top_classes,
        "top_tokens": top_tokens,
        "mode": mode,
        "remarks": remarks,
        "operating_point": {
            "class_order": STATE.class_order,
            "mode": mode,
        },
        "counts": {STATE.class_order[i]: float(class_idx_counts[i]) for i in range(len(STATE.class_order))},
    }


# ----------------------------
# API models
# ----------------------------
class PredictPayload(BaseModel):
    text: str
    mode: Optional[str] = "balanced"
    meta: Optional[Dict] = None


class BatchPredictPayload(BaseModel):
    texts: List[str]
    mode: Optional[str] = "balanced"
    meta: Optional[Dict] = None


# ----------------------------
# FastAPI application
# ----------------------------
def create_api() -> FastAPI:
    app = FastAPI(title="Sentiment API", version="1.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "message": "Sentiment API running",
            "endpoints": {
                "POST /predict": {"body": {"text": "<string>", "mode": "balanced|high_precision_negative"}},
                "POST /predict/batch": {"body": {"texts": ["<string>"], "mode": "balanced|high_precision_negative"}},
                "GET /operating-point": {},
                "GET /diagnostics": {},
            },
        }

    @app.get("/operating-point")
    def get_operating_point():
        return {
            "class_order": STATE.class_order,
            "modes": STATE.operating_point.get("modes", {}),
            "copy": STATE.operating_point.get("copy", {}),
        }

    @app.post("/predict")
    def predict(payload: PredictPayload):
        text = (payload.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")
        mode = (payload.mode or "balanced").strip() or "balanced"
        meta = payload.meta or {}
        result = predict_with_mode(text, mode, meta)
        probabilities = result.get("probabilities", {})
        result["scores"] = {
            "positive": probabilities.get("positive", 0.0),
            "neutral": probabilities.get("neutral", 0.0),
            "negative": probabilities.get("negative", 0.0),
        }
        return result

    @app.post("/predict/batch")
    def predict_batch_endpoint(payload: BatchPredictPayload):
        if not payload.texts:
            raise HTTPException(status_code=400, detail="texts must not be empty")
        mode = (payload.mode or "balanced").strip() or "balanced"
        meta = payload.meta or {}
        result = aggregate_predictions(payload.texts, mode, meta)
        probabilities = result.get("probabilities", {})
        result["scores"] = {
            "positive": probabilities.get("positive", 0.0),
            "neutral": probabilities.get("neutral", 0.0),
            "negative": probabilities.get("negative", 0.0),
        }
        return result

    @app.get("/diagnostics")
    def diagnostics():
        return {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device": str(STATE.device),
            "model_dir": STATE.model_dir,
            "bias": None if STATE.bias_vec is None else [float(x) for x in STATE.bias_vec.tolist()],
            "operating_point_loaded": bool(STATE.operating_point),
        }

    return app


def main() -> None:
    ensure_runtime_dir()
    prefer_weighted_default = os.path.isdir(MODEL_DIR_PRIMARY)
    STATE.device = get_device()
    reload_model(prefer_weighted_default, apply_bias=True)
    STATE.operating_point = load_operating_point(OPERATING_POINT_PATH)
    STATE.class_order = STATE.operating_point.get("class_order", LABELS)
    print("============================================")
    print("Multilingual Sentiment (XLM-R) - API Server")
    print(f"Device: {STATE.device}")
    print(f"Selected model dir: {STATE.model_dir}")
    print(f"Operating point loaded: {OPERATING_POINT_PATH}")
    print("API available at: http://127.0.0.1:7860/predict")
    print("============================================")

    api = create_api()
    uvicorn.run(api, host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
