import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ["negative", "neutral", "positive"]

def load_split(base_dir: Path, split: str, logits_override: Path | None = None):
    logits_path = Path(logits_override) if logits_override else base_dir / f"logits_{split}.npy"
    labels_path = base_dir / f"labels_{split}.npy"

    if not logits_path.exists():
        raise FileNotFoundError(f"Missing logits for '{split}' split at {logits_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels for '{split}' split at {labels_path}")

    logits = np.load(logits_path)
    labels = np.load(labels_path)
    return logits, labels


def eval_with_bias(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray):
    pred = (logits + bias).argmax(axis=1)
    report = classification_report(labels, pred, target_names=LABELS, digits=4, output_dict=True)

    return {
        "acc": accuracy_score(labels, pred),
        "macro_f1": f1_score(labels, pred, average="macro"),
        "per_class_f1": {LABELS[i]: report[LABELS[i]]["f1-score"] for i in range(len(LABELS))},
        "cm": confusion_matrix(labels, pred, labels=list(range(len(LABELS)))).tolist(),
    }


def grid_search_bias(logits: np.ndarray, labels: np.ndarray, grid: tuple[float, ...]):
    best: dict | None = None
    for b0 in grid:
        for b1 in grid:
            for b2 in grid:
                bias = np.array([b0, b1, b2], dtype=np.float32)
                metrics = eval_with_bias(logits, labels, bias)
                if (best is None) or (metrics["macro_f1"] > best["macro_f1"]):
                    best = {"bias": bias.tolist(), **metrics}
    assert best is not None
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path, help="Directory containing logits/labels npy files")
    parser.add_argument("--val_logits", type=Path, help="Override path for validation logits (npy)")
    parser.add_argument("--test_logits", type=Path, help="Override path for test logits (npy)")
    parser.add_argument("--out_dir", type=Path, help="Directory to write results (defaults to model_dir)")
    parser.add_argument("--grid", type=float, nargs="*", help="Custom bias grid values (defaults to preset)")
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    model_dir = model_dir.expanduser()

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir {model_dir} does not exist")

    grid = tuple(args.grid) if args.grid else (-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0)

    val_logits, val_labels = load_split(model_dir, "val", args.val_logits)
    best = grid_search_bias(val_logits, val_labels, grid)
    print(f"Best bias on VAL: {best['bias']} macro_f1: {best['macro_f1']:.4f}")

    test_logits, test_labels = load_split(model_dir, "test", args.test_logits)
    bias_array = np.array(best["bias"], dtype=np.float32)
    test_metrics = eval_with_bias(test_logits, test_labels, bias_array)
    print(f"\nTEST with tuned bias – acc: {test_metrics['acc']:.4f} macro_f1: {test_metrics['macro_f1']:.4f}")
    print(f"per-class F1: {test_metrics['per_class_f1']}")

    out_dir = args.out_dir.expanduser() if args.out_dir else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "bias_tuned_results.json"
    output_path.write_text(json.dumps({"val_best": best, "test": test_metrics}, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
