from __future__ import annotations

from typing import Dict, List, Sequence


DEFAULT_LABELS = ["SUPPORTED", "CONTRADICTED", "NOT_FOUND", "UNDECIDABLE"]


def canonical_label(label: str | None) -> str:
    text = str(label or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"NOTFOUND", "NOT_FOUND"}:
        return "NOT_FOUND"
    if text in {"SUPPORTED", "CONTRADICTED", "UNDECIDABLE"}:
        return text
    return "NOT_FOUND"


def label_f1(y_true: List[str], y_pred: List[str], labels: Sequence[str] | None = None) -> Dict[str, float]:
    labels = list(labels or DEFAULT_LABELS)
    true_norm = [canonical_label(v) for v in y_true]
    pred_norm = [canonical_label(v) for v in y_pred]
    f1s: Dict[str, float] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(true_norm, pred_norm) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_norm, pred_norm) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_norm, pred_norm) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s[label] = f1
    f1s["macro_f1"] = sum(f1s[l] for l in labels) / len(labels) if labels else 0.0
    return f1s


def evidence_f1(pred_sets: List[List[str]], gold_sets: List[List[str]]) -> float:
    if not gold_sets:
        return 1.0 if not pred_sets else 0.0
    best = 0.0
    for gold in gold_sets:
        gold_set = set(gold)
        for pred in pred_sets or [[]]:
            pred_set = set(pred)
            if not pred_set and not gold_set:
                best = max(best, 1.0)
                continue
            tp = len(pred_set & gold_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(gold_set) if gold_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            best = max(best, f1)
    return best


def fever_score(label_true: str, label_pred: str, pred_sets: List[List[str]], gold_sets: List[List[str]]) -> float:
    true_norm = canonical_label(label_true)
    pred_norm = canonical_label(label_pred)
    if true_norm != pred_norm:
        return 0.0
    if true_norm in {"NOT_FOUND", "UNDECIDABLE"}:
        return 1.0
    if not gold_sets:
        return 0.0
    for gold in gold_sets:
        gold_set = set(gold)
        for pred in pred_sets or [[]]:
            pred_set = set(pred)
            if gold_set.issubset(pred_set):
                return 1.0
    return 0.0
