from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .metrics import DEFAULT_LABELS, canonical_label, evidence_f1, fever_score, label_f1


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_evidence_sets(row: Dict[str, Any]) -> List[List[str]]:
    sets = []
    for ev_set in row.get("evidence_sets", []):
        if not isinstance(ev_set, list):
            continue
        ids = []
        for ev in ev_set:
            if not isinstance(ev, dict):
                continue
            eobj_id = ev.get("eobj_id")
            if eobj_id:
                ids.append(str(eobj_id))
        sets.append(ids)
    return sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="comma-separated labels for Macro-F1 (default: SUPPORTED,CONTRADICTED,NOT_FOUND,UNDECIDABLE)",
    )
    args = parser.parse_args()

    labels = [canonical_label(x) for x in args.labels.split(",") if x.strip()]
    labels = [x for x in labels if x in DEFAULT_LABELS]
    if not labels:
        labels = list(DEFAULT_LABELS)

    gold_rows = {r["claim_id"]: r for r in read_jsonl(args.gold)}
    pred_rows = {r["claim_id"]: r for r in read_jsonl(args.pred)}

    y_true: List[str] = []
    y_pred: List[str] = []
    ev_scores: List[float] = []
    fever_scores: List[float] = []

    for claim_id, gold in gold_rows.items():
        pred = pred_rows.get(claim_id, {"label": "NOT_FOUND", "evidence_sets": []})
        gold_label = canonical_label(gold.get("label"))
        pred_label = canonical_label(pred.get("label"))
        y_true.append(gold_label)
        y_pred.append(pred_label)
        gold_sets = extract_evidence_sets(gold)
        pred_sets = extract_evidence_sets(pred)
        ev_scores.append(evidence_f1(pred_sets, gold_sets))
        fever_scores.append(fever_score(gold_label, pred_label, pred_sets, gold_sets))

    label_scores = label_f1(y_true, y_pred, labels=labels)
    label_support = Counter(y_true)
    result = {
        **label_scores,
        "evidence_f1": sum(ev_scores) / max(len(ev_scores), 1),
        "fever_score": sum(fever_scores) / max(len(fever_scores), 1),
        "n_samples": len(y_true),
        "label_support": {k: label_support.get(k, 0) for k in DEFAULT_LABELS},
        "scored_labels": labels,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
