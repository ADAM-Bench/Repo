#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from reviewground.utils import load_yaml, normalize_text, sentence_split, simple_tokenize, write_jsonl
from tqdm import tqdm


ANCHOR_RE = re.compile(r"\b(?:Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?|Algorithm|Section|Sec\.?|Appendix)\s*[A-Za-z0-9\-\(\)]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
RANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:-|\u2013|to)\s*\d+(?:\.\d+)?%?\b")
PVAL_RE = re.compile(r"\bp\s*[<=>]\s*0?\.\d+\b", re.IGNORECASE)
CI_RE = re.compile(r"\b(?:ci|confidence interval|conf\.? interval)\b", re.IGNORECASE)
STD_RE = re.compile(r"\b(?:std|st\.?dev|standard deviation|variance|mean)\b", re.IGNORECASE)
METRIC_RE = re.compile(
    r"\b(?:accuracy|acc|f1|precision|recall|bleu|rouge|map|auc|perplexity|error rate|rmse|mae|mse)\b",
    re.IGNORECASE,
)
COMPARE_RE = re.compile(
    r"\b(outperform|underperform|improve|increase|decrease|drop|gain|better|worse|significant|significantly)\b",
    re.IGNORECASE,
)
ABLATION_RE = re.compile(r"\b(ablation|without|w/o|removing|variant|baseline)\b", re.IGNORECASE)
MISSING_RE = re.compile(r"\b(lack|missing|omit|absent|unclear|limitation|error)\b", re.IGNORECASE)
METHOD_RE = re.compile(
    r"\b(the paper claims|the paper proposes|we propose|our method|our model|approach|framework|algorithm)\b",
    re.IGNORECASE,
)
DATASET_RE = re.compile(r"\b(dataset|benchmark|corpus|train|test|dev|validation)\b", re.IGNORECASE)
EFFICIENCY_RE = re.compile(r"\b(runtime|latency|speed|complexity|memory|parameters|O\([^)]+\))\b", re.IGNORECASE)
THEORY_RE = re.compile(r"\b(theorem|proposition|lemma|proof|guarantee|bound)\b", re.IGNORECASE)
REPRO_RE = re.compile(r"\b(code|repo|implementation|hyperparameter|seed)\b", re.IGNORECASE)
TRIGGER_RULES = {
    "anchors": [ANCHOR_RE],
    "quantitative": [NUMBER_RE, RANGE_RE, PVAL_RE, CI_RE, STD_RE],
    "metrics": [METRIC_RE],
    "comparison": [COMPARE_RE],
    "ablation": [ABLATION_RE],
    "missing_or_limitations": [MISSING_RE],
    "method_claim": [METHOD_RE],
    "dataset": [DATASET_RE],
    "efficiency": [EFFICIENCY_RE],
    "significance": [PVAL_RE, CI_RE, STD_RE],
    "theory": [THEORY_RE],
    "repro": [REPRO_RE],
}


def detect_triggers(sentence: str) -> Dict[str, object]:
    triggers = set()
    anchors = list({m.group(0).strip() for m in ANCHOR_RE.finditer(sentence)})
    numbers = list({m.group(0) for m in NUMBER_RE.finditer(sentence)})
    for label, patterns in TRIGGER_RULES.items():
        for pattern in patterns:
            if pattern.search(sentence):
                triggers.add(label)
                break
    return {
        "triggers": sorted(triggers),
        "anchors": anchors,
        "numbers": numbers,
    }


def is_candidate(text: str, min_tokens: int) -> Tuple[bool, List[Dict[str, object]], List[str]]:
    tokens = simple_tokenize(text)
    if len(tokens) < min_tokens:
        return False, [], []
    sentences = sentence_split(text)
    trigger_sentences: List[Dict[str, object]] = []
    trigger_types: set[str] = set()
    for sent in sentences:
        sent = normalize_text(sent)
        if not sent:
            continue
        info = detect_triggers(sent)
        if info["triggers"]:
            trigger_types.update(info["triggers"])
            trigger_sentences.append({
                "text": sent,
                "triggers": info["triggers"],
                "anchors": info["anchors"],
                "numbers": info["numbers"],
            })
    return bool(trigger_sentences), trigger_sentences, sorted(trigger_types)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build.yaml")
    parser.add_argument("--manifest", default=None, help="optional manifest jsonl to filter paper_ids")
    parser.add_argument("--context-before", type=int, default=2, help="number of preceding utterances for dialogue_context")
    parser.add_argument("--context-after", type=int, default=0, help="number of following utterances for dialogue_context")
    parser.add_argument("--order", choices=["cdate", "original"], default="cdate", help="utterance ordering for context window")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    min_tokens = cfg["quality_filters"]["min_tokens_per_utterance"]
    threads_path = Path(cfg["paths"]["threads"])
    manifest_ids = None
    if args.manifest:
        manifest_ids = set()
        with Path(args.manifest).open("r", encoding="utf-8") as mf:
            for line in mf:
                if not line.strip():
                    continue
                row = json.loads(line)
                pid = row.get("paper_id") or row.get("forum_id")
                if pid:
                    manifest_ids.add(pid)

    candidates: List[Dict[str, object]] = []
    skipped_threads = 0
    with threads_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    with threads_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Selecting candidate utterances"):
            if not line.strip():
                continue
            thread = json.loads(line)
            if manifest_ids is not None and thread.get("paper_id") not in manifest_ids:
                skipped_threads += 1
                continue
            utterances = thread.get("utterances", [])
            indexed = list(enumerate(utterances))
            if args.order == "cdate":
                indexed.sort(key=lambda x: (x[1].get("cdate") is None, x[1].get("cdate", 0), x[0]))
            index_map = {u["utterance_id"]: idx for idx, (_, u) in enumerate(indexed)}
            for utt in thread.get("utterances", []):
                text = normalize_text(utt.get("text", ""))
                if not text:
                    continue
                ok, trigger_sentences, trigger_types = is_candidate(text, min_tokens)
                if not ok:
                    continue
                ctx: List[Dict[str, str]] = []
                if utt.get("utterance_id") in index_map:
                    pos = index_map[utt["utterance_id"]]
                    start = max(0, pos - args.context_before)
                    end = min(len(indexed), pos + args.context_after + 1)
                    for i in range(start, end):
                        if i == pos:
                            continue
                        _, other = indexed[i]
                        ctx.append({
                            "utterance_id": other.get("utterance_id"),
                            "role": other.get("role"),
                            "stage": other.get("stage"),
                            "text": normalize_text(other.get("text", "")),
                        })
                candidates.append({
                    "thread_id": thread["thread_id"],
                    "paper_id": thread["paper_id"],
                    "utterance_id": utt["utterance_id"],
                    "role": utt["role"],
                    "stage": utt["stage"],
                    "text": text,
                    "dialogue_context": ctx,
                    "trigger_types": trigger_types,
                    "trigger_sentences": trigger_sentences,
                    "anchors_mentioned": utt.get("anchors_mentioned", []),
                    "surface_cues": utt.get("surface_cues", []),
                    "year": thread["year"],
                })

    write_jsonl(cfg["paths"]["candidate_utterances"], candidates)
    print(f"candidate utterances: {len(candidates)} -> {cfg['paths']['candidate_utterances']}")
    if manifest_ids is not None:
        print(f"threads skipped by manifest filter: {skipped_threads}")


if __name__ == "__main__":
    main()
