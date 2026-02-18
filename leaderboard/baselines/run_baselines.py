#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from reviewground.llm_locator import locate_evidence_llm
from reviewground.retrieval import BM25Index, TfidfIndex, rrf_fuse, topk
from reviewground.utils import load_yaml, normalize_text


def load_claims(path: str) -> List[Dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def canonical_label(label: object) -> str:
    text = str(label or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"NOTFOUND", "NOT_FOUND"}:
        return "NOT_FOUND"
    if text in {"SUPPORTED", "CONTRADICTED", "UNDECIDABLE"}:
        return text
    return "NOT_FOUND"


def extract_gold_evidence_sets(row: Dict) -> List[List[str]]:
    sets: List[List[str]] = []
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
        if ids:
            sets.append(ids)
    return sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build.yaml")
    parser.add_argument("--split", default="data/gold/gold_dev.jsonl")
    parser.add_argument("--mode", choices=["r-only", "llm-direct", "hybrid", "topk-ub"], default="hybrid")
    parser.add_argument("--output", default="leaderboard/submissions/preds.jsonl")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--model", default=None, help="override LLM model for llm-direct/hybrid")
    parser.add_argument("--api-base", default=None, help="override LLM API base")
    parser.add_argument("--api-key-env", default=None, help="override API key env var")
    parser.add_argument(
        "--topk-input",
        type=int,
        default=12,
        help="Top-K retrieval candidates treated as LLM input for topk-ub baseline",
    )
    parser.add_argument("--sample", type=int, default=0, help="randomly sample N claims for quick test")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--progress", action="store_true", help="print progress lines (progress: N/total)")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    models_cfg = load_yaml(args.models)
    retrieval_cfg = cfg["retrieval"]
    llm_cfg = models_cfg.get("llm", {})

    claims = load_claims(args.split)
    if args.sample and args.sample > 0 and args.sample < len(claims):
        rng = random.Random(args.seed)
        claims = rng.sample(claims, args.sample)

    api_base = args.api_base or llm_cfg.get("api_base", "")
    api_key_env = args.api_key_env or llm_cfg.get("api_key_env", "LLM_API_KEY")
    api_key = None
    if api_key_env:
        api_key = os.getenv(api_key_env)
    model_name = args.model or llm_cfg.get("locator", "")
    temperature = llm_cfg.get("temperature", 0.0)
    max_output_tokens = llm_cfg.get("leaderboard_max_output_tokens", llm_cfg.get("max_output_tokens", 256))
    by_model = llm_cfg.get("leaderboard_max_output_tokens_by_model", {})
    if isinstance(by_model, dict):
        key = str(model_name or "")
        override = by_model.get(key)
        if override is None:
            override = by_model.get(key.lower())
        if override is not None:
            try:
                max_output_tokens = int(override)
            except Exception:
                pass
    disable_thinking = bool(llm_cfg.get("disable_thinking", True))
    thinking_type = llm_cfg.get("thinking_type")
    thinking_effort = llm_cfg.get("thinking_effort")
    thinking_type = str(thinking_type).strip() if thinking_type is not None else None
    thinking_effort = str(thinking_effort).strip() if thinking_effort is not None else None
    if not thinking_type:
        thinking_type = None
    if not thinking_effort:
        thinking_effort = None
    if disable_thinking:
        thinking_type = None
        thinking_effort = None

    if args.mode in {"hybrid", "llm-direct"}:
        if not api_key:
            raise RuntimeError(f"missing API key env {api_key_env} for {args.mode} baseline")
        if not api_base or not model_name:
            raise RuntimeError("missing llm.api_base or llm.locator in models config")

    eobjs_by_paper: Dict[str, List[Dict]] = {}
    if args.mode != "llm-direct":
        eobj_df = pd.read_parquet(cfg["paths"]["evidence_parquet"])
        for row in eobj_df.to_dict(orient="records"):
            eobjs_by_paper.setdefault(row["paper_id"], []).append(row)

    outputs = []
    index_cache: Dict[str, Dict[str, object]] = {}

    total = len(claims)
    if args.progress:
        print(f"progress: 0/{total}", flush=True)

    for idx, claim in enumerate(claims, 1):
        if args.mode == "llm-direct":
            locator = locate_evidence_llm(
                claim=claim,
                candidates=[],
                api_base=api_base,
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_candidates=0,
                max_chars=0,
                ids_only=True,
                disable_thinking=disable_thinking,
                thinking_type=thinking_type,
                effort=thinking_effort,
            )
            outputs.append({
                "claim_id": claim["claim_id"],
                "label": locator.get("label", "NOT_FOUND"),
                "evidence_sets": [],
            })
            if args.progress and (idx % args.log_every == 0 or idx == total):
                print(f"progress: {idx}/{total}", flush=True)
            continue

        paper_id = claim["paper_id"]
        eobjs = eobjs_by_paper.get(paper_id, [])
        if not eobjs:
            if args.mode == "topk-ub":
                outputs.append({
                    "claim_id": claim["claim_id"],
                    "label": canonical_label(claim.get("label")),
                    "evidence_sets": [],
                })
            else:
                outputs.append({"claim_id": claim["claim_id"], "label": "NOT_FOUND", "evidence_sets": []})
            if args.progress and (idx % args.log_every == 0 or idx == total):
                print(f"progress: {idx}/{total}", flush=True)
            continue
        if paper_id not in index_cache:
            texts = [normalize_text(e["text_concat"]) for e in eobjs]
            index_cache[paper_id] = {
                "bm25": BM25Index(texts),
                "tfidf": TfidfIndex(texts),
                "texts": texts,
            }
        paper_idx = index_cache[paper_id]
        bm25_scores = paper_idx["bm25"].score(claim["claim_text"])
        tfidf_scores = paper_idx["tfidf"].score(claim["claim_text"])
        sparse_rank = topk(bm25_scores, retrieval_cfg["topk_sparse"])
        dense_rank = topk(tfidf_scores, retrieval_cfg["topk_dense"])
        fused_rank = rrf_fuse({"sparse": sparse_rank, "dense": dense_rank})
        if args.mode == "topk-ub":
            label = canonical_label(claim.get("label"))
            if label in {"NOT_FOUND", "UNDECIDABLE"}:
                pred_sets: List[List[Dict]] = []
            else:
                k = max(int(args.topk_input), 0)
                try:
                    k = min(k, int(retrieval_cfg.get("topk_fused", k)))
                except Exception:
                    pass
                top_ids = fused_rank[:k] if k else []
                candidate_ids = []
                for cid in top_ids:
                    eobj_id = eobjs[cid].get("eobj_id")
                    if eobj_id:
                        candidate_ids.append(str(eobj_id))
                candidate_id_set = set(candidate_ids)
                pred_sets = []
                for gold_set in extract_gold_evidence_sets(claim):
                    chosen = [eid for eid in gold_set if eid in candidate_id_set]
                    if chosen:
                        pred_sets.append([{"eobj_id": eid, "spans": []} for eid in chosen])
            outputs.append({
                "claim_id": claim["claim_id"],
                "label": label,
                "evidence_sets": pred_sets,
            })
            if args.progress and (idx % args.log_every == 0 or idx == total):
                print(f"progress: {idx}/{total}", flush=True)
            continue

        top_id = fused_rank[0]
        top_eobj = eobjs[top_id]

        if args.mode == "r-only":
            outputs.append({
                "claim_id": claim["claim_id"],
                "label": "SUPPORTED",
                "evidence_sets": [[{"eobj_id": top_eobj["eobj_id"], "spans": []}]],
            })
            if args.progress and (idx % args.log_every == 0 or idx == total):
                print(f"progress: {idx}/{total}", flush=True)
            continue

        locator = locate_evidence_llm(
            claim=claim,
            candidates=[{"eobj_id": top_eobj["eobj_id"], "text": top_eobj["text_concat"], "type": top_eobj.get("type"), "section_path": top_eobj.get("section_path"), "page_no": top_eobj.get("page_no")}],
            api_base=api_base,
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_candidates=1,
            max_chars=140,
            ids_only=True,
            disable_thinking=disable_thinking,
            thinking_type=thinking_type,
            effort=thinking_effort,
        )
        pred_sets = []
        for ev_set in locator.get("evidence_sets", []):
            cleaned = []
            for ev in ev_set:
                if isinstance(ev, dict) and ev.get("eobj_id"):
                    cleaned.append({"eobj_id": ev.get("eobj_id"), "spans": []})
            if cleaned:
                pred_sets.append(cleaned)
        outputs.append({
            "claim_id": claim["claim_id"],
            "label": locator.get("label", "NOT_FOUND"),
            "evidence_sets": pred_sets,
        })

        if args.progress and (idx % args.log_every == 0 or idx == total):
            print(f"progress: {idx}/{total}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    print(f"predictions: {len(outputs)} -> {args.output}")


if __name__ == "__main__":
    main()
