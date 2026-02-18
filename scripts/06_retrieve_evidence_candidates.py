#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from reviewground.http_client import urlopen_with_ssl
from reviewground.retrieval import BM25Index, TfidfIndex, rrf_fuse, topk
from reviewground.utils import load_yaml, normalize_text, write_jsonl
from tqdm import tqdm


def load_anchor_index(path: str) -> Dict[str, Dict[str, str]]:
    anchors: Dict[str, Dict[str, str]] = {}
    if not Path(path).exists():
        return anchors
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in tqdm(lines, desc="Parsing anchor index", total=len(lines)):
        if not line.strip():
            continue
        row = json.loads(line)
        anchors.setdefault(row["paper_id"], {})[row["anchor"]] = row["eobj_id"]
    return anchors


def _call_embeddings(api_base: str, api_key: str, model: str, texts: List[str]) -> List[List[float]]:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/embeddings"
    else:
        url = f"{base}/v1/embeddings"
    payload = {"model": model, "input": texts}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DMXAPI/1.0.0",
        },
    )
    with urlopen_with_ssl(req, timeout=300) as resp:
        raw = resp.read().decode("utf-8")
    obj = json.loads(raw)
    return [d["embedding"] for d in obj.get("data", [])]


METRIC_RE = re.compile(r"\b(?:accuracy|acc|f1|precision|recall|bleu|rouge|map|auc|perplexity|rmse|mae|mse)\b", re.IGNORECASE)

def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base", default=None)
    parser.add_argument("--embedding-key-env", default=None)
    parser.add_argument("--embedding-cache", default="data/embeddings")
    parser.add_argument("--embedding-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--processed", default="data/claimcards/retrieval_processed.jsonl")
    parser.add_argument("--failures", default="data/claimcards/retrieval_failures.jsonl")
    parser.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dense-gate",
        choices=["none", "signal", "bm25", "signal_or_bm25"],
        default="signal_or_bm25",
        help="skip embedding if gate condition is satisfied",
    )
    parser.add_argument("--bm25-threshold", type=float, default=1.2)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    models_cfg = load_yaml(args.models)
    paths = cfg["paths"]
    retrieval_cfg = cfg["retrieval"]
    print("[step] load eobjs parquet")

    eobj_df = pd.read_parquet(paths["evidence_parquet"])
    eobjs_by_paper: Dict[str, List[Dict]] = {}
    rows = eobj_df.to_dict(orient="records")
    for row in tqdm(rows, desc="Indexing eobjs", total=len(rows)):
        eobjs_by_paper.setdefault(row["paper_id"], []).append(row)
    print(f"[step] eobjs loaded: {len(eobj_df)} rows, {len(eobjs_by_paper)} papers")

    print("[step] load anchor index")
    anchor_index = load_anchor_index(paths["anchor_index"])
    print(f"[step] anchor index loaded: {sum(len(v) for v in anchor_index.values())} anchors")

    embed_cfg = models_cfg.get("embedding", {}) if isinstance(models_cfg, dict) else {}
    embed_model = args.embedding_model or embed_cfg.get("model", "text-embedding-3-small")
    embed_base = args.embedding_base or embed_cfg.get("api_base", "")
    embed_key_env = args.embedding_key_env or embed_cfg.get("api_key_env", "EMBEDDING_API_KEY")
    embed_batch = args.embedding_batch or embed_cfg.get("batch_size", 64)
    use_embeddings = bool(embed_base and embed_model and embed_model != "tfidf")
    api_key = (os.getenv(embed_key_env) or "").strip() if use_embeddings else ""
    if use_embeddings and not api_key:
        raise RuntimeError("Embedding configured but missing API key")
    if use_embeddings:
        print(f"[step] embedding enabled: model={embed_model}")
    else:
        print("[step] embedding disabled, using tfidf only")

    cache_dir = Path(args.embedding_cache)
    if use_embeddings:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print("[step] load claimcards")
    claim_lines = [line for line in Path(paths["claimcards"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    claim_rows = [json.loads(line) for line in tqdm(claim_lines, desc="Parsing claimcards", total=len(claim_lines))]
    print(f"[step] claims loaded: {len(claim_rows)}")
    outputs = []

    processed_set = set()
    processed_path = Path(args.processed)
    if args.resume and processed_path.exists():
        print(f"[step] resume enabled, reading processed: {processed_path}")
        with processed_path.open("r", encoding="utf-8") as pf:
            for line in pf:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("status") in {"success", "skipped_no_eobj"} and row.get("claim_id"):
                    processed_set.add(row["claim_id"])

    pending_claims = [c for c in claim_rows if not (args.resume and c.get("claim_id") in processed_set)]
    print(f"[step] pending claims: {len(pending_claims)} (skipped {len(claim_rows)-len(pending_claims)})")

    # build per-paper indexes
    index_cache: Dict[str, Dict[str, object]] = {}
    embed_cache: Dict[str, np.ndarray] = {}
    unique_papers = sorted({c.get("paper_id") for c in pending_claims if c.get("paper_id")})
    print(f"[step] build per-paper indexes: {len(unique_papers)} papers")
    for paper_id in tqdm(unique_papers, desc="Building indexes", total=len(unique_papers)):
        eobjs = eobjs_by_paper.get(paper_id, [])
        if not eobjs:
            continue
        texts = [normalize_text(e["text_concat"]) for e in eobjs]
        index_cache[paper_id] = {
            "bm25": BM25Index(texts),
            "tfidf": TfidfIndex(texts),
            "texts": texts,
        }

    retrieval_path = Path(paths["retrieval"])
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = Path(args.failures)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    retrieval_mode = "a" if args.resume and retrieval_path.exists() else "w"
    processed_mode = "a" if args.resume and processed_path.exists() else "w"
    failures_mode = "a" if args.resume and failures_path.exists() else "w"
    retrieval_f = retrieval_path.open(retrieval_mode, encoding="utf-8")
    processed_f = processed_path.open(processed_mode, encoding="utf-8")
    failures_f = failures_path.open(failures_mode, encoding="utf-8")

    def process_claims_for_paper(paper_id: str, claims: List[Dict[str, object]]) -> Dict[str, object]:
        if paper_id not in eobjs_by_paper:
            return {"results": [{"status": "skipped_no_eobj", "claim": c} for c in claims], "dense_used": 0, "dense_skipped": len(claims)}
        eobjs = eobjs_by_paper.get(paper_id, [])
        if not eobjs:
            return {"results": [{"status": "skipped_no_eobj", "claim": c} for c in claims], "dense_used": 0, "dense_skipped": len(claims)}

        idx = index_cache[paper_id]
        bm25_scores_list = [idx["bm25"].score(c["claim_text"]) for c in claims]
        tfidf_scores_list = [idx["tfidf"].score(c["claim_text"]) for c in claims]

        dense_needed_idx: List[int] = []
        dense_used = 0
        dense_skipped = 0
        for i, claim in enumerate(claims):
            anchors = claim.get("anchors_mentioned") or []
            numbers = claim.get("numbers") or []
            text = claim.get("claim_text") or ""
            has_signal = bool(anchors) or bool(numbers) or bool(METRIC_RE.search(text))
            top_bm25 = max(bm25_scores_list[i]) if bm25_scores_list[i] else 0.0
            gate_hit = False
            if args.dense_gate == "signal":
                gate_hit = has_signal
            elif args.dense_gate == "bm25":
                gate_hit = top_bm25 >= args.bm25_threshold
            elif args.dense_gate == "signal_or_bm25":
                gate_hit = has_signal or top_bm25 >= args.bm25_threshold
            if args.dense_gate == "none":
                gate_hit = False
            if gate_hit:
                dense_skipped += 1
            else:
                dense_needed_idx.append(i)

        dense_scores_list = {}
        if use_embeddings and dense_needed_idx:
            cache_path = cache_dir / f"{paper_id}.npz"
            eobj_ids = [e["eobj_id"] for e in eobjs]
            emb = None
            if cache_path.exists():
                data = np.load(cache_path, allow_pickle=True)
                cached_ids = list(data["eobj_ids"])
                if cached_ids == eobj_ids:
                    emb = data["embeddings"].astype(np.float32)
            if emb is None:
                texts = [normalize_text(e["text_concat"]) for e in eobjs]
                vectors: List[List[float]] = []
                for i in range(0, len(texts), embed_batch):
                    batch = texts[i : i + embed_batch]
                    vectors.extend(_call_embeddings(embed_base, api_key, embed_model, batch))
                emb = np.asarray(vectors, dtype=np.float32)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / np.maximum(norms, 1e-12)
                np.savez_compressed(cache_path, eobj_ids=np.array(eobj_ids), embeddings=emb)
            embed_cache[paper_id] = emb

            claim_texts = [claims[i]["claim_text"] for i in dense_needed_idx]
            vectors: List[List[float]] = []
            for i in range(0, len(claim_texts), embed_batch):
                batch = claim_texts[i : i + embed_batch]
                vectors.extend(_call_embeddings(embed_base, api_key, embed_model, batch))
            mat = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.maximum(norms, 1e-12)
            scores = (emb @ mat.T).T.tolist()
            for local_idx, claim_idx in enumerate(dense_needed_idx):
                dense_scores_list[claim_idx] = scores[local_idx]
                dense_used += 1

        results: List[Dict[str, object]] = []
        anchor_map = anchor_index.get(paper_id, {})
        for idx_claim, claim in enumerate(claims):
            bm25_scores = bm25_scores_list[idx_claim]
            tfidf_scores = tfidf_scores_list[idx_claim]
            sparse_rank = topk(bm25_scores, retrieval_cfg["topk_sparse"])
            dense_scores = None
            if use_embeddings and idx_claim in dense_scores_list:
                dense_scores = dense_scores_list[idx_claim]
                dense_rank = topk(dense_scores, retrieval_cfg["topk_dense"])
            else:
                dense_rank = topk(tfidf_scores, retrieval_cfg["topk_dense"])
            fused_rank = rrf_fuse({"sparse": sparse_rank, "dense": dense_rank})

            anchors = claim.get("anchors_mentioned", [])
            boosted = []
            for anchor in anchors:
                if anchor in anchor_map:
                    eobj_id = anchor_map[anchor]
                    for i, e in enumerate(eobjs):
                        if e["eobj_id"] == eobj_id:
                            boosted.append(i)
            fused_rank = boosted + [i for i in fused_rank if i not in boosted]

            topk_fused = fused_rank[: retrieval_cfg["topk_fused"]]
            candidates = []
            for rank, idx_id in enumerate(topk_fused):
                eobj = eobjs[idx_id]
                section_path = _to_builtin(eobj.get("section_path")) or []
                page_no = _to_builtin(eobj.get("page_no"))
                candidates.append({
                    "eobj_id": eobj["eobj_id"],
                    "rank": rank,
                    "score_sparse": _to_builtin(bm25_scores[idx_id]),
                    "score_dense": _to_builtin(tfidf_scores[idx_id]) if not use_embeddings else (_to_builtin(dense_scores[idx_id]) if dense_scores else None),
                    "type": eobj["type"],
                    "section_path": section_path,
                    "page_no": page_no,
                    "text": eobj["text_concat"],
                    "media_path": eobj.get("media_path"),
                })
            results.append({
                "status": "success",
                "row": {"claim_id": claim["claim_id"], "paper_id": paper_id, "candidates": candidates},
            })
        return {"results": results, "dense_used": dense_used, "dense_skipped": dense_skipped}

    inflight_limit = max(2, args.workers * 2)
    processed_count = 0
    skipped_count = 0
    failures_count = 0
    dense_used_total = 0
    dense_skipped_total = 0

    pending_by_paper: Dict[str, List[Dict[str, object]]] = {}
    for claim in claim_rows:
        claim_id = claim.get("claim_id")
        if args.resume and claim_id in processed_set:
            processed_count += 1
            continue
        pid = claim.get("paper_id")
        if not pid:
            skipped_count += 1
            continue
        pending_by_paper.setdefault(pid, []).append(claim)

    print(f"[step] parallel retrieval workers={args.workers}, papers={len(pending_by_paper)}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        pbar = tqdm(total=len(claim_rows), desc="Retrieving evidence")
        if processed_count or skipped_count:
            pbar.update(processed_count + skipped_count)
        for paper_id, claims in pending_by_paper.items():
            fut = executor.submit(process_claims_for_paper, paper_id, claims)
            futures[fut] = (paper_id, claims)
            if len(futures) >= inflight_limit:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for dfut in done:
                    payload = dfut.result()
                    results = payload["results"]
                    dense_used_total += payload.get("dense_used", 0)
                    dense_skipped_total += payload.get("dense_skipped", 0)
                    paper_id, claims = futures.pop(dfut)
                    for claim_obj, result in zip(claims, results):
                        if result.get("status") == "success":
                            retrieval_f.write(json.dumps(result["row"], ensure_ascii=False) + "\n")
                            processed_f.write(json.dumps({"claim_id": claim_obj.get("claim_id"), "status": "success"}, ensure_ascii=False) + "\n")
                            processed_count += 1
                        elif result.get("status") == "skipped_no_eobj":
                            processed_f.write(json.dumps({"claim_id": claim_obj.get("claim_id"), "status": "skipped_no_eobj"}, ensure_ascii=False) + "\n")
                            skipped_count += 1
                        else:
                            failures_f.write(json.dumps({"claim_id": claim_obj.get("claim_id"), "error": "unknown"}, ensure_ascii=False) + "\n")
                            failures_count += 1
                        retrieval_f.flush()
                        processed_f.flush()
                        failures_f.flush()
                        pbar.update(1)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for dfut in done:
                payload = dfut.result()
                results = payload["results"]
                dense_used_total += payload.get("dense_used", 0)
                dense_skipped_total += payload.get("dense_skipped", 0)
                paper_id, claims = futures.pop(dfut)
                for claim_obj, result in zip(claims, results):
                    if result.get("status") == "success":
                        retrieval_f.write(json.dumps(result["row"], ensure_ascii=False) + "\n")
                        processed_f.write(json.dumps({"claim_id": claim_obj.get("claim_id"), "status": "success"}, ensure_ascii=False) + "\n")
                        processed_count += 1
                    elif result.get("status") == "skipped_no_eobj":
                        processed_f.write(json.dumps({"claim_id": claim_obj.get("claim_id"), "status": "skipped_no_eobj"}, ensure_ascii=False) + "\n")
                        skipped_count += 1
                    else:
                        failures_f.write(json.dumps({"claim_id": claim_obj.get("claim_id"), "error": "unknown"}, ensure_ascii=False) + "\n")
                        failures_count += 1
                    retrieval_f.flush()
                    processed_f.flush()
                    failures_f.flush()
                    pbar.update(1)
        pbar.close()

    retrieval_f.close()
    processed_f.close()
    failures_f.close()
    print(f"retrieval results -> {paths['retrieval']}")
    print(f"processed: {processed_count}, skipped_no_eobj: {skipped_count}, failures: {failures_count}")
    if use_embeddings:
        print(f"dense_used: {dense_used_total}, dense_skipped: {dense_skipped_total}, gate={args.dense_gate}, bm25_threshold={args.bm25_threshold}")


if __name__ == "__main__":
    main()
