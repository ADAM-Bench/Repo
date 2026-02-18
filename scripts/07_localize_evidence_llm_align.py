#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from reviewground.aligner import align_quotes
from reviewground.llm_locator import locate_evidence_llm
from reviewground.utils import load_yaml


def _meta_path(index_path: Path) -> Path:
    return index_path.with_suffix(".meta.json")


def _load_index(index_path: Path, retrieval_path: Path) -> Optional[Dict[str, int]]:
    meta_path = _meta_path(index_path)
    if not index_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    stat = retrieval_path.stat()
    if meta.get("size") != stat.st_size or meta.get("mtime") != stat.st_mtime:
        return None
    index: Dict[str, int] = {}
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            cid = row.get("claim_id")
            offset = row.get("offset")
            if cid and isinstance(offset, int):
                index[cid] = offset
    return index


def _write_index_meta(index_path: Path, retrieval_path: Path, count: int) -> None:
    meta_path = _meta_path(index_path)
    stat = retrieval_path.stat()
    meta = {"size": stat.st_size, "mtime": stat.st_mtime, "count": count}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_retrieval_at(fp, offset: int) -> Optional[Dict]:
    try:
        fp.seek(offset)
        line = fp.readline()
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base", default=None)
    parser.add_argument("--llm-key-env", default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--llm-max-output", type=int, default=None)
    parser.add_argument("--llm-max-candidates", type=int, default=12)
    parser.add_argument("--llm-max-chars", type=int, default=400)
    parser.add_argument(
        "--llm-use-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="attach candidate images (media_path) to the LLM request when available",
    )
    parser.add_argument("--llm-max-images", type=int, default=4, help="max candidate images per claim")
    parser.add_argument(
        "--llm-max-image-bytes",
        type=int,
        default=800000,
        help="skip image files larger than this many bytes",
    )
    parser.add_argument(
        "--llm-image-detail",
        choices=["low", "high", "auto"],
        default="low",
        help="OpenAI-compatible image detail hint",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only process first N claims")
    parser.add_argument("--sample", type=int, default=None, help="randomly sample N claims")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-every", type=int, default=200, help="log timing stats every N claims")
    parser.add_argument("--retrieval-index", default="data/claimcards/retrieval_index.jsonl")
    parser.add_argument("--no-index", action="store_true", help="disable retrieval index cache")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", help="resume from existing locator outputs")
    resume_group.add_argument("--no-resume", action="store_true", help="ignore existing locator outputs")
    args = parser.parse_args()

    print("[step] load config")
    cfg = load_yaml(args.config)
    models_cfg = load_yaml(args.models)
    paths = cfg["paths"]
    loc_cfg = cfg["localization"]

    llm_cfg = models_cfg.get("llm", {}) if isinstance(models_cfg, dict) else {}
    llm_model = args.llm_model or llm_cfg.get("locator", "heuristic")
    llm_base = args.llm_base or llm_cfg.get("api_base", "")
    llm_key_env = args.llm_key_env or llm_cfg.get("api_key_env", "LLM_API_KEY")
    llm_temperature = args.llm_temperature if args.llm_temperature is not None else llm_cfg.get("temperature", 0.0)
    llm_max_output = args.llm_max_output if args.llm_max_output is not None else llm_cfg.get("max_output_tokens", 1536)
    api_key = (os.getenv(llm_key_env) or "").strip()
    if llm_model == "heuristic" or not llm_base or not api_key:
        raise RuntimeError("LLM locator configured but missing api base/key/model")
    print(f"[step] llm locator: model={llm_model}, base={llm_base}")
    print(
        f"[step] llm multimodal: use_images={args.llm_use_images}, "
        f"max_images={args.llm_max_images}, max_image_bytes={args.llm_max_image_bytes}, detail={args.llm_image_detail}"
    )

    print("[step] load claimcards")
    claim_lines = [line for line in Path(paths["claimcards"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    claim_rows = [json.loads(line) for line in tqdm(claim_lines, desc="Parsing claimcards", total=len(claim_lines))]
    if args.sample:
        rng = random.Random(args.seed)
        claim_rows = rng.sample(claim_rows, min(args.sample, len(claim_rows)))
        print(f"[test] random sample: {len(claim_rows)} claims")
    elif args.limit:
        claim_rows = claim_rows[: args.limit]
        print(f"[test] limit: {len(claim_rows)} claims")
    print("[step] prepare retrieval stream")
    retrieval_path = Path(paths["retrieval"])
    if not retrieval_path.exists():
        raise FileNotFoundError(retrieval_path)

    print("[step] load evidence parquet")
    eobj_df = pd.read_parquet(paths["evidence_parquet"])
    rows = eobj_df.to_dict(orient="records")
    eobj_map: Dict[str, Dict] = {}
    for row in tqdm(rows, desc="Indexing eobjs", total=len(rows)):
        eobj_map[row["eobj_id"]] = row

    claim_map = {c["claim_id"]: c for c in claim_rows}
    output_path = Path(paths["locator"])
    resume = args.resume or not args.no_resume
    existing_ids = set()
    if resume and output_path.exists():
        print(f"[step] resume enabled: reading {output_path}")
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                cid = row.get("claim_id")
                if cid:
                    existing_ids.add(cid)
    pending_ids = {cid for cid in claim_map.keys() if cid not in existing_ids}
    target_total = len(pending_ids)
    print(f"[step] localizing claims: {target_total} (skipped {len(existing_ids)})")
    if target_total == 0:
        print("[step] nothing to do")
        return
    start_time = time.time()
    last_log_time = start_time
    last_log_idx = 0
    failures = 0
    processed = 0
    scan_total = retrieval_path.stat().st_size
    def process_claim(claim: Dict, candidates: List[Dict]) -> Tuple[Dict, bool]:
        try:
            locator = locate_evidence_llm(
                claim,
                candidates,
                api_base=llm_base,
                api_key=api_key,
                model=llm_model,
                temperature=llm_temperature,
                max_output_tokens=llm_max_output,
                max_candidates=args.llm_max_candidates,
                max_chars=args.llm_max_chars,
                use_images=args.llm_use_images,
                max_images=args.llm_max_images,
                max_image_bytes=args.llm_max_image_bytes,
                image_detail=args.llm_image_detail,
            )
        except Exception:
            return {"claim_id": claim.get("claim_id"), "label": "NOT_FOUND", "evidence_sets": []}, False

        aligned_sets: List[List[Dict]] = []
        for ev_set in locator.get("evidence_sets", []):
            aligned_set: List[Dict] = []
            for ev in ev_set:
                eobj = eobj_map.get(ev["eobj_id"])
                if not eobj:
                    continue
                quotes = ev.get("quotes", [])[: loc_cfg["max_quotes_per_evidence"]]
                clipped = []
                for q in quotes:
                    if not isinstance(q, str):
                        continue
                    q = q.strip()
                    if not q:
                        continue
                    if len(q) > loc_cfg["max_quote_chars"]:
                        q = q[: loc_cfg["max_quote_chars"]].rstrip()
                    clipped.append(q)
                spans = align_quotes(eobj, clipped)
                aligned_set.extend(spans)
            if aligned_set:
                aligned_sets.append(aligned_set)

        return {
            "claim_id": claim["claim_id"],
            "label": locator.get("label", "NOT_FOUND"),
            "evidence_sets": aligned_sets,
        }, True

    def drain_futures(done, out_f, pbar):
        nonlocal processed, failures, last_log_idx, last_log_time
        for fut in done:
            try:
                out_row, ok = fut.result()
            except Exception:
                ok = False
                out_row = {"claim_id": None, "label": "NOT_FOUND", "evidence_sets": []}
            if not ok:
                failures += 1
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1
            pbar.update(1)
            if args.log_every and processed % args.log_every == 0:
                now = time.time()
                batch_elapsed = now - last_log_time
                total_elapsed = now - start_time
                batch_rate = (processed - last_log_idx) / batch_elapsed if batch_elapsed > 0 else 0.0
                total_rate = processed / total_elapsed if total_elapsed > 0 else 0.0
                batch_fail_rate = failures / processed if processed > 0 else 0.0
                print(f"[stat] {processed}/{target_total} | batch_rate={batch_rate:.2f}/s | total_rate={total_rate:.2f}/s | fail_rate={batch_fail_rate:.2%}")
                last_log_time = now
                last_log_idx = processed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_mode = "a" if resume and output_path.exists() else "w"
    index_path = Path(args.retrieval_index)
    use_index = not args.no_index
    retrieval_index = _load_index(index_path, retrieval_path) if use_index else None
    max_inflight = max(1, args.workers * 4)

    with output_path.open(out_mode, encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending_futures = set()
            if retrieval_index:
                print(f"[step] using retrieval index: {index_path}")
                with retrieval_path.open("rb") as rf:
                    pbar = tqdm(total=target_total, desc="LLM localize")
                    for claim_id in sorted(pending_ids):
                        offset = retrieval_index.get(claim_id)
                        if offset is None:
                            continue
                        retrieval = _read_retrieval_at(rf, offset)
                        if not retrieval:
                            continue
                        claim = claim_map.get(claim_id)
                        if not claim:
                            continue
                        candidates = retrieval.get("candidates", [])
                        pending_futures.add(executor.submit(process_claim, claim, candidates))
                        if len(pending_futures) >= max_inflight:
                            done, pending_futures = concurrent.futures.wait(
                                pending_futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            drain_futures(done, out_f, pbar)
                    if pending_futures:
                        done, _ = concurrent.futures.wait(pending_futures)
                        drain_futures(done, out_f, pbar)
                    pbar.close()
            else:
                if use_index:
                    print(f"[step] build retrieval index cache: {index_path}")
                scan_total = retrieval_path.stat().st_size
                with retrieval_path.open("rb") as rf:
                    scan_pbar = tqdm(total=scan_total, desc="Scanning retrieval", unit="B", unit_scale=True)
                    pbar = tqdm(total=target_total, desc="LLM localize")
                    index_writer = index_path.open("w", encoding="utf-8") if use_index else None
                    found = 0
                    count_index = 0
                    while True:
                        offset = rf.tell()
                        line = rf.readline()
                        if not line:
                            break
                        scan_pbar.update(len(line))
                        if not line.strip():
                            continue
                        try:
                            retrieval = json.loads(line)
                        except Exception:
                            continue
                        claim_id = retrieval.get("claim_id")
                        if claim_id and use_index and index_writer:
                            index_writer.write(json.dumps({"claim_id": claim_id, "offset": offset}, ensure_ascii=False) + "\n")
                            count_index += 1
                        if claim_id not in pending_ids:
                            continue
                        pending_ids.discard(claim_id)
                        found += 1
                        claim = claim_map.get(claim_id)
                        if not claim:
                            continue
                        candidates = retrieval.get("candidates", [])
                        pending_futures.add(executor.submit(process_claim, claim, candidates))
                        if len(pending_futures) >= max_inflight:
                            done, pending_futures = concurrent.futures.wait(
                                pending_futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            drain_futures(done, out_f, pbar)
                        if found >= target_total:
                            break
                    if pending_futures:
                        done, _ = concurrent.futures.wait(pending_futures)
                        drain_futures(done, out_f, pbar)
                    if index_writer:
                        index_writer.close()
                        _write_index_meta(index_path, retrieval_path, count_index)
                    pbar.close()
                    scan_pbar.close()

    print(f"locator outputs (appended): {processed} -> {paths['locator']}")


if __name__ == "__main__":
    main()
