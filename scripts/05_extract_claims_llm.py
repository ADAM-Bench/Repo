#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from reviewground.llm_extract import (
    LLMExtractionError,
    extract_claims_from_utterance,
    extract_claims_from_utterance_llm,
)
from reviewground.utils import load_yaml, write_jsonl, simple_tokenize
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--max_claims", type=int, default=None)
    parser.add_argument("--use-llm", action="store_true", help="force LLM extraction if configured")
    parser.add_argument("--llm-model", default=None, help="override LLM model name")
    parser.add_argument("--llm-base", default=None, help="override LLM api base url")
    parser.add_argument("--llm-key-env", default=None, help="override env var for API key")
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--llm-max-output", type=int, default=None)
    parser.add_argument("--llm-max-claims", type=int, default=None, help="max claims per utterance for LLM")
    parser.add_argument(
        "--llm-fuzzy-match",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow fuzzy repair for near-substring spans",
    )
    parser.add_argument("--report", default="data/claimcards/claim_quality_report.json", help="claim quality report path")
    parser.add_argument("--failures", default="data/claimcards/llm_failures.jsonl", help="LLM failure log path")
    parser.add_argument("--processed", default="data/claimcards/processed_utterances.jsonl", help="processed utterance log for resume")
    parser.add_argument(
        "--fail-on-llm-error",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="exit non-zero if any LLM error (default: true when using LLM)",
    )
    parser.add_argument("--retry-max", type=int, default=2, help="max retry rounds for failed utterances")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="resume from processed log")
    parser.add_argument("--debug-dir", default=None, help="optional dir to dump LLM failure inputs/outputs")
    parser.add_argument("--workers", type=int, default=4, help="number of parallel workers for extraction")
    parser.add_argument(
        "--omit-context-utterances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="drop context_utterances field in output claimcards rows to reduce file size",
    )
    parser.add_argument(
        "--inflight-multiplier",
        type=int,
        default=4,
        help="max in-flight tasks = workers * inflight-multiplier",
    )
    parser.add_argument("--flush-every", type=int, default=100, help="flush output files every N written rows")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    models_cfg = load_yaml(args.models)
    max_tokens = cfg["quality_filters"]["max_tokens_per_claim"]
    dedup_threshold = cfg["quality_filters"].get("dedup_sim_threshold", 0.92)
    in_path = Path(cfg["paths"]["candidate_utterances"])

    llm_cfg = models_cfg.get("llm", {}) if isinstance(models_cfg, dict) else {}
    llm_model = args.llm_model or llm_cfg.get("extractor", "heuristic")
    llm_base = args.llm_base or llm_cfg.get("api_base", "")
    llm_key_env = args.llm_key_env or llm_cfg.get("api_key_env", "LLM_API_KEY")
    llm_temperature = args.llm_temperature if args.llm_temperature is not None else llm_cfg.get("temperature", 0.0)
    llm_max_output = args.llm_max_output if args.llm_max_output is not None else llm_cfg.get("max_output_tokens", 1536)
    llm_max_claims = args.llm_max_claims if args.llm_max_claims is not None else llm_cfg.get("max_claims_per_utterance", None)
    if isinstance(llm_max_claims, int) and llm_max_claims <= 0:
        llm_max_claims = None
    llm_fuzzy_match = args.llm_fuzzy_match if args.llm_fuzzy_match is not None else llm_cfg.get("fuzzy_match", True)

    use_llm = args.use_llm or (llm_model and llm_model != "heuristic")
    api_key = None
    if use_llm:
        api_key = (os.getenv(llm_key_env) or "").strip()
        if not api_key or not llm_base or llm_model == "heuristic":
            raise RuntimeError("LLM configured but missing api key/base/model")

    if args.fail_on_llm_error is None:
        args.fail_on_llm_error = bool(use_llm)
    if args.workers <= 0:
        raise ValueError("--workers must be >= 1")
    if args.inflight_multiplier <= 0:
        raise ValueError("--inflight-multiplier must be >= 1")
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be >= 1")

    processed_set = set()
    processed_path = Path(args.processed)
    if args.resume and processed_path.exists():
        with processed_path.open("r", encoding="utf-8") as pf:
            for line in pf:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("status") in {"success", "no_claims"}:
                    processed_set.add((row.get("paper_id"), row.get("utterance_id")))

    claimcards_path = Path(cfg["paths"]["claimcards"])
    claimcards_mode = "a" if args.resume and claimcards_path.exists() else "w"
    processed_mode = "a" if args.resume and processed_path.exists() else "w"
    claimcards_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    claimcards_f = claimcards_path.open(claimcards_mode, encoding="utf-8")
    processed_f = processed_path.open(processed_mode, encoding="utf-8")

    failures = []
    pending_failures = []
    utterance_total = 0
    utterance_with_claims = 0
    utterance_no_claims = 0
    claim_lengths_chars = []
    claim_lengths_tokens = []
    claim_type_counts = {}

    if args.max_claims and args.workers > 1:
        print("warning: --max_claims used with --workers>1; forcing workers=1 for deterministic stop.")
        args.workers = 1

    def run_extract(utt: dict, attempt: int) -> dict:
        try:
            if use_llm:
                claim_objs = extract_claims_from_utterance_llm(
                    utt,
                    max_tokens=max_tokens,
                    dedup_threshold=dedup_threshold,
                    api_base=llm_base,
                    api_key=api_key,
                    model=llm_model,
                    temperature=llm_temperature,
                    max_output_tokens=llm_max_output,
                    max_claims_per_utterance=llm_max_claims,
                    fuzzy_match=bool(llm_fuzzy_match),
                )
            else:
                claim_objs = extract_claims_from_utterance(utt, max_tokens=max_tokens, dedup_threshold=dedup_threshold)
            return {"ok": True, "claims": claim_objs}
        except LLMExtractionError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "raw_response": exc.raw_response,
                "spans": exc.spans,
                "utterance_text": exc.utterance_text or utt.get("text"),
                "attempt": attempt,
            }

    with in_path.open("r", encoding="utf-8") as f:
        total_lines = 0
        unique_papers = set()
        for line in f:
            if not line.strip():
                continue
            total_lines += 1
            try:
                row = json.loads(line)
                pid = row.get("paper_id")
                if pid:
                    unique_papers.add(pid)
            except Exception:
                continue
    print(f"candidate utterances: {total_lines} (unique papers: {len(unique_papers)})")

    processed_skipped = 0
    inflight_limit = max(2, args.workers * args.inflight_multiplier)
    write_rows_since_flush = 0

    def flush_outputs(force: bool = False) -> None:
        nonlocal write_rows_since_flush
        if force or write_rows_since_flush >= args.flush_every:
            processed_f.flush()
            claimcards_f.flush()
            write_rows_since_flush = 0

    def handle_success(utt: dict, claim_objs: list[object]) -> None:
        nonlocal utterance_with_claims, utterance_no_claims, write_rows_since_flush
        for claim in claim_objs:
            row = claim.__dict__ if hasattr(claim, "__dict__") else dict(claim)
            if args.omit_context_utterances and isinstance(row, dict):
                row = dict(row)
                row.pop("context_utterances", None)
            claimcards_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            text = claim.claim_text if hasattr(claim, "claim_text") else claim.get("claim_text", "")
            claim_lengths_chars.append(len(text))
            claim_lengths_tokens.append(len(simple_tokenize(text)))
            ctype = claim.claim_type if hasattr(claim, "claim_type") else claim.get("claim_type", "unknown")
            claim_type_counts[ctype] = claim_type_counts.get(ctype, 0) + 1
            if args.max_claims and len(claim_lengths_chars) >= args.max_claims:
                break
        if claim_objs:
            utterance_with_claims += 1
        else:
            utterance_no_claims += 1
        status = "success" if claim_objs else "no_claims"
        processed_f.write(json.dumps({"paper_id": utt.get("paper_id"), "utterance_id": utt.get("utterance_id"), "status": status}, ensure_ascii=False) + "\n")
        write_rows_since_flush += len(claim_objs) + 1
        flush_outputs()

    def handle_failure(utt: dict, result: dict, fail_list: list) -> None:
        failure = {
            "paper_id": utt.get("paper_id"),
            "utterance_id": utt.get("utterance_id"),
            "error": result.get("error", "LLM extraction failed"),
            "attempt": result.get("attempt", 1),
        }
        if args.debug_dir:
            debug_dir = Path(args.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            pid = (utt.get("paper_id") or "paper").replace("/", "_")
            uid = (utt.get("utterance_id") or "utt").replace("/", "_")
            attempt = result.get("attempt", 1)
            debug_path = debug_dir / f"{pid}_{uid}_attempt{attempt}.json"
            debug_payload = {
                "paper_id": utt.get("paper_id"),
                "utterance_id": utt.get("utterance_id"),
                "error": result.get("error"),
                "utterance_text": result.get("utterance_text") or utt.get("text"),
                "raw_response": result.get("raw_response"),
                "spans": result.get("spans"),
            }
            debug_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            failure["debug_path"] = str(debug_path)
        fail_list.append((utt, failure))

    def update_progress(pbar: tqdm, fail_count: int) -> None:
        pbar.set_postfix({
            "claims": len(claim_lengths_chars),
            "fail": fail_count,
            "skipped": processed_skipped,
            "no_claims": utterance_no_claims,
            "with_claims": utterance_with_claims,
        })

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        with in_path.open("r", encoding="utf-8") as f:
            pbar = tqdm(total=total_lines, desc="Extracting claims")
            for line in f:
                if not line.strip():
                    continue
                utt = json.loads(line)
                key = (utt.get("paper_id"), utt.get("utterance_id"))
                if args.resume and key in processed_set:
                    processed_skipped += 1
                    pbar.update(1)
                    update_progress(pbar, len(pending_failures))
                    continue
                if args.max_claims and len(claim_lengths_chars) >= args.max_claims:
                    break
                utterance_total += 1
                fut = executor.submit(run_extract, utt, 1)
                futures[fut] = utt
                if len(futures) >= inflight_limit:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for dfut in done:
                        result = dfut.result()
                        dut = futures.pop(dfut)
                        if result.get("ok"):
                            handle_success(dut, result["claims"])
                        else:
                            handle_failure(dut, result, pending_failures)
                        pbar.update(1)
                        update_progress(pbar, len(pending_failures))
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for dfut in done:
                    result = dfut.result()
                    dut = futures.pop(dfut)
                    if result.get("ok"):
                        handle_success(dut, result["claims"])
                    else:
                        handle_failure(dut, result, pending_failures)
                    pbar.update(1)
                    update_progress(pbar, len(pending_failures))

    # Retry failed utterances after main pass
    retry_rounds = max(0, args.retry_max)
    attempt = 1
    while pending_failures and attempt <= retry_rounds:
        next_failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            pbar = tqdm(total=len(pending_failures), desc=f"Retry {attempt}/{retry_rounds}")
            for utt, failure in pending_failures:
                if args.max_claims and len(claim_lengths_chars) >= args.max_claims:
                    next_failures.append((utt, failure))
                    pbar.update(1)
                    continue
                fut = executor.submit(run_extract, utt, attempt + 1)
                futures[fut] = utt
                if len(futures) >= inflight_limit:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for dfut in done:
                        result = dfut.result()
                        dut = futures.pop(dfut)
                        if result.get("ok"):
                            handle_success(dut, result["claims"])
                        else:
                            handle_failure(dut, result, next_failures)
                        pbar.update(1)
                    update_progress(pbar, len(next_failures))
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for dfut in done:
                    result = dfut.result()
                    dut = futures.pop(dfut)
                    if result.get("ok"):
                        handle_success(dut, result["claims"])
                    else:
                        handle_failure(dut, result, next_failures)
                    pbar.update(1)
                    update_progress(pbar, len(next_failures))
        pending_failures = next_failures
        attempt += 1

    failures = [f for _, f in pending_failures]
    flush_outputs(force=True)
    claimcards_f.close()
    processed_f.close()

    print(f"claims: {len(claim_lengths_chars)} -> {cfg['paths']['claimcards']}")
    print(f"processed_skipped (resume): {processed_skipped}")
    print(f"processed_this_run: {utterance_total} (with_claims: {utterance_with_claims}, no_claims: {utterance_no_claims})")
    if failures:
        write_jsonl(args.failures, failures)
        print(f"llm failures: {len(failures)} -> {args.failures}")

    report = {
        "utterances_total": utterance_total,
        "utterances_no_claims": utterance_no_claims,
        "utterances_skipped_resume": processed_skipped,
        "utterances_with_claims": utterance_with_claims,
        "claims_total": len(claim_lengths_chars),
        "claims_per_utterance_mean": (len(claim_lengths_chars) / utterance_total) if utterance_total else 0.0,
        "claim_length_chars_mean": statistics.mean(claim_lengths_chars) if claim_lengths_chars else 0.0,
        "claim_length_chars_median": statistics.median(claim_lengths_chars) if claim_lengths_chars else 0.0,
        "claim_length_tokens_mean": statistics.mean(claim_lengths_tokens) if claim_lengths_tokens else 0.0,
        "claim_length_tokens_median": statistics.median(claim_lengths_tokens) if claim_lengths_tokens else 0.0,
        "claim_type_counts": claim_type_counts,
        "llm_model": llm_model if use_llm else "heuristic",
        "llm_fuzzy_match": bool(llm_fuzzy_match) if use_llm else False,
        "llm_max_claims_per_utterance": llm_max_claims,
        "failures_total": len(failures),
        "retry_max": retry_rounds,
        "workers": args.workers,
        "inflight_limit": inflight_limit,
        "flush_every": args.flush_every,
        "omit_context_utterances": args.omit_context_utterances,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {args.report}")

    if failures and args.fail_on_llm_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
