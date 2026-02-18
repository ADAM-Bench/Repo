#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from reviewground.utils import load_yaml, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    splits_cfg = cfg["splits"]

    print("[step] load config")

    # load manifest
    print("[step] load manifest")
    manifest_path = Path(paths["manifests"])
    manifest_lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = {}
    for line in tqdm(manifest_lines, desc="Manifest", total=len(manifest_lines)):
        row = json.loads(line)
        manifest[row["paper_id"]] = row
    print(f"[step] manifest rows: {len(manifest)}")

    # load threads for dialog requirement
    print("[step] load threads")
    thread_flags: Dict[str, Dict[str, bool]] = defaultdict(lambda: {"has_review": False, "has_decision": False})
    threads_path = Path(paths["threads"])
    thread_lines = [line for line in threads_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in tqdm(thread_lines, desc="Threads", total=len(thread_lines)):
        thread = json.loads(line)
        for utt in thread.get("utterances", []):
            inv = utt.get("invitation", "")
            if "Official_Review" in inv:
                thread_flags[thread["paper_id"]]["has_review"] = True
            if "Decision" in inv or "Meta_Review" in inv or "Acceptance_Decision" in inv:
                thread_flags[thread["paper_id"]]["has_decision"] = True
    print(f"[step] threads parsed: {len(thread_flags)}")

    # load gold
    print("[step] load gold_clean")
    gold_path = Path(paths["gold_clean"])
    gold_lines = [line for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold_rows = []
    for line in tqdm(gold_lines, desc="Gold", total=len(gold_lines)):
        gold_rows.append(json.loads(line))
    print(f"[step] gold rows: {len(gold_rows)}")

    allowlist = set(cfg.get("license_allowlist") or [])
    filtered: List[Dict] = []
    print("[step] apply filters")
    for row in tqdm(gold_rows, desc="Filter", total=len(gold_rows)):
        paper_id = row["paper_id"]
        meta = manifest.get(paper_id)
        if not meta:
            continue
        if allowlist:
            license_ok = meta.get("license") in allowlist
            if not license_ok:
                continue
        flags = thread_flags.get(paper_id, {})
        if not flags.get("has_review") or not flags.get("has_decision"):
            continue
        filtered.append(row)
    print(f"[step] filtered rows: {len(filtered)}")

    # group by paper
    papers = defaultdict(list)
    print("[step] group by paper")
    for row in tqdm(filtered, desc="Group", total=len(filtered)):
        papers[row["paper_id"]].append(row)
    print(f"[step] papers: {len(papers)}")

    # stratify by year + primary_area
    print("[step] stratify")
    groups: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    for paper_id in tqdm(papers, desc="Stratify", total=len(papers)):
        meta = manifest.get(paper_id, {})
        key = (meta.get("year", 0), meta.get("primary_area") or "unknown")
        groups[key].append(paper_id)
    print(f"[step] strata: {len(groups)}")

    rng = random.Random(splits_cfg.get("random_seed", 13))
    dev_papers = set()
    test_papers = set()

    print("[step] split dev/test")
    for _, paper_list in groups.items():
        rng.shuffle(paper_list)
        n = len(paper_list)
        n_dev = max(1, int(n * splits_cfg["dev_ratio"])) if n >= 10 else max(0, int(n * splits_cfg["dev_ratio"]))
        n_test = max(1, int(n * splits_cfg["test_ratio"])) if n >= 10 else max(0, int(n * splits_cfg["test_ratio"]))
        dev_papers.update(paper_list[:n_dev])
        test_papers.update(paper_list[n_dev : n_dev + n_test])

    train_rows: List[Dict] = []
    dev_rows: List[Dict] = []
    test_rows: List[Dict] = []

    print("[step] assign rows to splits")
    for paper_id, rows in tqdm(papers.items(), desc="Assign", total=len(papers)):
        if paper_id in dev_papers:
            dev_rows.extend(rows)
        elif paper_id in test_papers:
            test_rows.extend(rows)
        else:
            train_rows.extend(rows)

    print("[step] write splits")
    write_jsonl(paths["gold_train"], train_rows)
    write_jsonl(paths["gold_dev"], dev_rows)
    write_jsonl(paths["gold_test"], test_rows)

    print(f"train: {len(train_rows)} -> {paths['gold_train']}")
    print(f"dev:   {len(dev_rows)} -> {paths['gold_dev']}")
    print(f"test:  {len(test_rows)} -> {paths['gold_test']}")


if __name__ == "__main__":
    main()
