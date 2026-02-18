#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from reviewground.openreview_ingest import build_thread, thread_to_dict
from reviewground.utils import load_yaml, write_jsonl


def iter_forums(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    raw_path = Path(paths["raw_openreview"])
    min_tokens = cfg["quality_filters"]["min_tokens_per_utterance"]

    threads = []
    for forum in tqdm(iter_forums(raw_path), desc="Building threads"):
        year = int(forum.get("year"))
        if year not in set(cfg["years"]):
            continue
        thread = build_thread(forum, cfg["venue"], min_tokens=min_tokens)
        if not thread:
            continue
        threads.append(thread_to_dict(thread))

    write_jsonl(paths["threads"], threads)
    print(f"threads: {len(threads)} -> {paths['threads']}")


if __name__ == "__main__":
    main()
