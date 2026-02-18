#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reviewground.openreview_ingest import build_manifest_entry
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
    parser.add_argument("--input", default="openreview.jsonl")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    raw_path = Path(paths["raw_openreview"])
    input_path = Path(args.input)

    if input_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if input_path.resolve() != raw_path.resolve():
            shutil.copyfile(input_path, raw_path)
    else:
        raise FileNotFoundError(input_path)

    years = set(cfg["years"])
    include_2026 = bool(cfg.get("include_2026_initial_only"))

    manifest_rows = []
    year_counts = {}
    for forum in iter_forums(raw_path):
        year = int(forum.get("year"))
        year_counts[year] = year_counts.get(year, 0) + 1
        if year not in years:
            if year == 2026 and include_2026:
                pass
            else:
                continue
        manifest = build_manifest_entry(forum, cfg["venue"])
        if not manifest:
            continue
        manifest_rows.append(manifest.__dict__)

    write_jsonl(paths["manifests"], manifest_rows)

    years_present = sorted(year_counts)
    print(f"raw forums: {sum(year_counts.values())}")
    print(f"years present: {years_present}")
    if 2026 in year_counts and 2026 not in years:
        print("note: 2026 present in raw file but excluded by config years")

    print(f"manifest rows: {len(manifest_rows)} -> {paths['manifests']}")


if __name__ == "__main__":
    main()
