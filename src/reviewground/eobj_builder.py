from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .mineru_parse import load_mineru_outputs
from .utils import ensure_parent, write_jsonl
from tqdm import tqdm

def build_all_eobjs(manifest_path: str, mineru_dir: str, parquet_path: str, anchor_index_path: str) -> Tuple[int, int]:
    eobj_rows: List[Dict] = []
    anchor_rows: List[Dict] = []
    missing = 0
    total = 0

    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building evidence objects"):
            if not line.strip():
                continue
            record = json.loads(line)
            paper_id = record["paper_id"]
            total += 1
            try:
                eobjs, anchor_index = load_mineru_outputs(mineru_dir, paper_id)
            except FileNotFoundError:
                missing += 1
                continue
            for eobj in eobjs:
                eobj_rows.append(eobj.__dict__)
            for anchor, eobj_id in anchor_index.items():
                anchor_rows.append({"paper_id": paper_id, "anchor": anchor, "eobj_id": eobj_id})

    if eobj_rows:
        ensure_parent(parquet_path)
        df = pd.DataFrame(eobj_rows)
        df.to_parquet(parquet_path, index=False)

    if anchor_rows:
        write_jsonl(anchor_index_path, anchor_rows)

    return total, missing
