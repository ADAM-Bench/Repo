#!/usr/bin/env python
from __future__ import annotations

import argparse

from reviewground.eobj_builder import build_all_eobjs
from reviewground.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]

    total, missing = build_all_eobjs(
        manifest_path=paths["manifests"],
        mineru_dir=paths["mineru_dir"],
        parquet_path=paths["evidence_parquet"],
        anchor_index_path=paths["anchor_index"],
    )

    print(f"processed papers: {total}, missing mineru outputs: {missing}")
    print(f"eobjs -> {paths['evidence_parquet']}")
    print(f"anchors -> {paths['anchor_index']}")


if __name__ == "__main__":
    main()
