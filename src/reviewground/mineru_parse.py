from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schemas import EvidenceObject
from .utils import normalize_text


ANCHOR_RE = re.compile(r"\b(?:Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?|Algorithm)\s*\(?\s*\d+[a-zA-Z]?\s*\)?", re.IGNORECASE)


def load_content_list(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "content_list" in data:
        return data["content_list"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized content_list format in {path}")


def _get_bbox(item: Dict[str, Any]) -> List[float]:
    bbox = item.get("bbox") or item.get("bbox_norm") or item.get("bbox_normed")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(x) for x in bbox]
    return [0.0, 0.0, 0.0, 0.0]


def _get_page_no(item: Dict[str, Any]) -> int | None:
    for key in ("page", "page_no", "page_id", "page_idx", "page_index"):
        if key in item:
            try:
                return int(item[key])
            except Exception:
                pass
    return None


def _get_text_level(item: Dict[str, Any]) -> int:
    for key in ("text_level", "level"):
        if key in item:
            try:
                return int(item[key])
            except Exception:
                pass
    return 0


def _map_type(item: Dict[str, Any], text_level: int) -> str:
    raw_type = (item.get("type") or "").lower()
    if text_level >= 1 or raw_type in {"title", "heading", "header"}:
        return "heading"
    if raw_type in {"table", "table_caption"}:
        return "table"
    if raw_type in {"figure", "image", "fig", "figure_caption"}:
        return "figure"
    if raw_type in {"equation", "formula"}:
        return "equation"
    if raw_type in {"code", "algorithm"}:
        return raw_type
    if raw_type in {"list", "bullet"}:
        return "list"
    return "paragraph"


def _extract_text(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("text", "content", "caption", "latex", "equation", "table"):
        if key in item:
            value = item.get(key)
            if isinstance(value, dict) and "text" in value:
                value = value.get("text")
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            if isinstance(value, str) and value.strip():
                parts.append(value)
    for key in ("image_caption", "table_caption", "table_body"):
        if key in item:
            value = item.get(key)
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            if isinstance(value, str) and value.strip():
                parts.append(value)
    if isinstance(item.get("cells"), list):
        parts.append(" ".join(str(c) for c in item["cells"]))
    if not parts:
        return ""
    return normalize_text(" ".join(parts))


def extract_anchors(text: str) -> List[str]:
    return list({m.group(0).strip() for m in ANCHOR_RE.finditer(text)})


def union_bbox(bboxes: List[List[float]]) -> List[float]:
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def build_eobjs_from_content_list(
    content_list: List[Dict[str, Any]],
    paper_id: str,
    media_root: Path,
) -> Tuple[List[EvidenceObject], Dict[str, str]]:
    eobjs: List[EvidenceObject] = []
    anchor_index: Dict[str, str] = {}
    section_stack: List[str] = []

    for cl_id, item in enumerate(content_list):
        text_level = _get_text_level(item)
        eobj_type = _map_type(item, text_level)
        text = _extract_text(item)
        if not text:
            continue
        if eobj_type == "heading":
            level = max(text_level, 1)
            section_stack = section_stack[: level - 1]
            section_stack.append(text)
        section_path = list(section_stack)
        bbox = _get_bbox(item)
        page_no = _get_page_no(item)
        anchors = extract_anchors(text)
        media_path = None
        img_path = item.get("img_path")
        if isinstance(img_path, str) and img_path.strip():
            media_path = str(media_root / img_path)
        eobj_id = f"{paper_id}_{eobj_type}_{len(eobjs):05d}"
        char_map = [{"start": 0, "end": len(text), "cl_id": cl_id, "bbox": bbox}]
        eobj = EvidenceObject(
            eobj_id=eobj_id,
            paper_id=paper_id,
            type=eobj_type,
            section_path=section_path,
            page_no=page_no,
            bbox_union_norm=bbox,
            text_concat=text,
            source_cl_ids=[cl_id],
            char_map=char_map,
            anchors=anchors,
            media_path=media_path,
        )
        eobjs.append(eobj)
        for anchor in anchors:
            anchor_index[anchor] = eobj_id
    return eobjs, anchor_index


def load_mineru_outputs(base_dir: str | Path, paper_id: str) -> Tuple[List[EvidenceObject], Dict[str, str]]:
    base_dir = Path(base_dir)
    auto_dir = base_dir / paper_id / "auto"
    content_path = auto_dir / f"{paper_id}_content_list.json"
    if not content_path.exists():
        raise FileNotFoundError(content_path)
    content_list = load_content_list(content_path)
    return build_eobjs_from_content_list(content_list, paper_id, auto_dir)
