from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

from .utils import normalize_text


WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_match(text: str) -> Tuple[str, List[int]]:
    norm_chars: List[str] = []
    mapping: List[int] = []
    for idx, ch in enumerate(text):
        if ch.isspace():
            continue
        norm_chars.append(ch.lower())
        mapping.append(idx)
    return "".join(norm_chars), mapping


def find_quote_span(text: str, quote: str) -> Optional[Tuple[int, int]]:
    text_norm = normalize_text(text)
    quote_norm = normalize_text(quote)
    if not text_norm or not quote_norm:
        return None
    pos = text_norm.find(quote_norm)
    if pos >= 0:
        return pos, pos + len(quote_norm)

    norm_text, mapping = _normalize_for_match(text)
    norm_quote, _ = _normalize_for_match(quote)
    pos = norm_text.find(norm_quote)
    if pos >= 0:
        start = mapping[pos]
        end = mapping[pos + len(norm_quote) - 1] + 1
        return start, end

    matcher = difflib.SequenceMatcher(None, norm_text, norm_quote)
    match = matcher.find_longest_match(0, len(norm_text), 0, len(norm_quote))
    if match.size / max(len(norm_quote), 1) < 0.6:
        return None
    start = mapping[match.a]
    end = mapping[match.a + match.size - 1] + 1
    return start, end


def span_to_cl_ids(char_map: List[Dict[str, Any]], span: Tuple[int, int]) -> List[int]:
    start, end = span
    cl_ids: List[int] = []
    for segment in char_map:
        seg_start = segment["start"]
        seg_end = segment["end"]
        if end <= seg_start or start >= seg_end:
            continue
        cl_ids.append(segment["cl_id"])
    return cl_ids


def span_to_bbox(char_map: List[Dict[str, Any]], span: Tuple[int, int]) -> List[float]:
    start, end = span
    bboxes: List[List[float]] = []
    for segment in char_map:
        seg_start = segment["start"]
        seg_end = segment["end"]
        if end <= seg_start or start >= seg_end:
            continue
        if "bbox" in segment:
            bboxes.append(segment["bbox"])
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def align_quotes(eobj: Dict[str, Any], quotes: List[str]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    text = eobj.get("text_concat", "")
    char_map = eobj.get("char_map", [])
    for quote in quotes:
        span = find_quote_span(text, quote)
        if not span:
            continue
        cl_ids = span_to_cl_ids(char_map, span)
        bbox = span_to_bbox(char_map, span)
        spans.append(
            {
                "eobj_id": eobj["eobj_id"],
                "span": {"char_start": span[0], "char_end": span[1]},
                "source_cl_ids": cl_ids,
                "bbox_union_norm": bbox,
                "page_no": eobj.get("page_no"),
                "quote": quote,
            }
        )
    return spans
