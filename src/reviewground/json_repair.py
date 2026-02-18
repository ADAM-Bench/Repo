from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List


CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
UNQUOTED_LABEL_RE = re.compile(
    r'("label"\s*:\s*)(SUPPORTED|CONTRADICTED|NOT_FOUND|UNDECIDABLE)(\b)',
    re.IGNORECASE,
)

SMART_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def _extract_first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _escape_invalid_backslashes_in_strings(text: str) -> str:
    out: List[str] = []
    in_str = False
    esc = False
    for ch in text:
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            continue
        if esc:
            if ch not in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                out.append("\\")
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = False
        out.append(ch)
    return "".join(out)


def _repair_json_like(text: str) -> str:
    fixed = text.translate(SMART_QUOTES).strip().strip(";")
    fixed = fixed.replace("\ufeff", "")
    fixed = re.sub(r"^\s*json\s*", "", fixed, flags=re.IGNORECASE)
    fixed = UNQUOTED_LABEL_RE.sub(lambda m: f'{m.group(1)}"{m.group(2).upper()}"', fixed)
    fixed = _escape_invalid_backslashes_in_strings(fixed)
    fixed = TRAILING_COMMA_RE.sub(r"\1", fixed)
    return fixed


def _insert_comma_at(text: str, pos: int) -> str | None:
    if pos <= 0 or pos > len(text):
        return None
    i = pos - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    j = pos
    while j < len(text) and text[j].isspace():
        j += 1
    if i < 0 or j >= len(text):
        return None

    prev_ch = text[i]
    next_ch = text[j]
    if prev_ch in "{[:,":
        return None
    if next_ch in "},]:":
        return None
    return text[:pos] + "," + text[pos:]


def _guided_repairs_from_error(text: str, exc: json.JSONDecodeError) -> List[str]:
    out: List[str] = []
    msg = exc.msg
    if "Expecting ',' delimiter" in msg:
        patched = _insert_comma_at(text, exc.pos)
        if patched:
            out.append(patched)
        if 0 <= exc.pos < len(text):
            next_quote = text.find('"', exc.pos)
            if next_quote > exc.pos:
                patched = _insert_comma_at(text, next_quote)
                if patched:
                    out.append(patched)
    if "Extra data" in msg:
        obj = _extract_first_balanced_object(text)
        if obj and obj != text:
            out.append(obj)
    if "Invalid \\escape" in msg:
        patched = _escape_invalid_backslashes_in_strings(text)
        if patched != text:
            out.append(patched)
    return out


def _candidate_texts(raw: str) -> Iterable[str]:
    base = raw.strip()
    if base:
        yield base

    for block in CODE_FENCE_RE.findall(raw):
        block = block.strip()
        if block:
            yield block

    balanced = _extract_first_balanced_object(raw)
    if balanced:
        yield balanced.strip()

    if base:
        balanced2 = _extract_first_balanced_object(base)
        if balanced2:
            yield balanced2.strip()


def parse_json_object_with_repair(raw: str) -> Dict[str, Any]:
    seen = set()
    errors: List[str] = []
    for candidate in _candidate_texts(raw):
        queue: List[str] = [candidate, _repair_json_like(candidate)]
        while queue:
            variant = queue.pop(0)
            if not variant or variant in seen:
                continue
            seen.add(variant)
            try:
                data = json.loads(variant)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                for repaired in _guided_repairs_from_error(variant, exc):
                    if repaired and repaired not in seen:
                        queue.append(repaired)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
    if errors:
        raise ValueError(errors[0])
    raise ValueError("LLM output is not a valid JSON object")
