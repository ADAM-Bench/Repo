from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .utils import simple_tokenize


MAX_EOBJS_PER_SET = 4


def overlap_ratio(claim_text: str, evidence_text: str) -> float:
    claim_tokens = set(simple_tokenize(claim_text))
    evidence_tokens = set(simple_tokenize(evidence_text))
    if not claim_tokens or not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def minimize_evidence_set(
    claim_text: str,
    evidence_set: List[Dict[str, Any]],
    eobj_map: Dict[str, Dict[str, Any]],
    threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for span in evidence_set:
        eobj = eobj_map.get(span["eobj_id"], {})
        score = overlap_ratio(claim_text, eobj.get("text_concat", ""))
        if score >= threshold:
            kept.append(span)
    return kept or evidence_set[:1]


def dedup_evidence_set(evidence_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result: List[Dict[str, Any]] = []
    for span in evidence_set:
        key = (span.get("eobj_id"), span.get("span", {}).get("char_start"), span.get("span", {}).get("char_end"))
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return result


def qc_sample(
    sample: Dict[str, Any],
    eobj_map: Dict[str, Dict[str, Any]],
    require_supported_has_evidence: bool,
    allow_not_found_empty: bool,
    evidence_minimize: bool,
) -> Tuple[bool, Dict[str, Any]]:
    label = sample.get("label")
    evidence_sets = sample.get("evidence_sets", [])

    if label in {"SUPPORTED", "CONTRADICTED"} and require_supported_has_evidence:
        if not evidence_sets:
            return False, sample
    if label == "NOT_FOUND" and not allow_not_found_empty:
        if not evidence_sets:
            return False, sample

    cleaned_sets: List[List[Dict[str, Any]]] = []
    for ev_set in evidence_sets:
        ev_set = dedup_evidence_set(ev_set)
        if len(ev_set) > MAX_EOBJS_PER_SET:
            return False, sample
        if any(ev.get("eobj_id") not in eobj_map for ev in ev_set):
            return False, sample
        if evidence_minimize:
            ev_set = minimize_evidence_set(sample.get("claim_text", ""), ev_set, eobj_map)
        cleaned_sets.append(ev_set)

    sample["evidence_sets"] = cleaned_sets
    return True, sample
