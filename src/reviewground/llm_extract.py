from __future__ import annotations

import difflib
import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List

from .http_client import urlopen_with_ssl
from .schemas import Claim
from .utils import normalize_text, sentence_split, simple_tokenize


ANCHOR_RE = re.compile(
    r"\b(?:Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?|Section|Sec\.?|Appendix|Algorithm)\s*[A-Za-z0-9\-\(\)]+",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
METRIC_RE = re.compile(
    r"\b(?:accuracy|acc|f1|precision|recall|bleu|rouge|map|auc|perplexity|error rate|rmse|mae|mse)\b",
    re.IGNORECASE,
)


def classify_claim_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(lack|missing|omit|not include|absent)\b", lowered):
        return "missing_exp"
    if re.search(r"\b(outperform|underperform|improve|increase|decrease|drop|better than|worse than)\b", lowered):
        return "experiment_result"
    if re.search(r"\b(we propose|the paper proposes|our method|algorithm|model|approach)\b", lowered):
        return "method_desc"
    return "method_desc"


def extract_numbers(text: str) -> List[Dict[str, Any]]:
    numbers = []
    for m in NUMBER_RE.finditer(text):
        val = m.group(0)
        unit = "%" if val.endswith("%") else ""
        try:
            value = float(val.rstrip("%"))
        except Exception:
            continue
        numbers.append({"value": value, "unit": unit})
    return numbers


def extract_entities(text: str) -> List[str]:
    # naive: capitalized tokens or dataset-like tokens
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-]+\b", text)
    return list(dict.fromkeys(tokens))


def is_checkable(text: str) -> bool:
    if ANCHOR_RE.search(text):
        return True
    if NUMBER_RE.search(text):
        return True
    if extract_entities(text):
        return True
    return False


def is_checkable_strict(text: str) -> bool:
    if ANCHOR_RE.search(text):
        return True
    if NUMBER_RE.search(text):
        return True
    if METRIC_RE.search(text):
        return True
    return False


def dedup_claims(existing: List[str], candidate: str, threshold: float) -> bool:
    for prior in existing:
        if difflib.SequenceMatcher(None, prior, candidate).ratio() >= threshold:
            return True
    return False


def extract_claims_from_utterance(
    utterance: Dict[str, Any],
    max_tokens: int,
    dedup_threshold: float,
) -> List[Claim]:
    text = utterance["text"]
    dialogue_context = utterance.get("context_utterances")
    if not dialogue_context:
        dialogue_context = utterance.get("dialogue_context") or []
    sentences = sentence_split(text)
    claims: List[Claim] = []
    seen_norm: List[str] = []

    for idx, sent in enumerate(sentences):
        sent = normalize_text(sent)
        if not sent:
            continue
        token_count = len(simple_tokenize(sent))
        if token_count > max_tokens:
            continue
        if not is_checkable(sent):
            continue
        norm = sent.lower()
        if dedup_claims(seen_norm, norm, dedup_threshold):
            continue
        seen_norm.append(norm)
        claim_id = f"{utterance['paper_id']}_{utterance['utterance_id']}_{idx:03d}"
        anchors = list({m.group(0).strip() for m in ANCHOR_RE.finditer(sent)})
        claim = Claim(
            claim_id=claim_id,
            paper_id=utterance["paper_id"],
            thread_id=utterance["thread_id"],
            utterance_id=utterance["utterance_id"],
            claim_text=sent,
            claim_type=classify_claim_type(sent),
            checkability="paper_only",
            anchors_mentioned=anchors,
            entities=extract_entities(sent),
            numbers=extract_numbers(sent),
            raw_span=sent,
            context_utterances=(
                [{"role": utterance["role"], "text": text, "stage": utterance.get("stage"), "utterance_id": utterance.get("utterance_id")}]
                + dialogue_context
            ),
        )
        claims.append(claim)

    return claims


class LLMExtractionError(RuntimeError):
    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
        spans: List[str] | None = None,
        utterance_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.spans = spans or []
        self.utterance_text = utterance_text


def _call_chat_completions(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DMXAPI/1.0.0",
        },
    )
    with urlopen_with_ssl(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    try:
        obj = json.loads(raw)
        return obj["choices"][0]["message"]["content"]
    except Exception:
        return raw


def _parse_json_array_strict(text: str) -> List[Any]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    raise LLMExtractionError("LLM output is not a valid JSON array", raw_response=text)


def _normalize_with_map(text: str) -> tuple[str, List[int]]:
    out_chars: List[str] = []
    mapping: List[int] = []
    in_ws = False
    for idx, ch in enumerate(text):
        if ch.isspace():
            if not in_ws:
                out_chars.append(" ")
                mapping.append(idx)
                in_ws = True
            continue
        out_chars.append(ch)
        mapping.append(idx)
        in_ws = False
    return "".join(out_chars), mapping


def _best_sentence_match(span: str, text: str, min_ratio: float = 0.85) -> str | None:
    span_norm = normalize_text(span).lower()
    span_tokens = simple_tokenize(span_norm)
    if len(span_tokens) < 4:
        return None
    span_numbers = [m.group(0) for m in NUMBER_RE.finditer(span_norm)]
    span_anchors = [m.group(0) for m in ANCHOR_RE.finditer(span_norm)]

    best_ratio = 0.0
    best_sent = None
    for sent in sentence_split(text):
        sent_norm = normalize_text(sent)
        if not sent_norm:
            continue
        sent_low = sent_norm.lower()
        if span_numbers and not all(num in sent_low for num in span_numbers):
            continue
        if span_anchors and not any(anchor in sent_low for anchor in span_anchors):
            continue
        sent_tokens = simple_tokenize(sent_low)
        if not sent_tokens:
            continue
        ratio = difflib.SequenceMatcher(None, span_tokens, sent_tokens).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_sent = sent_norm
    if best_ratio >= min_ratio:
        return best_sent
    return None


def _resolve_span_fuzzy(span: str, text: str) -> str | None:
    if not span:
        return None
    cleaned = span.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    if cleaned in text:
        return cleaned

    low_text = text.lower()
    low_span = cleaned.lower()
    idx = low_text.find(low_span)
    if idx != -1:
        return text[idx : idx + len(cleaned)]

    norm_text, mapping = _normalize_with_map(text)
    norm_span = normalize_text(cleaned)
    norm_idx = norm_text.find(norm_span)
    if norm_idx != -1:
        start = mapping[norm_idx]
        end = mapping[norm_idx + len(norm_span) - 1] + 1
        return text[start:end]
    best_sentence = _best_sentence_match(cleaned, text)
    if best_sentence is not None:
        return best_sentence
    return None


def _normalize_span(span: str) -> str:
    return normalize_text(span)


def extract_claims_from_utterance_llm(
    utterance: Dict[str, Any],
    max_tokens: int,
    dedup_threshold: float,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    max_claims_per_utterance: int | None = 6,
    fuzzy_match: bool = True,
) -> List[Claim]:
    text = utterance["text"]
    dialogue_context = utterance.get("context_utterances") or utterance.get("dialogue_context") or []

    context_lines: List[str] = []
    for item in dialogue_context[:3]:
        role = item.get("role", "unknown")
        stage = item.get("stage", "")
        ctx_text = normalize_text(item.get("text", ""))
        if ctx_text:
            prefix = f"{role}/{stage}".rstrip("/")
            context_lines.append(f"- {prefix}: {ctx_text}")
    context_block = "\n".join(context_lines)

    system_prompt = (
        "You extract atomic, paper‑groundable claims from a review utterance.\n"
        "Output ONLY a JSON array. Each item: {\"span\": \"...\"}.\n"
        "SPAN MUST be an EXACT, verbatim substring of the utterance.\n"
        "Only output a span if it is a concrete, checkable claim that is verifiable by the paper itself.\n"
        "A valid span should include an explicit anchor/number/metric (Figure/Table/Section/Equation or quantitative values/metrics).\n"
        "SKIP subjective/opinion statements (clarity/novelty/interesting), reviewer confidence, speculation, or vague summaries.\n"
        "Do NOT output anything to fill a quota. If unsure or no valid claims, output [] only.\n"
        "STRICT: no paraphrase, no grammar fixes, no added words, no quotes/ellipses."
    )
    user_prompt = "Task: list atomic, paper-checkable claims.\n"
    if max_claims_per_utterance and max_claims_per_utterance > 0:
        user_prompt += f"Max claims: {max_claims_per_utterance}\n"
    user_prompt += "Utterance:\n"
    user_prompt += f"\"\"\"\n{text}\n\"\"\"\n"
    if context_block:
        user_prompt += f"\nContext (optional):\n{context_block}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        content = _call_chat_completions(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise LLMExtractionError(f"LLM call failed: {exc}") from exc

    items = _parse_json_array_strict(content)
    spans: List[str] = []
    for item in items:
        if isinstance(item, str):
            spans.append(item)
        elif isinstance(item, dict):
            span = item.get("span") or item.get("claim") or item.get("text")
            if isinstance(span, str):
                spans.append(span)
        else:
            raise LLMExtractionError(
                "Invalid JSON item type",
                raw_response=content,
                spans=spans,
                utterance_text=text,
            )

    claims: List[Claim] = []
    seen_norm: List[str] = []
    invalid_spans: List[str] = []
    for idx, span in enumerate(spans):
        if max_claims_per_utterance and max_claims_per_utterance > 0 and idx >= max_claims_per_utterance:
            break
        span = _normalize_span(span)
        if not span:
            continue
        if span not in text:
            if fuzzy_match:
                resolved = _resolve_span_fuzzy(span, text)
                if resolved is None:
                    invalid_spans.append(span)
                    continue
                span = resolved
            else:
                invalid_spans.append(span)
                continue
        token_count = len(simple_tokenize(span))
        if token_count > max_tokens:
            continue
        if not is_checkable_strict(span):
            continue
        norm = span.lower()
        if dedup_claims(seen_norm, norm, dedup_threshold):
            continue
        seen_norm.append(norm)
        claim_id = f"{utterance['paper_id']}_{utterance['utterance_id']}_{idx:03d}"
        anchors = list({m.group(0).strip() for m in ANCHOR_RE.finditer(span)})
        claim = Claim(
            claim_id=claim_id,
            paper_id=utterance["paper_id"],
            thread_id=utterance["thread_id"],
            utterance_id=utterance["utterance_id"],
            claim_text=span,
            claim_type=classify_claim_type(span),
            checkability="paper_only",
            anchors_mentioned=anchors,
            entities=extract_entities(span),
            numbers=extract_numbers(span),
            raw_span=span,
            context_utterances=(
                [{"role": utterance["role"], "text": text, "stage": utterance.get("stage"), "utterance_id": utterance.get("utterance_id")}]
                + dialogue_context
            ),
        )
        claims.append(claim)

    if not claims and invalid_spans:
        raise LLMExtractionError(
            "All spans are not exact substrings of utterance",
            raw_response=content,
            spans=spans,
            utterance_text=text,
        )

    return claims
