from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from .http_client import urlopen_with_ssl
from .json_repair import parse_json_object_with_repair
from .utils import normalize_text


ALLOWED_LABELS = {"SUPPORTED", "CONTRADICTED", "NOT_FOUND", "UNDECIDABLE"}
LABEL_RE = re.compile(r"\b(SUPPORTED|CONTRADICTED|NOT_FOUND|UNDECIDABLE)\b", re.IGNORECASE)

SYSTEM_UNSUPPORTED_MODELS = (
    # Some OpenAI-compatible proxies reject system messages for specific model families (e.g. o1).
    "o1",
    "o3",
    # DMX may expose certain Qwen3 variants without system-role support.
    "qwen3",
)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"reasoning", "analysis", "thinking"}:
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text.strip():
            return text
        parts = content.get("parts")
        if isinstance(parts, list):
            out: List[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "").strip().lower()
                if part_type in {"reasoning", "analysis", "thinking"}:
                    continue
                ptext = part.get("text")
                if isinstance(ptext, str) and ptext.strip():
                    out.append(ptext)
            if out:
                return "\n".join(out)
    return ""


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    detail = ""
    try:
        detail = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        detail = ""
    detail = (detail or "").strip()
    if not detail:
        return ""
    try:
        obj = json.loads(detail)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        err = obj.get("error")
        if isinstance(err, dict):
            for key in ("message", "msg", "detail"):
                val = err.get(key)
                if val:
                    detail = str(val).strip()
                    break
        if not detail:
            for key in ("message", "msg", "detail"):
                val = obj.get(key)
                if val:
                    detail = str(val).strip()
                    break
    detail = re.sub(r"\s+", " ", detail).strip()
    if len(detail) > 500:
        detail = detail[:500].rstrip()
    return detail


def _looks_like_system_role_rejection(detail: str) -> bool:
    if not detail:
        return False
    lower = detail.lower()
    if "system" not in lower:
        return False
    en_markers = (
        "not support",
        "not supported",
        "unsupported",
        "not allowed",
        "invalid",
        "disallow",
        "forbidden",
    )
    if any(m in lower for m in en_markers):
        return True
    return False


def _merge_system_into_user(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_texts: List[str] = []
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        if role == "system":
            sys_text = _content_to_text(msg.get("content")).strip()
            if sys_text:
                system_texts.append(sys_text)
            continue
        out.append(dict(msg))

    merged = "\n\n".join(system_texts).strip()
    if not merged:
        return out or messages

    if not out:
        return [{"role": "user", "content": merged}]

    for i, msg in enumerate(out):
        if str(msg.get("role") or "") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = f"{merged}\n\n{content}".strip()
        elif isinstance(content, list):
            msg["content"] = [{"type": "text", "text": merged}] + content
        else:
            msg["content"] = (f"{merged}\n\n{_content_to_text(content)}").strip() or merged
        out[i] = msg
        break
    else:
        out.insert(0, {"role": "user", "content": merged})
    return out


def _extract_response_text(obj: Dict[str, Any]) -> str:
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}

        candidates = [
            message.get("content"),
            message.get("output_text"),
            choice0.get("content"),
            choice0.get("text"),
        ]
        for candidate in candidates:
            text = _content_to_text(candidate)
            if normalize_text(text):
                return text

        reasoning_candidates = [
            message.get("reasoning_content"),
            message.get("reasoning"),
            message.get("analysis"),
            choice0.get("reasoning_content"),
            choice0.get("reasoning"),
            choice0.get("analysis"),
        ]
        for candidate in reasoning_candidates:
            text = _content_to_text(candidate)
            if normalize_text(text):
                return text

        delta = choice0.get("delta")
        if isinstance(delta, dict):
            text = _content_to_text(delta.get("content"))
            if normalize_text(text):
                return text

    # Fallback for Gemini-native style payloads via proxy.
    gemini_candidates = obj.get("candidates")
    if isinstance(gemini_candidates, list) and gemini_candidates:
        cand0 = gemini_candidates[0] if isinstance(gemini_candidates[0], dict) else {}
        text = _content_to_text(cand0.get("content"))
        if normalize_text(text):
            return text
        text = _content_to_text(cand0.get("output"))
        if normalize_text(text):
            return text

    text = _content_to_text(obj.get("output_text"))
    if normalize_text(text):
        return text
    response = obj.get("response")
    if isinstance(response, dict):
        text = _content_to_text(response.get("output_text"))
        if normalize_text(text):
            return text
    for key in ("reasoning_content", "reasoning", "analysis"):
        text = _content_to_text(obj.get(key))
        if normalize_text(text):
            return text
    return ""


def _extract_finish_reason(obj: Dict[str, Any]) -> str:
    choices = obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if reason:
            return str(reason)
    gemini_candidates = obj.get("candidates")
    if isinstance(gemini_candidates, list) and gemini_candidates and isinstance(gemini_candidates[0], dict):
        reason = gemini_candidates[0].get("finishReason")
        if reason:
            return str(reason)
    return ""


def _call_chat_completions(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
    disable_thinking: bool = True,
    thinking_type: str | None = None,
    effort: str | None = None,
) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"
    model_lower = str(model or "").lower()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    # DashScope-compatible endpoints support a hard switch for no-thinking mode.
    if disable_thinking and "dashscope" in base.lower():
        payload["enable_thinking"] = False
    elif not disable_thinking and "claude" in model_lower:
        if thinking_type:
            payload["thinking"] = {"type": str(thinking_type)}
        if effort:
            payload["effort"] = str(effort)

    retryable_http_codes = {408, 409, 425, 429, 500, 502, 503, 504}

    def _post(p: Dict[str, Any]) -> str:
        data = json.dumps(p).encode("utf-8")
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
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                with urlopen_with_ssl(req, timeout=60) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                last_err = exc
                if exc.code not in retryable_http_codes or attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))
            except urllib.error.URLError as exc:
                last_err = exc
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))
        if last_err is not None:
            raise last_err
        raise RuntimeError("LLM request failed without explicit error")

    try:
        raw = _post(payload)
    except urllib.error.HTTPError as exc:
        detail = _http_error_detail(exc)
        has_system = any(str(m.get("role") or "") == "system" for m in messages)
        system_family = any(model_lower.startswith(prefix) for prefix in SYSTEM_UNSUPPORTED_MODELS)
        if exc.code == 400 and has_system and (_looks_like_system_role_rejection(detail) or system_family):
            alt = dict(payload)
            alt["messages"] = _merge_system_into_user(messages)
            try:
                raw = _post(alt)
            except urllib.error.HTTPError as exc2:
                detail2 = _http_error_detail(exc2)
                raise RuntimeError(f"HTTP {exc2.code}: {detail2 or exc2.reason}") from None
        else:
            raise RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}") from None

    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    text = _extract_response_text(obj)
    if normalize_text(text):
        return text
    finish_reason = _extract_finish_reason(obj) or "-"
    raise RuntimeError(f"empty model content; finish_reason={finish_reason}")


def _parse_json_obj(text: str) -> Dict[str, Any]:
    return parse_json_object_with_repair(text)


def _format_candidates(candidates: List[Dict[str, Any]], max_chars: int) -> str:
    lines: List[str] = []
    for cand in candidates:
        text = normalize_text(cand.get("text", ""))
        if len(text) > max_chars:
            text = text[:max_chars].rstrip()
        section = "/".join(cand.get("section_path") or [])
        lines.append(
            f"- id={cand.get('eobj_id')} type={cand.get('type')} page={cand.get('page_no')} section={section}\n"
            f"  text={text}"
        )
    return "\n".join(lines)


def _encode_image_data_url(media_path: str, max_image_bytes: int) -> str | None:
    if not isinstance(media_path, str) or not media_path.strip():
        return None
    path_str = media_path.strip()
    if path_str.startswith("http://") or path_str.startswith("https://"):
        return path_str
    try:
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return None
        file_size = path.stat().st_size
        if file_size <= 0:
            return None
        if max_image_bytes > 0 and file_size > max_image_bytes:
            return None
        raw = path.read_bytes()
        if max_image_bytes > 0 and len(raw) > max_image_bytes:
            return None
        mime, _ = mimetypes.guess_type(path.name)
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def _build_user_content_with_images(
    claim_text: str,
    cand_block: str,
    candidates: List[Dict[str, Any]],
    max_images: int,
    max_image_bytes: int,
    image_detail: str,
) -> tuple[List[Dict[str, Any]], int]:
    text_prompt = (
        f"Claim:\n{claim_text}\n\nCandidates:\n{cand_block}\n\n"
        "Candidate images (if any) are attached below. Each image is preceded by its candidate metadata."
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]

    if max_images <= 0:
        return content, 0

    used = 0
    seen_paths: set[str] = set()
    for cand in candidates:
        if used >= max_images:
            break
        media_path = cand.get("media_path")
        if not isinstance(media_path, str) or not media_path.strip():
            continue
        norm_path = media_path.strip()
        if norm_path in seen_paths:
            continue
        seen_paths.add(norm_path)
        image_url = _encode_image_data_url(norm_path, max_image_bytes=max_image_bytes)
        if not image_url:
            continue
        section = "/".join(cand.get("section_path") or [])
        content.append(
            {
                "type": "text",
                "text": (
                    f"[candidate_image] id={cand.get('eobj_id')} type={cand.get('type')} "
                    f"page={cand.get('page_no')} section={section}"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": image_detail},
            }
        )
        used += 1
    return content, used


def _fallback_extract_from_raw(text: str, allowed_ids: set[str]) -> Dict[str, Any] | None:
    m = LABEL_RE.search(text)
    if not m:
        return None
    label = m.group(1).upper()
    if label not in ALLOWED_LABELS:
        return None
    if label in {"NOT_FOUND", "UNDECIDABLE"}:
        return {"label": label, "evidence_sets": []}

    hits = []
    for eobj_id in allowed_ids:
        pos = text.find(eobj_id)
        if pos >= 0:
            hits.append((pos, eobj_id))
    hits.sort(key=lambda x: x[0])
    ordered = []
    seen = set()
    for _, eobj_id in hits:
        if eobj_id in seen:
            continue
        seen.add(eobj_id)
        ordered.append(eobj_id)
    if label in {"SUPPORTED", "CONTRADICTED"} and not ordered:
        return {"label": "NOT_FOUND", "evidence_sets": []}

    evidence_sets: List[List[Dict[str, Any]]] = []
    for eobj_id in ordered[:3]:
        evidence_sets.append([{"eobj_id": eobj_id, "quotes": []}])
    return {"label": label, "evidence_sets": evidence_sets}


def locate_evidence_llm(
    claim: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    max_candidates: int,
    max_chars: int,
    use_images: bool = True,
    max_images: int = 4,
    max_image_bytes: int = 800_000,
    image_detail: str = "low",
    ids_only: bool = False,
    disable_thinking: bool = True,
    thinking_type: str | None = None,
    effort: str | None = None,
) -> Dict[str, Any]:
    claim_text = normalize_text(claim.get("claim_text", ""))
    cand_subset = candidates[:max_candidates] if max_candidates > 0 else candidates
    cand_block = _format_candidates(cand_subset, max_chars=max_chars)

    if ids_only:
        system_prompt = (
            "You are a strict JSON evidence locator for scientific claims.\n"
            "Output exactly one JSON object and nothing else.\n"
            "Schema: {\"label\":\"SUPPORTED|CONTRADICTED|NOT_FOUND|UNDECIDABLE\",\"evidence_sets\":[[{\"eobj_id\":\"...\"}]]}.\n"
            "Use only eobj_id values from candidates.\n"
            "If candidate images are provided, use them only to reason about support/contradiction.\n"
            "Do not ask follow-up questions, do not request additional input, and do not address the user.\n"
            "Do not output reasoning traces or analysis; output the final JSON directly.\n"
            "Use double quotes for all keys and strings.\n"
            "Escape backslashes as \\\\ and newlines as \\n in JSON strings.\n"
            "No markdown, no code fence, no comments, no extra keys.\n"
            "Max 3 evidence sets, max 5 evidence items per set, and keep evidence minimal.\n"
            "If insufficient evidence, return {\"label\":\"NOT_FOUND\",\"evidence_sets\":[]}."
        )
    else:
        system_prompt = (
            "You are a strict JSON evidence locator for scientific claims.\n"
            "Output exactly one JSON object and nothing else.\n"
            "Schema: {\"label\":\"SUPPORTED|CONTRADICTED|NOT_FOUND|UNDECIDABLE\",\"evidence_sets\":[[{\"eobj_id\":\"...\",\"quotes\":[\"...\"]}]]}.\n"
            "Use only eobj_id values from candidates. quotes must be exact substrings from candidate text.\n"
            "If candidate images are provided, use them only to reason about support/contradiction.\n"
            "Still cite evidence via eobj_id and text quotes from candidate text only.\n"
            "Do not ask follow-up questions, do not request additional input, and do not address the user.\n"
            "Do not output reasoning traces or analysis; output the final JSON directly.\n"
            "Use double quotes for all keys and strings.\n"
            "Escape backslashes as \\\\ and newlines as \\n in JSON strings.\n"
            "No markdown, no code fence, no comments, no extra keys.\n"
            "Max 3 evidence sets, max 5 evidence items per set, and keep evidence minimal.\n"
            "If insufficient evidence, return {\"label\":\"NOT_FOUND\",\"evidence_sets\":[]}."
        )
    no_think_prefix = "/no_think\n" if disable_thinking else ""
    user_prompt = f"{no_think_prefix}Claim:\n{claim_text}\n\nCandidates:\n{cand_block}\n"
    text_only_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages = text_only_messages
    used_images = 0
    if use_images:
        content_with_images, used_images = _build_user_content_with_images(
            claim_text=f"{no_think_prefix}{claim_text}",
            cand_block=cand_block,
            candidates=cand_subset,
            max_images=max_images,
            max_image_bytes=max_image_bytes,
            image_detail=image_detail,
        )
        if used_images > 0:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_with_images},
            ]

    try:
        content = _call_chat_completions(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            disable_thinking=disable_thinking,
            thinking_type=thinking_type,
            effort=effort,
        )
    except Exception:
        if used_images <= 0:
            raise
        content = _call_chat_completions(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=text_only_messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            disable_thinking=disable_thinking,
            thinking_type=thinking_type,
            effort=effort,
        )

    allowed_ids = {c.get("eobj_id") for c in cand_subset if c.get("eobj_id")}
    try:
        data = _parse_json_obj(content)
    except Exception as exc:
        fallback = _fallback_extract_from_raw(content, allowed_ids)
        if fallback is None:
            snippet = normalize_text(content)[:400]
            raise ValueError(f"{exc}; raw_snippet={snippet}") from exc
        data = fallback
    label = str(data.get("label", "NOT_FOUND")).upper()
    if label not in ALLOWED_LABELS:
        label = "NOT_FOUND"
    evidence_sets = data.get("evidence_sets", [])
    if not isinstance(evidence_sets, list):
        evidence_sets = []

    cleaned_sets: List[List[Dict[str, Any]]] = []
    for ev_set in evidence_sets[:3]:
        if not isinstance(ev_set, list):
            continue
        cleaned: List[Dict[str, Any]] = []
        for ev in ev_set[:5]:
            if not isinstance(ev, dict):
                continue
            eobj_id = ev.get("eobj_id")
            if eobj_id not in allowed_ids:
                continue
            if ids_only:
                cleaned.append({"eobj_id": eobj_id})
            else:
                quotes = ev.get("quotes") or []
                if not isinstance(quotes, list):
                    quotes = []
                quotes = [normalize_text(q) for q in quotes if isinstance(q, str) and normalize_text(q)]
                cleaned.append({"eobj_id": eobj_id, "quotes": quotes})
        if cleaned:
            cleaned_sets.append(cleaned)

    if label in {"NOT_FOUND", "UNDECIDABLE"}:
        cleaned_sets = []

    return {
        "claim_id": claim.get("claim_id"),
        "label": label,
        "evidence_sets": cleaned_sets,
    }
