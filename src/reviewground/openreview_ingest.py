from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .schemas import PaperManifest, Thread, Utterance
from .utils import normalize_text, simple_tokenize


ANCHOR_RE = re.compile(
    r"\b(?:Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?|Section|Sec\.?|Appendix)\s*[A-Za-z0-9\-\(\)]+",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def extract_text_fields(content: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    for key, value in content.items():
        value = unwrap_value(value)
        if isinstance(value, str):
            if value.strip():
                fields.append(value.strip())
        elif isinstance(value, list):
            joined = ", ".join(str(v) for v in value if str(v).strip())
            if joined:
                fields.append(joined)
    return fields


def _get_invitations(note: Dict[str, Any]) -> List[str]:
    invitations: List[str] = []
    invitation = note.get("invitation")
    if isinstance(invitation, str) and invitation:
        invitations.append(invitation)
    invs = note.get("invitations")
    if isinstance(invs, list):
        invitations.extend(str(i) for i in invs if str(i))
    elif isinstance(invs, str) and invs:
        invitations.append(invs)
    return invitations


def find_submission_note(replies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for note in replies:
        for invitation in _get_invitations(note):
            if "Blind_Submission" in invitation or "Submission" in invitation:
                return note
    # fallback: first note that has title+abstract+pdf
    for note in replies:
        content = note.get("content", {})
        if "title" in content and "abstract" in content and "pdf" in content:
            return note
    return None


def normalize_pdf_url(pdf_value: str) -> str:
    if pdf_value.startswith("http://") or pdf_value.startswith("https://"):
        return pdf_value
    if pdf_value.startswith("/pdf/"):
        return f"https://openreview.net{pdf_value}"
    return pdf_value


def extract_license(content: Dict[str, Any]) -> Optional[str]:
    license_val = content.get("license")
    license_val = unwrap_value(license_val)
    if isinstance(license_val, str) and license_val.strip():
        return license_val.strip()
    # some versions store license in content["data"]
    data = content.get("data")
    if isinstance(data, dict):
        data_val = unwrap_value(data.get("license"))
        if isinstance(data_val, str) and data_val.strip():
            return data_val.strip()
    return None


def extract_primary_area(content: Dict[str, Any]) -> Optional[str]:
    primary = unwrap_value(content.get("primary_area"))
    if isinstance(primary, str) and primary.strip():
        return primary.strip()
    # fallback: first keyword
    keywords = unwrap_value(content.get("keywords"))
    if isinstance(keywords, list) and keywords:
        return str(keywords[0])
    return None


def extract_keywords(content: Dict[str, Any]) -> List[str]:
    keywords = unwrap_value(content.get("keywords"))
    if isinstance(keywords, list):
        return [str(k) for k in keywords if str(k).strip()]
    if isinstance(keywords, str) and keywords.strip():
        return [k.strip() for k in keywords.split(",") if k.strip()]
    return []


def extract_authors(content: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    authors = unwrap_value(content.get("authors"))
    author_ids = unwrap_value(content.get("authorids"))
    if isinstance(authors, list):
        authors = [str(a) for a in authors if str(a).strip()]
    else:
        authors = []
    if isinstance(author_ids, list):
        author_ids = [str(a) for a in author_ids if str(a).strip()]
    else:
        author_ids = []
    return authors, author_ids


def is_author_signature(signatures: List[str], author_ids: List[str]) -> bool:
    author_ids_norm = {a.lower() for a in author_ids}
    for sig in signatures:
        if "author" in sig.lower():
            return True
        tail = sig.split("/")[-1].lower()
        if tail in author_ids_norm or sig.lower() in author_ids_norm:
            return True
    return False


def classify_role_stage(invitation: str, signatures: List[str], author_ids: List[str]) -> Optional[Tuple[str, str]]:
    invitation = invitation or ""
    if "Official_Review" in invitation:
        return "reviewer", "initial_review"
    if "Meta_Review" in invitation:
        return "ac", "meta_review"
    if "Decision" in invitation or "Acceptance_Decision" in invitation:
        return "pc", "decision"
    if "Official_Comment" in invitation or "Comment" in invitation:
        if is_author_signature(signatures, author_ids):
            return "author", "rebuttal_or_discussion"
        return "reviewer", "discussion"
    return None


def extract_surface_cues(text: str) -> Tuple[List[str], List[str]]:
    anchors = list({m.group(0).strip() for m in ANCHOR_RE.finditer(text)})
    numbers = list({m.group(0) for m in NUMBER_RE.finditer(text)})
    cues = anchors + numbers
    return cues, anchors


def is_greeting_only(text: str, min_tokens: int) -> bool:
    tokens = simple_tokenize(text)
    if len(tokens) >= min_tokens:
        return False
    if not tokens:
        return True
    keywords = {"thanks", "thank", "appreciate", "great", "nice", "good", "interesting"}
    return all(t in keywords for t in tokens)


def build_manifest_entry(forum_obj: Dict[str, Any], venue: str) -> Optional[PaperManifest]:
    replies = forum_obj.get("replies", [])
    submission = find_submission_note(replies)
    if not submission:
        return None
    content = submission.get("content", {})
    pdf_val = unwrap_value(content.get("pdf"))
    pdf_url = normalize_pdf_url(pdf_val) if isinstance(pdf_val, str) else None
    authors, author_ids = extract_authors(content)
    manifest = PaperManifest(
        paper_id=forum_obj.get("forum_id"),
        forum_id=forum_obj.get("forum_id"),
        year=int(forum_obj.get("year")),
        venue=venue,
        title=unwrap_value(content.get("title")),
        pdf_url=pdf_url,
        license=extract_license(content),
        primary_area=extract_primary_area(content),
        keywords=extract_keywords(content),
        openreview_note_ids=[r.get("id") for r in replies if r.get("id")],
        authors=authors,
        author_ids=author_ids,
    )
    return manifest


def build_thread(
    forum_obj: Dict[str, Any],
    venue: str,
    min_tokens: int,
) -> Optional[Thread]:
    replies = forum_obj.get("replies", [])
    submission = find_submission_note(replies)
    if not submission:
        return None
    content = submission.get("content", {})
    authors, author_ids = extract_authors(content)
    utterances: List[Utterance] = []
    for note in replies:
        signatures = note.get("signatures") or []
        matched_invitation = ""
        role_stage = None
        for invitation in _get_invitations(note):
            role_stage = classify_role_stage(invitation, signatures, author_ids)
            if role_stage:
                matched_invitation = invitation
                break
        if not role_stage:
            continue
        role, stage = role_stage
        text_fields = extract_text_fields(note.get("content", {}))
        text = normalize_text("\n\n".join(text_fields))
        if not text:
            continue
        if is_greeting_only(text, min_tokens):
            continue
        cues, anchors = extract_surface_cues(text)
        token_count = len(simple_tokenize(text))
        utterances.append(
            Utterance(
                utterance_id=str(note.get("id")),
                role=role,
                stage=stage,
                text=text,
                forum_id=forum_obj.get("forum_id"),
                paper_id=forum_obj.get("forum_id"),
                year=int(forum_obj.get("year")),
                cdate=note.get("cdate"),
                invitation=matched_invitation,
                signatures=signatures,
                surface_cues=cues,
                anchors_mentioned=anchors,
                token_count=token_count,
            )
        )
    if not utterances:
        return None
    return Thread(
        thread_id=forum_obj.get("forum_id"),
        forum_id=forum_obj.get("forum_id"),
        paper_id=forum_obj.get("forum_id"),
        year=int(forum_obj.get("year")),
        utterances=utterances,
    )


def thread_to_dict(thread: Thread) -> Dict[str, Any]:
    return {
        "thread_id": thread.thread_id,
        "forum_id": thread.forum_id,
        "paper_id": thread.paper_id,
        "year": thread.year,
        "utterances": [asdict(u) for u in thread.utterances],
    }
