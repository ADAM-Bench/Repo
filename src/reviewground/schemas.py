from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PaperManifest:
    paper_id: str
    forum_id: str
    year: int
    venue: str
    title: Optional[str] = None
    pdf_url: Optional[str] = None
    license: Optional[str] = None
    primary_area: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    openreview_note_ids: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    author_ids: List[str] = field(default_factory=list)


@dataclass
class Utterance:
    utterance_id: str
    role: str
    stage: str
    text: str
    forum_id: str
    paper_id: str
    year: int
    cdate: Optional[int] = None
    invitation: Optional[str] = None
    signatures: List[str] = field(default_factory=list)
    surface_cues: List[str] = field(default_factory=list)
    anchors_mentioned: List[str] = field(default_factory=list)
    token_count: int = 0


@dataclass
class Thread:
    thread_id: str
    forum_id: str
    paper_id: str
    year: int
    utterances: List[Utterance] = field(default_factory=list)


@dataclass
class EvidenceObject:
    eobj_id: str
    paper_id: str
    type: str
    section_path: List[str]
    page_no: Optional[int]
    bbox_union_norm: List[float]
    text_concat: str
    source_cl_ids: List[int]
    char_map: List[Dict[str, int]]
    anchors: List[str] = field(default_factory=list)
    media_path: Optional[str] = None


@dataclass
class Claim:
    claim_id: str
    paper_id: str
    thread_id: str
    utterance_id: str
    claim_text: str
    claim_type: str
    checkability: str
    anchors_mentioned: List[str]
    entities: List[str]
    numbers: List[Dict[str, Any]]
    raw_span: str
    context_utterances: List[Dict[str, str]]


@dataclass
class EvidenceSpan:
    eobj_id: str
    span: Dict[str, int]
    source_cl_ids: List[int]
    bbox_union_norm: List[float]
    page_no: Optional[int]
    quote: Optional[str] = None


@dataclass
class ClaimCard:
    claim: Claim
    candidate_evidence: List[Dict[str, Any]] = field(default_factory=list)
    locator_suggestion: Optional[Dict[str, Any]] = None


@dataclass
class Annotation:
    claim_id: str
    label: str
    evidence_sets: List[List[EvidenceSpan]]
    annotator_ids_hash: Optional[str] = None
    adjudication_notes: Optional[str] = None
