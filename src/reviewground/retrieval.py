from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import simple_tokenize


class BM25Index:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_tokens = [simple_tokenize(d) for d in docs]
        self.doc_freqs = []
        self.df = Counter()
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        total_len = 0
        for tokens in self.doc_tokens:
            total_len += len(tokens)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for term in freqs:
                self.df[term] += 1
        self.avgdl = total_len / max(len(self.doc_tokens), 1)

    def _idf(self, term: str) -> float:
        n_docs = len(self.doc_tokens)
        df = self.df.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> List[float]:
        tokens = simple_tokenize(query)
        scores = [0.0] * len(self.doc_tokens)
        for term in tokens:
            idf = self._idf(term)
            for i, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(term, 0)
                if freq == 0:
                    continue
                doc_len = len(self.doc_tokens[i])
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                scores[i] += idf * (freq * (self.k1 + 1) / denom)
        return scores


class TfidfIndex:
    def __init__(self, docs: List[str]):
        self.vectorizer = TfidfVectorizer(min_df=1)
        self.matrix = self.vectorizer.fit_transform(docs)

    def score(self, query: str) -> List[float]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        return sims.tolist()


def rrf_fuse(rankings: Dict[str, List[int]], k: int = 60) -> List[int]:
    scores = defaultdict(float)
    for ranks in rankings.values():
        for rank, idx in enumerate(ranks):
            scores[idx] += 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def topk(scores: List[float], k: int) -> List[int]:
    return [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]]


def build_indexes(eobj_texts: List[str]) -> Tuple[BM25Index, TfidfIndex]:
    return BM25Index(eobj_texts), TfidfIndex(eobj_texts)


def retrieve_hybrid(
    claim_text: str,
    eobj_texts: List[str],
    topk_sparse: int,
    topk_dense: int,
    topk_fused: int,
) -> Tuple[List[int], Dict[str, List[int]]]:
    bm25 = BM25Index(eobj_texts)
    tfidf = TfidfIndex(eobj_texts)
    sparse_scores = bm25.score(claim_text)
    dense_scores = tfidf.score(claim_text)
    sparse_rank = topk(sparse_scores, topk_sparse)
    dense_rank = topk(dense_scores, topk_dense)
    fused_rank = rrf_fuse({"sparse": sparse_rank, "dense": dense_rank})
    return fused_rank[:topk_fused], {"sparse": sparse_rank, "dense": dense_rank}
