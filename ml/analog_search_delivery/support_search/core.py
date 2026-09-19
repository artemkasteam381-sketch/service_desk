from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from collections import defaultdict
from typing import Callable, Mapping, Protocol, Sequence
from pathlib import Path
import hashlib
import json
import re
import time
import unicodedata
import numpy as np


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def text_key(text: str) -> str:
    return normalize(text).casefold()


@dataclass(frozen=True)
class Record:
    id: str
    original_text: str
    topic: str
    support_line: str
    result: str
    duplicate_group: str = ""
    source_url: str | None = None
    support_group: str = ""


class Encoder(Protocol):
    signature: str
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def canonical_groups(records: Sequence[Record], parser: Callable[[str], str]):
    """Union supplied near-duplicate groups AND exact raw/parsed duplicates, without labels.

    Run on the complete dataset BEFORE splitting or preparing classifier manifests.
    Near-duplicate discovery itself must be supplied/audited separately.
    """
    records = list(records)
    if len({r.id for r in records}) != len(records):
        raise ValueError("Record IDs must be unique")
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(records):
        for kind, key in [("group", row.duplicate_group), ("raw", text_key(row.original_text)),
                          ("clean", text_key(parser(row.original_text)))]:
            if not key: continue
            k = (kind, key)
            if k in seen:
                a, b = root(i), root(seen[k])
                parent[max(a, b)] = min(a, b)
            else: seen[k] = i
    members = defaultdict(list)
    for i, row in enumerate(records): members[root(i)].append(row.id)
    names = {i: "g_" + hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:20]
             for i, ids in members.items()}
    return [replace(row, duplicate_group=names[root(i)]) for i, row in enumerate(records)]


@dataclass
class TopicAdapter:
    predict_topics: Callable[[str], Mapping[str, float]]
    labels: Sequence[str]
    model_version: str
    # For evaluation, include training, early stopping, calibration and model-selection data.
    seen_groups: frozenset[str] | None = None
    dataset_version: str | None = None

    def __post_init__(self):
        self.labels = tuple(normalize(x) for x in self.labels)
        if len(self.labels) != 15 or len(set(self.labels)) != 15:
            raise ValueError("Expected exactly 15 unique class names")

    def signals(self, raw_text: str) -> dict[str, float]:
        # The classifier owns its preprocessing. Do not feed it the search parser's output.
        raw = self.predict_topics(raw_text)
        values = {normalize(str(k)): float(v) for k, v in raw.items()}
        if len(values) != len(raw) or set(values) != set(self.labels):
            raise ValueError("Classifier must return scores for all and only the 15 labels")
        scores = np.array([values[k] for k in self.labels])
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite classifier score")
        # Centered midrank, in [-.5, .5]; invariant to monotonic score transforms.
        # Ties get identical scores; a flat distribution contributes exactly zero.
        return {label: float(((scores < s).sum() + .5*((scores == s).sum()-1)) / (len(scores)-1) - .5)
                for label, s in zip(self.labels, scores)}


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be a finite 2-D matrix")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class SearchIndex:
    def __init__(self, records: Sequence[Record], encoder: Encoder,
                 parser: Callable[[str], str] = normalize, parser_version="normalize-v1",
                 dataset_version="unversioned", reranker=None, classifier: TopicAdapter | None = None):
        self.records = list(records)
        if len({r.id for r in self.records}) != len(self.records): raise ValueError("Duplicate IDs")
        if any(not r.duplicate_group for r in self.records):
            raise ValueError("Run canonical_groups before building the index")
        self.encoder, self.parser = encoder, parser
        self.parser_version, self.dataset_version = parser_version, dataset_version
        self.reranker, self.classifier = reranker, classifier
        self.texts = [normalize(parser(r.original_text)) for r in self.records]
        if not all(isinstance(t, str) for t in self.texts): raise TypeError("Parser must return str")
        self.groups = defaultdict(list)
        for i, row in enumerate(self.records): self.groups[row.duplicate_group].append(i)
        self.matrix = None

    def fingerprint(self):
        payload = {"records": [asdict(r) for r in self.records], "texts": self.texts,
                   "encoder": self.encoder.signature, "parser": self.parser_version,
                   "dataset": self.dataset_version}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def build(self, cache_dir: str | Path | None = None):
        if cache_dir and not getattr(self.encoder, "cacheable", True):
            raise ValueError("This encoder must rebuild its fitted vocabulary; omit cache_dir")
        key = self.fingerprint()
        file = Path(cache_dir) / (key + ".npz") if cache_dir else None
        if file and file.exists():
            with np.load(file, allow_pickle=False) as data:
                if str(data["fingerprint"]) != key: raise ValueError("Bad cache fingerprint")
                self.matrix = _unit(data["matrix"])
        else:
            if not self.records: raise ValueError("History is empty")
            self.matrix = _unit(self.encoder.encode_documents(self.texts))
            if file:
                file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(file, matrix=self.matrix, fingerprint=key)
        if self.matrix.shape[0] != len(self.records): raise ValueError("Embedding row count mismatch")
        return self

    def search(self, text: str, *, weight=0.0, **kwargs):
        return self.search_variants(text, weights=[weight], **kwargs)[float(weight)]

    def search_variants(self, text: str, *, weights=(0.0,), top_k=5, candidate_k=100,
                        rerank=True, exclude_ids=(), exclude_groups=(),
                        exclude_query_duplicates=False, strict_classifier=False):
        if self.matrix is None: raise RuntimeError("Build index first")
        weights = tuple(float(w) for w in weights)
        if not weights or any(not np.isfinite(w) or w < 0 or w > 1 for w in weights):
            raise ValueError("Weights must be in [0,1]")
        if not 1 <= top_k <= candidate_k: raise ValueError("Require 1 <= top_k <= candidate_k")
        started = time.perf_counter()
        clean = normalize(self.parser(text))
        warnings = []
        if not clean:
            return {w: {"results": [], "warnings": ["empty_after_parsing"], "timing_ms": {"total": 0.0}}
                    for w in weights}
        excluded = set(exclude_groups)
        excluded.update(r.duplicate_group for r in self.records if r.id in set(exclude_ids))
        if exclude_query_duplicates:
            excluded.update(r.duplicate_group for r, t in zip(self.records, self.texts)
                            if text_key(t) == text_key(clean) or text_key(r.original_text) == text_key(text))
        q = _unit(self.encoder.encode_queries([clean]))
        if q.shape != (1, self.matrix.shape[1]): raise ValueError("Query embedding dimension mismatch")
        sim = self.matrix @ q[0]
        # Keep the best textual representative of each group before selecting 100 groups.
        representatives = []
        seen = set()
        for i in np.argsort(-sim, kind="stable"):
            group = self.records[i].duplicate_group
            if group in excluded or group in seen or not self.texts[i]: continue
            seen.add(group)
            representatives.append(int(i))
            if len(representatives) >= candidate_k: break
        retrieval_ms = (time.perf_counter()-started)*1000
        ce_scores = {}
        rerank_start = time.perf_counter()
        if rerank and self.reranker is not None and representatives:
            scores = np.asarray(self.reranker.score(clean, [self.texts[i] for i in representatives]), dtype=float).reshape(-1)
            if len(scores) != len(representatives) or not np.isfinite(scores).all():
                raise ValueError("Invalid reranker scores")
            ce_scores = dict(zip(representatives, scores.tolist()))
            representatives.sort(key=lambda i: (-ce_scores[i], -float(sim[i]), self.records[i].id))
            warnings.extend(getattr(self.reranker, "warnings", []))
        elif rerank:
            warnings.append("reranker_not_configured")
        rerank_ms = (time.perf_counter()-rerank_start)*1000
        base_ms = (time.perf_counter()-started)*1000
        signals, classifier_ms, classifier_warnings = {}, 0.0, []
        if any(w > 0 for w in weights):
            t = time.perf_counter()
            try:
                if self.classifier is None: raise ValueError("Classifier not configured")
                signals = self.classifier.signals(text)
            except Exception as exc:
                if strict_classifier: raise
                classifier_warnings.append("classifier_unavailable:" + type(exc).__name__)
            classifier_ms = (time.perf_counter()-t)*1000
        results = {}
        for weight in weights:
            sort_start = time.perf_counter()
            candidates = []
            for rank, i in enumerate(representatives, 1):
                row = self.records[i]
                # Unknown labels are NEUTRAL, not a fictitious 16th class with score zero.
                # Conflicting group labels are averaged over unique labels, not cherry-picked.
                labels = sorted({normalize(self.records[j].topic) for j in self.groups[row.duplicate_group]})
                signal = float(np.mean([signals.get(label, 0.0) for label in labels])) if signals else 0.0
                score = 1/(60+rank) + weight*signal/61
                candidates.append({**asdict(row), "duplicate_variants": [asdict(self.records[j]) for j in self.groups[row.duplicate_group] if j != i],
                    "category_conflict": len(labels)>1, "cosine_similarity": float(sim[i]),
                    "cross_encoder_score": ce_scores.get(i), "base_rank": rank,
                    "topic_rank_signal": signal if weight else 0.0, "ranking_score": score})
            candidates.sort(key=lambda r: (-r["ranking_score"], r["base_rank"], r["id"]))
            total = base_ms + (classifier_ms if weight else 0) + (time.perf_counter()-sort_start)*1000
            results[weight] = {"results": candidates[:top_k], "candidate_ids": [self.records[i].id for i in representatives],
                "candidate_groups": [self.records[i].duplicate_group for i in representatives],
                "warnings": list(warnings) + (classifier_warnings if weight else []), "topic_weight": weight,
                "score_definition": "ranking heuristic, not probability of usefulness",
                "timing_ms": {"retrieval":retrieval_ms,"rerank":rerank_ms,
                              "classifier":classifier_ms if weight else 0.0,"total":total}}
        return results
