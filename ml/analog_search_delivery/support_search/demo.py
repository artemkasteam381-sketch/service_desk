"""Run from the parent directory: python -m support_search.demo --help."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
from .core import Record, SearchIndex, canonical_groups
from .models import E5Encoder, LocalTfidfEncoder, MiniLMReranker
from .parser import search_text
from .evaluation import collect_pool, score_pool, MemoryMonitor


def read_history(path, parser=search_text):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records = [Record(
        id=r.get("record_id") or r["Номер запроса"],
        original_text=r["Описание 2"], topic=r["Вид запроса"],
        support_line=r.get("Линия поддержки", r.get("Кем решен (группа)", "")),
        result=r.get("Результат работ", ""), duplicate_group=r.get("duplicate_group", ""),
        source_url=r.get("source_url") or None,
    ) for r in rows]
    records = canonical_groups(records, parser)
    version = hashlib.sha256(path.read_bytes()).hexdigest()
    return records, {r.id: row.get("split", "") for r, row in zip(records, rows)}, version


def main():
    p = argparse.ArgumentParser(description="Поиск аналогов; CSV из предыдущего notebook")
    p.add_argument("--data", required=True)
    p.add_argument("--backend", choices=["lexical", "e5"], default="lexical")
    p.add_argument("--reranker", action="store_true")
    p.add_argument("--query")
    p.add_argument("--out", required=True)
    p.add_argument("--review-split", choices=["validation", "test"])
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--weights", type=float, nargs="+", default=[0.0])
    p.add_argument("--classifier-factory", help="module:function returning TopicAdapter")
    p.add_argument("--parser", default="support_search.parser:search_text")
    p.add_argument("--parser-version", default="notebook-snapshot-v1")
    p.add_argument("--cache-dir")
    a = p.parse_args()
    if not a.query and not a.review_split: p.error("Set --query or --review-split")
    if a.query and a.review_split: p.error("Choose only one query mode")
    if a.limit < 1: p.error("--limit must be positive")
    if any(w > 0 for w in a.weights) and not a.classifier_factory:
        p.error("Nonzero weights require --classifier-factory")
    module, name = a.parser.split(":")
    parser = getattr(importlib.import_module(module), name)
    records, splits, version = read_history(a.data, parser)
    out = Path(a.out)
    if out.exists(): p.error("Output already exists; choose a new path to preserve results")
    queries = []
    if a.review_split:
        # Entire non-training groups excluded, including groups merged across previous splits.
        held_groups = {r.duplicate_group for r in records if splits[r.id] != "train"}
        seen = set()
        for r in records:
            if splits[r.id] == a.review_split and r.duplicate_group not in seen:
                queries.append(r); seen.add(r.duplicate_group)
        import random
        random.Random(42).shuffle(queries)
        queries = queries[:a.limit]
        records = [r for r in records if r.duplicate_group not in held_groups]
        if not queries: p.error("No queries found for the requested split")
    classifier = None
    with MemoryMonitor() as memory:
        if a.classifier_factory:
            module, name = a.classifier_factory.split(":")
            classifier = getattr(importlib.import_module(module), name)()
        encoder = E5Encoder() if a.backend == "e5" else LocalTfidfEncoder()
        reranker = MiniLMReranker() if a.reranker else None
        engine = SearchIndex(records, encoder, parser=parser, parser_version=a.parser_version,
                             dataset_version=version, reranker=reranker, classifier=classifier).build(a.cache_dir)
        if a.review_split:
            collect_pool(engine, queries, out, weights=a.weights, use_reranker=a.reranker)
            report = score_pool(out/"runs.json", out/"judgments.csv")
            (out/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            result = engine.search_variants(a.query, weights=a.weights, rerank=a.reranker, strict_classifier=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output":str(out), "dataset_version":version,
                      "index_records":len(records), "index_groups":len(engine.groups),
                      "query_count":len(queries) if a.review_split else 1,
                      "memory_including_build":memory.report}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
