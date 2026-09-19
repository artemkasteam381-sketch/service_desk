"""Leakage gates, blinded judgment pools, paired relevance and resource measurements."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import os
import resource
import sys
import threading
import time
import numpy as np
from .core import text_key


class MemoryMonitor:
    """Sample host RSS incl. native allocations. CUDA allocation reported separately.
    Prefer one method per fresh process for attributable peaks; MPS GPU memory is unavailable.
    """
    def __enter__(self):
        import psutil
        self.proc = psutil.Process(os.getpid())
        self.baseline = self.peak = self.proc.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.01): self.peak = max(self.peak, self.proc.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.stop.set(); self.thread.join()
        unit = 1 if sys.platform == "darwin" else 1024
        self.report = {"rss_baseline_mb":self.baseline/2**20,"sampled_peak_rss_mb":self.peak/2**20,
                       "process_lifetime_peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit/2**20,
                       "gpu_memory":"not_measured"}
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            self.report["cuda_process_peak_allocated_mb"] = torch.cuda.max_memory_allocated()/2**20
            self.report["gpu_memory"] = "CUDA allocation only, not total device memory"


def validate_holdout(engine, queries, *, with_classifier):
    query_groups = {q.duplicate_group for q in queries}
    if not all(query_groups): raise ValueError("Queries need audited duplicate groups")
    if len({q.id for q in queries}) != len(queries): raise ValueError("Query IDs must be unique")
    if query_groups & set(engine.groups):
        raise ValueError("Held-out query duplicate groups are still in the history index")
    keys = {text_key(t) for t in engine.texts if t}
    raw_keys = {text_key(r.original_text) for r in engine.records if r.original_text}
    if any(text_key(q.original_text) in raw_keys for q in queries):
        raise ValueError("Raw query duplicate detected despite supplied group IDs")
    if any(text_key(engine.parser(q.original_text)) in keys for q in queries):
        raise ValueError("Exact query duplicate detected despite supplied group IDs")
    if with_classifier:
        c = engine.classifier
        if c is None or c.seen_groups is None or c.dataset_version != engine.dataset_version:
            raise ValueError("Joint evaluation requires versioned classifier seen_groups provenance")
        if query_groups & set(c.seen_groups):
            raise ValueError("Classifier trained/tuned on held-out query groups")


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def collect_pool(engine, queries, out_dir, weights=(0.0,), use_reranker=True):
    """One retrieval/cross-encoder pass per query shared by ALL classifier-weight variants."""
    weights = tuple(weights)
    validate_holdout(engine, queries, with_classifier=any(w>0 for w in weights))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"judgments.csv").exists():
        raise FileExistsError("Use a new run directory: do not overwrite human judgments")
    rows, pool, query_data = [], {}, {}
    # Warm-up deliberately separate. Report cold/load/index times outside this function.
    if queries:
        engine.search_variants(queries[0].original_text, weights=weights, rerank=use_reranker,
                               exclude_groups={q.duplicate_group for q in queries}, strict_classifier=True)
    with MemoryMonitor() as monitor:
        for q in queries:
            outputs = engine.search_variants(q.original_text, weights=weights, rerank=use_reranker,
                exclude_groups={x.duplicate_group for x in queries}, exclude_query_duplicates=True, strict_classifier=True)
            query_data[q.id] = {"query_id":q.id,"query_group":q.duplicate_group,"query_text":q.original_text}
            for weight, response in outputs.items():
                method = ("reranked" if use_reranker and engine.reranker else "retrieval") + f"+topics={weight:g}"
                rows.append({"query_id":q.id,"query_group":q.duplicate_group,"method":method,
                             "latency_ms":response["timing_ms"]["total"],"response":response})
                for hit in response["results"]:
                    key = (q.id, hit["duplicate_group"])
                    pool[key] = {"query_id":q.id,"candidate_group":hit["duplicate_group"],
                        "candidate_id":hit["id"],"query_text":q.original_text,
                        "candidate_text":hit["original_text"],"historical_result":hit["result"],
                        "other_records_json":json.dumps(hit["duplicate_variants"],ensure_ascii=False),
                        "relevance":"","reason":""}
    # Stable randomized ordering hides method/rank; do not show true query category.
    pool_rows = list(pool.values())
    np.random.default_rng(42).shuffle(pool_rows)
    write_csv(out/"judgments.csv",pool_rows,["query_id","candidate_group","candidate_id","query_text",
        "candidate_text","historical_result","other_records_json","relevance","reason"])
    payload = {"dataset_version":engine.dataset_version,"index_fingerprint":engine.fingerprint(),
               "encoder":engine.encoder.signature,"parser":engine.parser_version,
               "reranker":getattr(engine.reranker,"signature",None),
               "classifier":engine.classifier.model_version if engine.classifier else None,
               "queries":list(query_data.values()),"runs":rows,"resources":monitor.report,
               "latency_note":"warm component time; shared retrieval and classifier calls, sequential batch=1; not service p95 under load"}
    (out/"runs.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return payload


def score_pool(run_path, judgments_path):
    """Only fully judged queries are compared; blanks NEVER become irrelevant labels."""
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    judgments = {}
    with Path(judgments_path).open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            key = (row["query_id"],row["candidate_group"])
            if key in judgments: raise ValueError("Duplicate judgment pair")
            grade = None if row["relevance"].strip()=="" else float(row["relevance"])
            if grade not in (None,0.,1.,2.): raise ValueError("relevance must be 0/1/2")
            judgments[key] = grade
    by_query, empty_queries = {}, set()
    for q in run["queries"]:
        items = [r for r in run["runs"] if r["query_id"]==q["query_id"]]
        required = {h["duplicate_group"] for r in items for h in r["response"]["results"]}
        if any(judgments.get((q["query_id"],g)) is None for g in required): continue
        if not required: empty_queries.add(q["query_id"])
        pool = [judgments[(q["query_id"],g)] for g in required]
        ideal = sorted(pool,reverse=True)[:5]
        idcg = sum((2**g-1)/np.log2(i+2) for i,g in enumerate(ideal))
        by_query[q["query_id"]] = {}
        for r in items:
            grades = [judgments[(q["query_id"],h["duplicate_group"])] for h in r["response"]["results"][:5]]
            dcg = sum((2**g-1)/np.log2(i+2) for i,g in enumerate(grades))
            by_query[q["query_id"]][r["method"]] = {
                "precision_at_5":sum(g>=1 for g in grades)/5,
                "useful_at_5":float(any(g==2 for g in grades)),
                "pool_ndcg_at_5":dcg/idcg if idcg else 0.0,
                "mrr_at_5":next((1/(i+1) for i,g in enumerate(grades) if g>=1),0.0)}
    methods = sorted({r["method"] for r in run["runs"]})
    summary = {}
    for m in methods:
        values = [v[m] for v in by_query.values()]
        latency = [r["latency_ms"] for r in run["runs"] if r["method"]==m]
        human_count = len(by_query) - len(empty_queries)
        summary[m] = {"evaluated_queries":len(values),"manually_judged_queries":human_count,
                      "empty_output_queries":len(empty_queries),"total_queries":len(run["queries"]),
                      "warm_latency_p50_ms":float(np.median(latency)) if latency else None,
                      "warm_latency_p95_ms":float(np.quantile(latency,.95)) if latency else None}
        if values and human_count:
            summary[m].update({k:float(np.mean([v[k] for v in values])) for k in values[0]})
    return {"summary":summary,"per_query":by_query,"resources":run["resources"],
            "limitations":"nDCG is relative to the judged pool, not full-corpus recall; unjudged queries excluded; empty outputs score zero, but aggregate quality requires at least one manually judged query"}


def paired_interval(run_path, scored, baseline, challenger, seed=42, repetitions=1000):
    """Bootstrap paired nDCG changes by QUERY DUPLICATE GROUP, not independent rows."""
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    group_for = {q["query_id"]:q["query_group"] for q in run["queries"]}
    diffs = {}
    for qid, methods in scored["per_query"].items():
        delta = methods[challenger]["pool_ndcg_at_5"]-methods[baseline]["pool_ndcg_at_5"]
        diffs.setdefault(group_for[qid],[]).append(delta)
    if len(diffs)<2: return {"status":"insufficient_judged_query_groups"}
    v = np.array([np.mean(x) for x in diffs.values()])
    rng = np.random.default_rng(seed)
    means = [rng.choice(v,len(v),replace=True).mean() for _ in range(repetitions)]
    return {"mean_group_delta_ndcg":float(v.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),
            "groups":len(v),"note":"fixed systems; use independent test after validation weight selection"}


def merge_pools(run_dirs, out_dir):
    """Pool retrieval and reranked runs BEFORE annotation, on identical held-out queries."""
    out = Path(out_dir)
    if out.exists(): raise FileExistsError(out)
    payloads, judgments, seen_runs = [], {}, set()
    for directory in map(Path, run_dirs):
        payload = json.loads((directory/"runs.json").read_text(encoding="utf-8"))
        if payloads:
            for key in ("dataset_version", "index_fingerprint", "queries"):
                if payload[key] != payloads[0][key]:
                    raise ValueError("Pools must share dataset, index and ordered queries")
        for row in payload["runs"]:
            key = (row["query_id"], row["method"])
            if key in seen_runs: raise ValueError("Duplicate method/query; merge distinct methods only")
            seen_runs.add(key)
        payloads.append(payload)
        with (directory/"judgments.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["relevance"].strip(): raise ValueError("Merge before annotation; existing labels found")
                judgments[(row["query_id"], row["candidate_group"])] = row
    if not payloads: raise ValueError("No pools")
    result = dict(payloads[0])
    result["runs"] = [row for p in payloads for row in p["runs"]]
    result["resources"] = {"separate_runs": [p["resources"] for p in payloads]}
    result["source_runs"] = [{k:v for k,v in p.items() if k not in ("runs", "queries")} for p in payloads]
    rows = list(judgments.values())
    np.random.default_rng(42).shuffle(rows)
    out.mkdir(parents=True)
    write_csv(out/"judgments.csv", rows, ["query_id","candidate_group","candidate_id","query_text",
        "candidate_text","historical_result","other_records_json","relevance","reason"])
    (out/"runs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
