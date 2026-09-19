"""Text-only BM25 / BGE-M3 / reciprocal-rank fusion, with reusable notebook caches."""
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import re
import time
import numpy as np
from .core import Record, canonical_groups, _unit
from .parser import parse_description
from .evaluation import MemoryMonitor, validate_holdout, write_csv, score_pool
from .models import E5Encoder

METHODS = ("bm25", "bge", "hybrid")


def core_text(text):
    return parse_description(text)["text_core"]


def load_source(directory):
    """Verify full text order + model revision against the original cache hash."""
    directory = Path(directory)
    path = directory / "results/appeals_parsed.csv"
    with path.open(encoding="utf-8-sig", newline="") as f: rows = list(csv.DictReader(f))
    manifests = json.loads((directory/"results/embedding_manifest.json").read_text())
    manifest = next(m for m in manifests if m["model"] == "BAAI/bge-m3")
    texts = [r["text_core"] for r in rows]
    if any(core_text(r["Описание 2"]) != t for r,t in zip(rows,texts)):
        raise ValueError("Parser differs from cached texts. Rebuild embeddings before comparison.")
    payload = {"model":manifest["model"], "revision":manifest["revision"], "texts":texts,
               "chunk":384, "overlap":64}
    key = hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest()[:20]
    if key != manifest["cache_key"]: raise ValueError("Embedding cache text/order/revision mismatch")
    with np.load(directory/f"results/embeddings_{key}.npz",allow_pickle=False) as data:
        queries, documents = _unit(data["query"]), _unit(data["passage"])
    if queries.shape != documents.shape or documents.shape[0] != len(rows):
        raise ValueError("Invalid cache dimensions")
    records = canonical_groups([Record(r["record_id"],r["Описание 2"],r["Вид запроса"],
        r["Кем решен (группа)"],r["Результат работ"],r["duplicate_group"]) for r in rows],core_text)
    version = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, records, texts, queries, documents, manifest, version


class BGEEncoder(E5Encoder):
    """Match notebook policy exactly: no prefix; normalized mean of 384-token windows."""
    def __init__(self, model_path, device="cpu"):
        super().__init__(str(model_path), device=device)
        self.signature = f"bge-m3-local:{model_path}:mean384-overlap64"
    def encode_documents(self,texts): return self._encode(texts, "")
    def encode_queries(self,texts): return self._encode(texts, "")


class AnalogSearch:
    def __init__(self, records, vectors, encoder=None):
        from rank_bm25 import BM25Okapi
        self.records=list(records)
        self.texts=[core_text(r.original_text) for r in records]
        self.parser=core_text
        self.vectors=_unit(vectors)
        if self.vectors.shape[0] != len(records): raise ValueError("Vector row mismatch")
        self.groups={}
        for i,r in enumerate(records): self.groups.setdefault(r.duplicate_group,[]).append(i)
        self.encoder=encoder
        self.bm25=BM25Okapi([self.tokenize(t) for t in self.texts])

    @staticmethod
    def tokenize(text): return re.findall(r"(?u)\b\w+\b",text.lower())

    def rank_groups(self,scores,excluded=(),limit=100,positive=False):
        ranked=[]; seen=set(excluded)
        for i in np.argsort(-scores,kind="stable"):
            group=self.records[i].duplicate_group
            if group in seen or not self.texts[i]: continue
            if positive and scores[i]<=0: continue
            seen.add(group); ranked.append((int(i),float(scores[i])))
            if len(ranked)>=limit: break
        return ranked

    def hits(self,ranked):
        out=[]
        for i,score in ranked[:5]:
            r=self.records[i]
            out.append({**asdict(r), "ranking_score":score,
                "category_conflict":len({self.records[j].topic for j in self.groups[r.duplicate_group]})>1,
                "duplicate_variants":[asdict(self.records[j]) for j in self.groups[r.duplicate_group] if j!=i]})
        return out

    def search(self,text,method="hybrid",query_vector=None,exclude_groups=()):
        if method not in METHODS: raise ValueError("Unknown method")
        started=time.perf_counter(); clean=self.parser(text)
        if not clean:
            return {"results":[],"warnings":["empty_after_parsing"],"timing_ms":{"total":0},"method":method}
        encoded_ms=0.
        if method in ("bge","hybrid"):
            t=time.perf_counter()
            if query_vector is None:
                if self.encoder is None: raise ValueError("New text needs a loaded BGE encoder")
                query_vector=self.encoder.encode_queries([clean])[0]
                encoded_ms=(time.perf_counter()-t)*1000
            q=_unit(np.asarray(query_vector).reshape(1,-1))[0]
            if q.shape[0]!=self.vectors.shape[1]: raise ValueError("Query dimension mismatch")
            dense=self.rank_groups(self.vectors@q,exclude_groups)
        if method in ("bm25","hybrid"):
            lexical=self.rank_groups(self.bm25.get_scores(self.tokenize(clean)),exclude_groups,positive=True)
        if method=="bge": ranked=dense
        elif method=="bm25": ranked=lexical
        else:
            # Fuse GROUP ranks, so duplicates cannot multiply their group's influence.
            combined={}; representatives={}
            for ranking in (dense,lexical):
                for rank,(i,_) in enumerate(ranking,1):
                    g=self.records[i].duplicate_group
                    combined[g]=combined.get(g,0)+1/(60+rank)
                    representatives.setdefault(g,i)  # best dense representative when available
            ranked=[(representatives[g],s) for g,s in sorted(combined.items(),key=lambda x:(-x[1],x[0]))]
        return {"results":self.hits(ranked),"method":method,"warnings":[],
            "score_definition":"ranking score, not probability of usefulness",
            "timing_ms":{"total":(time.perf_counter()-started)*1000,"query_encoding":encoded_ms}}


def prepare_review(source,out,limit=40):
    out=Path(out)
    if out.exists(): raise FileExistsError("Choose a new output directory to preserve judgments")
    with MemoryMonitor() as memory:
        rows,records,texts,qvec,dvec,manifest,version=load_source(source)
        excluded={r.duplicate_group for r,row in zip(records,rows) if row["split"]!="train"}
        ids=[i for i,r in enumerate(records) if r.duplicate_group not in excluded and texts[i]]
        engine=AnalogSearch([records[i] for i in ids],dvec[ids])
        query_ids=[]; seen=set()
        for i,(r,row) in enumerate(zip(records,rows)):
            if row["split"]=="validation" and r.duplicate_group not in seen:
                query_ids.append(i); seen.add(r.duplicate_group)
        random.Random(42).shuffle(query_ids); query_ids=query_ids[:limit]
        validate_holdout(engine,[records[i] for i in query_ids],with_classifier=False)
        runs=[]; pairs={}
        # Warm-up; cached vectors explicitly exclude query encoding from these timings.
        for method in METHODS:
            engine.search(records[query_ids[0]].original_text,method,qvec[query_ids[0]],excluded)
        for i in query_ids:
            q=records[i]
            for method in METHODS:
                response=engine.search(q.original_text,method,qvec[i],excluded)
                runs.append({"query_id":q.id,"query_group":q.duplicate_group,"method":method,
                             "latency_ms":response["timing_ms"]["total"],"response":response})
                for hit in response["results"]:
                    # Same representative for annotation across methods: include ALL variants,
                    # including the current representative, to judge the whole duplicate group.
                    group=engine.groups[hit["duplicate_group"]]
                    first=engine.records[group[0]]
                    pairs[(q.id,hit["duplicate_group"])]=dict(query_id=q.id,candidate_group=hit["duplicate_group"],
                        candidate_id=first.id,query_text=q.original_text,candidate_text=first.original_text,
                        historical_result=first.result,
                        other_records_json=json.dumps([asdict(engine.records[j]) for j in group[1:]],ensure_ascii=False),
                        relevance="",reason="")
    payload={"dataset_version":version,"encoder":manifest,"parser":"notebook-text_core-v1",
        "history_records":len(ids),"history_groups":len(engine.groups),
        "queries":[dict(query_id=records[i].id,query_group=records[i].duplicate_group,query_text=records[i].original_text) for i in query_ids],
        "runs":runs,"resources":memory.report,
        "latency_note":"Cached-query ranking only. EXCLUDES encoder loading and query embedding; not full service latency.",
        "evaluation_note":"Existing validation reused; exploratory, not a new independent test. No classifier or reranker."}
    out.mkdir(parents=True)
    pool=list(pairs.values());random.Random(42).shuffle(pool)
    fields=["query_id","candidate_group","candidate_id","query_text","candidate_text","historical_result","other_records_json","relevance","reason"]
    write_csv(out/"judgments.csv",pool,fields)
    (out/"runs.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    scores=score_pool(out/"runs.json",out/"judgments.csv")
    scores["latency_note"]=payload["latency_note"]
    (out/"metrics.json").write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding="utf-8")
    from .review_page import make_review
    make_review(pool,out/"review.html")
    return {"queries":len(query_ids),"pairs":len(pool),"history_records":len(ids),"output":str(out),"resources":memory.report}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,help="Folder containing results and model_cache")
    p.add_argument("--out",required=True)
    p.add_argument("--query")
    p.add_argument("--method",choices=METHODS,default="hybrid")
    p.add_argument("--device",default="cpu")
    p.add_argument("--limit",type=int,default=40)
    a=p.parse_args()
    if a.limit<1: p.error("limit must be positive")
    if not a.query:
        print(json.dumps(prepare_review(a.source,a.out,a.limit),ensure_ascii=False,indent=2));return
    out=Path(a.out)
    if out.exists(): raise FileExistsError(out)
    with MemoryMonitor() as memory:
        rows,records,texts,_,vectors,manifest,version=load_source(a.source)
        encoder=None; t=time.perf_counter()
        if a.method!="bm25":
            import torch
            torch.set_num_threads(2)
            model=Path(a.source)/"model_cache/hub/models--BAAI--bge-m3/snapshots"/manifest["revision"]
            if not model.is_dir(): raise FileNotFoundError(f"Local model unavailable: {model}")
            encoder=BGEEncoder(model,a.device)
        load_ms=(time.perf_counter()-t)*1000
        engine=AnalogSearch(records,vectors,encoder)
        response=engine.search(a.query,a.method)
    response.update(dataset_version=version,encoder=manifest,model_load_ms=load_ms,memory=memory.report)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(response,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out),"timing_ms":response["timing_ms"],"model_load_ms":load_ms,"memory":memory.report},indent=2))


if __name__=="__main__": main()
