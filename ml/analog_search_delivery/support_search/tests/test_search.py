import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from support_search.core import Record, SearchIndex, TopicAdapter, canonical_groups, normalize
from support_search.evaluation import collect_pool, score_pool, validate_holdout, paired_interval, merge_pools
from support_search.parser import search_text
from support_search.models import LocalTfidfEncoder, E5Encoder


LABELS = [f"category_{i}" for i in range(15)]


class Encoder:
    signature = "test-encoder-v1"
    def encode_documents(self, texts): return np.array([[1., .1*i] for i in range(len(texts))])
    def encode_queries(self, texts): return np.array([[1., 0.] for _ in texts])


class Reranker:
    signature = "test-reranker-v1"
    def score(self, query, docs): return np.arange(len(docs), dtype=float)


def history():
    return canonical_groups([Record(str(i), f"Issue {i}", LABELS[i] if i<5 else "outside",
                                    "(1 линия)", "" if i==0 else "historical result") for i in range(7)], normalize)


def engine(classifier=None, reranker=None):
    return SearchIndex(history(), Encoder(), classifier=classifier, reranker=reranker, dataset_version="v1").build()


def classifier(flat=False, seen=()):
    return TopicAdapter(lambda text: {k:0. if flat else i for i,k in enumerate(LABELS)},
                        LABELS, "test-model", frozenset(seen), "v1")


class SearchTests(unittest.TestCase):
    def test_groups_union_raw_clean_supplied_and_conflicts(self):
        rows = [Record("a","Hello","a","1","", "x"), Record("b"," hello ","b","2","", "y"),
                Record("c","Different","c","3","", "x")]
        result = canonical_groups(rows, normalize)
        self.assertEqual(len({r.duplicate_group for r in result}), 1)
        e = SearchIndex(result, Encoder()).build()
        hit = e.search("new", rerank=False)["results"][0]
        self.assertTrue(hit["category_conflict"])
        self.assertEqual(len(hit["duplicate_variants"]),2)
        self.assertEqual(hit["result"], "")
        self.assertEqual(e.search("new",exclude_ids=["b"],rerank=False)["results"],[])

    def test_soft_ranking_same_candidates_unknown_neutral(self):
        e = engine(classifier())
        outputs = e.search_variants("new",weights=[0,.2], top_k=7, rerank=False)
        self.assertEqual(outputs[0]["candidate_ids"],outputs[.2]["candidate_ids"])
        unknown = [x for x in outputs[.2]["results"] if x["topic"]=="outside"]
        self.assertEqual(len(unknown),2)
        self.assertTrue(all(x["topic_rank_signal"]==0 for x in unknown))
        self.assertNotEqual([x["id"] for x in outputs[0]["results"]], [x["id"] for x in outputs[.2]["results"]])

    def test_flat_classifier_preserves_order(self):
        e = engine(classifier(True))
        r = e.search_variants("new",weights=[0,1],rerank=False)
        self.assertEqual(r[0]["results"], r[1]["results"])

    def test_baseline_never_calls_classifier(self):
        c = classifier()
        c.predict_topics = lambda t: self.fail("Classifier was called")
        engine(c).search("new", weight=0, rerank=False)

    def test_classifier_failure_fallback_and_strict(self):
        c = classifier(); c.predict_topics = lambda t: {"wrong":1}
        e = engine(c)
        r = e.search_variants("new",weights=[0,.1],rerank=False)
        self.assertFalse(r[0]["warnings"])
        self.assertTrue(r[.1]["warnings"])
        self.assertEqual(r[0]["results"],r[.1]["results"])
        with self.assertRaises(ValueError): e.search("new",weight=.1, strict_classifier=True)

    def test_reranking_and_exact_exclusion(self):
        e = engine(reranker=Reranker())
        self.assertEqual(e.search("new")["results"][0]["id"], "6")
        self.assertNotIn("0",[r["id"] for r in e.search("Issue 0",exclude_query_duplicates=True)["results"]])

    def test_cache_and_versions(self):
        with tempfile.TemporaryDirectory() as d:
            a=engine(); a.build(d)
            b=engine(); b.build(d)
            np.testing.assert_allclose(a.matrix,b.matrix, atol=1e-7)
            before=a.fingerprint(); a.parser_version="new"
            self.assertNotEqual(before,a.fingerprint())
            with self.assertRaises(ValueError): SearchIndex(history(),LocalTfidfEncoder()).build(d)

    def test_parser_meaning(self):
        text="Здравствуйте! Тема письма: Ошибка входа Текст письма: Не работает ЕАС ОПС, ошибка 404, версия 2.1. С уважением, Иван test@example.com"
        clean=search_text(text)
        for fragment in ["не работает", "еас опс", "404", "2.1"]: self.assertIn(fragment,clean)
        self.assertNotIn("test@example.com",clean)
        self.assertNotIn("уважением",clean)

    def test_empty_and_invalid_scores(self):
        self.assertEqual(engine().search("  ")["results"],[])
        c=classifier(); c.predict_topics=lambda t:{k:float("nan") for k in LABELS}
        with self.assertRaises(ValueError): c.signals("new")
        with self.assertRaises(ValueError): engine().search("new",weight=-1)

    def test_holdout_gates(self):
        query=Record("q","held out","unknown","","", "held")
        validate_holdout(engine(classifier()),[query],with_classifier=True)
        with self.assertRaises(ValueError): validate_holdout(engine(classifier(seen=["held"])),[query],with_classifier=True)
        with self.assertRaises(ValueError): validate_holdout(engine(),[query],with_classifier=True)
        with self.assertRaises(ValueError): validate_holdout(engine(),[history()[0]],with_classifier=False)
        with self.assertRaises(ValueError): validate_holdout(engine(),[replace(query,original_text="Issue 0")],with_classifier=False)

    def test_judgment_completeness_and_metrics(self):
        queries=[Record("q","held out","","","","held")]
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"run"
            collect_pool(engine(classifier()),queries,out,weights=[0,.2],use_reranker=False)
            report=score_pool(out/"runs.json",out/"judgments.csv")
            self.assertEqual(report["summary"]["retrieval+topics=0"]["manually_judged_queries"],0)
            with (out/"judgments.csv").open(encoding="utf-8-sig",newline="") as f:
                reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
            for row in rows: row["relevance"]="2"
            with (out/"judgments.csv").open("w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
            report=score_pool(out/"runs.json",out/"judgments.csv")
            for method in report["summary"].values():
                self.assertEqual(method["precision_at_5"],1)
                self.assertEqual(method["pool_ndcg_at_5"],1)
            self.assertEqual(paired_interval(out/"runs.json",report,"retrieval+topics=0","retrieval+topics=0.2")["status"],"insufficient_judged_query_groups")
            with self.assertRaises(FileExistsError): collect_pool(engine(),queries,out)

    def test_empty_results_score_zero(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"run"
            collect_pool(engine(),[Record("q"," ","","","","held")],out,use_reranker=False)
            report=score_pool(out/"runs.json",out/"judgments.csv")
            s=report["summary"]["retrieval+topics=0"]
            self.assertEqual(s["empty_output_queries"],1)
            self.assertEqual(s["manually_judged_queries"],0)
            self.assertNotIn("precision_at_5",s)
            self.assertEqual(report["per_query"]["q"]["retrieval+topics=0"]["precision_at_5"],0)

    def test_merge_pools(self):
        with tempfile.TemporaryDirectory() as d:
            dirs=[Path(d)/str(i) for i in range(3)]
            q=[Record("q","held out","","","","held")]
            collect_pool(engine(),q,dirs[0],use_reranker=False)
            collect_pool(engine(reranker=Reranker()),q,dirs[1],use_reranker=True)
            merged=merge_pools(dirs[:2],dirs[2])
            self.assertEqual(len(merged["runs"]),2)

    def test_e5_prefixes_and_tail_chunks_without_model_download(self):
        class Tokenizer:
            def encode(self,text,**kwargs): return text.split()
            def decode(self,ids,**kwargs): return " ".join(ids)
        class Model:
            tokenizer=Tokenizer()
            def encode(self,texts,**kwargs):
                self.texts=texts
                return np.ones((len(texts),3),dtype=np.float32)
        adapter=E5Encoder.__new__(E5Encoder); adapter.model=Model(); adapter.batch_size=16
        adapter.encode_documents([" ".join(["token"]*700+["TAIL"])])
        self.assertTrue(all(x.startswith("passage: ") for x in adapter.model.texts))
        self.assertIn("TAIL",adapter.model.texts[-1])
        adapter.encode_queries(["query"])
        self.assertTrue(adapter.model.texts[0].startswith("query: "))


if __name__ == "__main__": unittest.main()
