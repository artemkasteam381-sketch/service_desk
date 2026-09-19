import tempfile
import unittest
from pathlib import Path
import numpy as np
from support_search.core import Record, canonical_groups
from support_search.focused import AnalogSearch,core_text
from support_search.review_page import make_review


class FocusedTests(unittest.TestCase):
    def setUp(self):
        records=canonical_groups([Record('1','Не приходит код sms','A','1',''),
            Record('2','Не приходит код sms','B','2','Решение предоставлено'),
            Record('3','Не печатается этикетка','C','1','Уточнение не предоставлено'),
            Record('4','Ошибка оплаты картой','D','2','')],core_text)
        self.e=AnalogSearch(records,np.array([[1,0],[1,0],[0,1],[.5,.5]]))
    def test_unique_groups_all_methods(self):
        for method in ['bm25','bge','hybrid']:
            r=self.e.search('Не приходит код sms',method,[1,0])['results']
            self.assertEqual(len({x['duplicate_group'] for x in r}),len(r))
            first=next(x for x in r if x['id']=='1')
            self.assertTrue(first['category_conflict'])
            self.assertEqual(first['result'],'')
            self.assertEqual(first['duplicate_variants'][0]['id'],'2')
    def test_exclusions_and_empty(self):
        g=self.e.records[0].duplicate_group
        r=self.e.search('Не приходит код sms','hybrid',[1,0],[g])
        self.assertTrue(all(x['duplicate_group']!=g for x in r['results']))
        self.assertEqual(self.e.search('','bm25')['results'],[])
        self.assertEqual(self.e.search('abcdefxyz','bm25')['results'],[])
    def test_new_text_needs_encoder(self):
        with self.assertRaises(ValueError):self.e.search('текст','bge')
    def test_html_escapes_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'review.html'; make_review([{'query_text':'</script><script>alert(1)</script>'}],p)
            self.assertNotIn('</script><script>alert',p.read_text())


if __name__=='__main__':unittest.main()
