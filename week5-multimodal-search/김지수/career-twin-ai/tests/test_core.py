import tempfile
import unittest
from pathlib import Path
from career_twin.ingest import extract
from career_twin.storage import Store
from career_twin.retrieval import search
from career_twin.services import validated_claims, experience_answer, judge_requirement, coverage, recommend_jobs

class CoreTests(unittest.TestCase):
    def test_separation_persistence_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            content = '# 프로젝트\nPython과 FAISS로 벡터 검색을 구현했다.'.encode()
            chunks, warnings = extract('portfolio.md', content)
            doc_id = store.add('portfolio.md', content, 'personal', chunks, warnings)
            self.assertEqual(store.add('portfolio.md', content, 'personal', chunks), doc_id)
            store.add('paper.md', b'FAISS paper', 'paper',
                      [{'text': 'Python 논문의 이론', 'section': '본문'}])
            self.assertEqual(len(store.documents()), 2)
            hits = search('FAISS', store.chunks('personal'))
            self.assertEqual(hits[0]['document_id'], doc_id)
            store.save_vectors('test', [(hits[0]['id'], [1, 0])])
            reopened = Store(Path(directory))
            self.assertTrue(reopened.get_vectors('test'))
            raw = Path(reopened.documents('personal')[0]['path'])
            reopened.delete(doc_id)
            self.assertFalse(raw.exists())
            self.assertEqual(reopened.get_vectors('test'), {})
            self.assertEqual(reopened.chunks('personal'), [])
            self.assertTrue(reopened.chunks('paper'))

    def test_quotes_and_sources(self):
        hits = [{'id':'one','text':'Python으로 FAISS 검색을 구현했다.', 'name':'보고서.md','section':'개발','kind':'personal'}]
        payload = {'claims':[
            {'text':'개발 경험', 'chunk_id':'one','quote':'Python으로 FAISS 검색'},
            {'text':'허위', 'chunk_id':'missing','quote':'Python으로 FAISS 검색'},
            {'text':'허위 수치', 'chunk_id':'one','quote':'정확도 99%'}]}
        accepted, rejected = validated_claims(payload, hits)
        self.assertEqual((len(accepted), rejected), (1,2))
        with self.assertRaises(ValueError):
            experience_answer('경험?', [{**hits[0], 'kind':'synthetic_job'}])

    def test_matching_rejects_unverified_evidence(self):
        class FakeAI:
            def chat(self,*args):
                return {'status':'직접 근거','chunk_id':'other','quote':'없는 원문', 'reason':'수행함','next_action':'배포'}
        hit = {'id':'one','kind':'personal','name':'기록','text':'Airflow를 학습할 예정이다.'}
        result = judge_requirement({'text':'Airflow 운영','category':'필수'},[hit],FakeAI())
        self.assertEqual(result['status'],'근거 없음')
        pending = judge_requirement({'text':'Airflow 운영','category':'필수'},[hit])
        self.assertIsNone(coverage([pending]))

    def test_public_evaluations_do_not_leak_into_search(self):
        rows = [{'id':'a','jobpost':'Python 개발','evaluation':'MAGIC_SECRET','resume':'MAGIC_SECRET'},
                {'id':'b','jobpost':'SQL 분석'}]
        self.assertEqual(recommend_jobs('MAGIC_SECRET',rows),[])
        self.assertEqual(recommend_jobs('Python',rows)[0]['id'],'a')

    def test_empty_and_invalid_upload(self):
        with self.assertRaises(ValueError): extract('empty.txt',b' ')
        with self.assertRaises(ValueError): extract('bad.exe',b'data')
        with self.assertRaises(ValueError): extract('bad.txt',b'\xff')

if __name__ == '__main__': unittest.main()
