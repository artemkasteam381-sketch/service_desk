"""Offline support-line classification and BGE → MiniLM analog search."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from parser import parse_description, normalize_metadata, PREPROCESS_VERSION
from models import BGEEncoder, MiniLMReranker

ROOT = Path(__file__).resolve().parent
FIELDS = ['Услуга', 'Тип запроса', 'Компонент услуги 1 уровня', 'Пользователь']


class SupportService:
    def __init__(self, root=ROOT):
        import torch
        torch.set_num_threads(2)
        root = Path(root)
        self.bundle = joblib.load(root/'data/classifier.joblib')
        if self.bundle.get('preprocess_version') != PREPROCESS_VERSION:
            raise ValueError('Предобработка и веса несовместимы: пересоберите весь пакет.')
        self.records = json.loads((root/'data/records.json').read_text(encoding='utf-8'))
        self.vectors = np.load(root/'data/vectors.npy', allow_pickle=False)
        if len(self.records) != len(self.vectors) or not np.isfinite(self.vectors).all():
            raise ValueError('Invalid search data')
        self.encoder = BGEEncoder(str(root/'models/bge-m3'))
        self.reranker = MiniLMReranker(str(root/'models/reranker'))
        self.options = {f: sorted({r['metadata'][f] for r in self.records if r['metadata'][f] != '__MISSING__'}) for f in FIELDS}

    @lru_cache(maxsize=64)
    def encode(self, text):
        # Match the original BGE pipeline: no E5 prefixes, mean windows 384/64.
        return self.encoder.encode([text])[0]

    def query(self, text, metadata=None):
        if not isinstance(text, str) or len(text) > 20000:
            raise ValueError('Text must be a string, at most 20000 characters')
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('Metadata must be an object')
        parsed = parse_description(text)
        core, subject = parsed['text_core'], parsed['text_with_subject']
        result = {'recommended_line': None, 'method': 'classifier', 'scores': {},
                  'search_method': 'bge_top50_minilm', 'analogs': [], 'notes': [], 'known_record_ids': []}
        if not core:
            result['notes'].append('Нет текста проблемы после очистки.')
            return result
        metadata = metadata or {}
        values = np.array([[normalize_metadata(metadata.get(f, '')) for f in FIELDS]], dtype=object)
        vector = self.encode(core)
        if (values != '__MISSING__').any():
            bundle = self.bundle['base']
            x = sparse.hstack([sparse.csr_matrix(vector.reshape(1, -1)), bundle['onehot'].transform(values)], format='csr')
        else:
            bundle = self.bundle['text']
            x = vector.reshape(1, -1)
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            result['notes'].append('Поля не заполнены: линия определена по тексту.')
        scores = bundle['model'].predict_proba(x)[0]
        result['scores'] = dict(zip(bundle['model'].classes_.tolist(), scores.tolist()))
        result['recommended_line'] = str(bundle['model'].classes_[np.argmax(scores)])
        matches = [r for r in self.records if r['group'] == parsed['group'] or r['core'] == core or (subject and r['text'] == subject)]
        excluded = {r['group'] for r in matches}
        result['known_record_ids'] = [r['id'] for r in matches]
        if matches:
            result['notes'].append('Текст встречается в загруженной истории. Совпадающие группы исключены из аналогов.')
        seen, candidates = set(excluded), []
        for i in np.argsort(-(self.vectors @ vector), kind='stable'):
            record = self.records[i]
            if not record['core'] or not record.get('model_eligible',True) or record['group'] in seen:
                continue
            seen.add(record['group'])
            candidates.append(int(i))
            if len(candidates) == 50:
                break
        if candidates:
            ranks = np.asarray(self.reranker.score(core, [self.records[i]['core'] for i in candidates]))
            if ranks.shape != (len(candidates),) or not np.isfinite(ranks).all():
                raise ValueError('Invalid reranker scores')
            for j in np.argsort(-ranks, kind='stable')[:5]:
                r = self.records[candidates[j]]
                result['analogs'].append({k: r[k] for k in ['id', 'text', 'line', 'result', 'category']} | {'score': float(ranks[j]), 'source_url': '/api/records/' + str(r['row_index'])})
        result['notes'].extend(getattr(self.reranker, 'warnings', []))
        if self.bundle.get('line_counts', {}).get(result['recommended_line'], 0) < 5:
            result['notes'].append('Для выбранной линии крайне мало обучающих примеров; нужен ручной разбор.')
        result['preprocess_version'] = PREPROCESS_VERSION
        result['notes'].append('Оценки моделей не являются проверенными вероятностями решения.')
        return result


def serve(port):
    service = SupportService()
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, mime='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode())

        def do_GET(self):
            if self.path == '/':
                self.send(200, (ROOT/'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/api/options':
                self.send(200, service.options)
            elif self.path.startswith('/api/records/'):
                try:
                    index = int(self.path.rsplit('/', 1)[-1])
                    if not 0 <= index < len(service.records): raise ValueError('Invalid record')
                    self.send(200, service.records[index])
                except ValueError:
                    self.send(404, {'error': 'Not found'})
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/api/predict':
                self.send(404, {'error': 'Not found'})
                return
            if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                self.send(403, {'error': 'Unexpected origin'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 100000:
                    raise ValueError('Request too large or empty')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Expected JSON object')
                self.send(200, service.query(data.get('text', ''), data.get('metadata')))
            except (ValueError, TypeError, KeyError) as error:
                self.send(400, {'error': str(error)})

        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f'Ready: http://127.0.0.1:{port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    serve(parser.parse_args().port)
