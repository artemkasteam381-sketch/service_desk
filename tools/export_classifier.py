"""Экспорт финального стекинга из RESULTS в переносимую папку сервиса.

Работает с директорией результатов или ZIP ноутбука. Используйте свой экспорт.
Пример: python tools/export_classifier.py metadata_results.zip --output models/classification
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def export_classifier(results, output, experiment='essence_first_256', kind='catboost_confidence'):
    results, output = Path(results), Path(output)
    allowed = {'catboost_confidence', 'catboost', 'logreg'}
    if kind not in allowed or Path(experiment).name != experiment:
        raise ValueError('Неизвестный вариант модели или неверное имя эксперимента')
    archive = zipfile.ZipFile(results) if results.is_file() else None
    def read(name):
        return archive.read(name) if archive else (results / name).read_bytes()
    try:
        payloads = {
            'stacking_bundle.joblib': read(f'{experiment}/stacking_bundle_{kind}.joblib'),
            'inference.py': read('inference.py'),
            'stacking_features.py': read('stacking_features.py'),
            'stacking_inference.py': read('stacking_inference.py'),
            'requirements_versions.txt': read('requirements_versions.txt'),
        }
        source_manifest = json.loads(read('manifest.json'))
        stack_manifest = json.loads(read(f'{experiment}/stacking_manifest.json'))
        runtime = payloads['stacking_inference.py'].decode('utf-8-sig')
        # Частный Python-пакет позволяет не изменять глобальный sys.path сервера.
        for old, new in [('from inference import ', 'from .inference import '),
                         ('from stacking_features import ', 'from .stacking_features import ')]:
            if old not in runtime and new not in runtime:
                raise ValueError('Неожиданный формат экспорта stacking_inference.py')
            runtime = runtime.replace(old, new)
        payloads['stacking_inference.py'] = runtime.encode('utf-8')
        payloads['__init__.py'] = b'"""Runtime exported together with the trained classifier."""\n'
        manifest = dict(
            format_version=1, model_file='stacking_bundle.joblib', experiment=experiment,
            meta_kind=kind, stack_feature_version=stack_manifest['stack_feature_version'],
            source_sha256=source_manifest['source_sha256'],
            rubert_model=stack_manifest['rubert_model'],
            rubert_revision=stack_manifest.get('rubert_revision', 'main'),
            rubert_text_mode=stack_manifest['rubert_text_mode'],
            rubert_max_length=stack_manifest['rubert_max_length'],
            tfidf_text_mode=stack_manifest['text_mode'],
            files_sha256={name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
        )
        # Все исходные файлы прочитаны до начала записи; ошибки выбора модели не портят экспорт.
        output.mkdir(parents=True, exist_ok=True)
        for name, content in payloads.items():
            (output / name).write_bytes(content)
        (output / 'model_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return output
    finally:
        if archive:
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/classification'))
    parser.add_argument('--experiment', default='essence_first_256')
    parser.add_argument('--kind', choices=['catboost_confidence', 'catboost', 'logreg'], default='catboost_confidence')
    args = parser.parse_args()
    folder = export_classifier(args.results, args.output, args.experiment, args.kind)
    print(f'Экспорт готов: {folder.resolve()}')
