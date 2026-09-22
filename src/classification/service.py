"""Принимает поля регистрации и возвращает JSON-совместимую рекомендацию."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from collections.abc import Mapping
import numpy as np
import pandas as pd

_IMPORT_LOCK = threading.RLock()


def _load_runtime(directory, manifest):
    required = {'__init__.py', 'inference.py', 'stacking_features.py', 'stacking_inference.py',
                'stacking_bundle.joblib', 'requirements_versions.txt'}
    hashes = manifest.get('files_sha256', {})
    if manifest.get('format_version') != 1 or set(hashes) != required:
        raise ValueError('Неподдерживаемый или неполный экспорт модели')
    for name, expected in hashes.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name}: файл отличается от экспорта модели; повторите экспорт целиком')
    signature = hashlib.sha256((str(directory) + json.dumps(hashes, sort_keys=True)).encode()).hexdigest()[:20]
    package = '_service_desk_model_' + signature
    with _IMPORT_LOCK:
        if package not in sys.modules:
            spec = importlib.util.spec_from_file_location(package, directory / '__init__.py',
                                                        submodule_search_locations=[str(directory)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(package, None)
                raise
        return importlib.import_module(package + '.stacking_inference')


def _text(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ''
    return str(value).strip()


class RequestClassifier:
    def __init__(self, model_dir, device='cpu', batch_size=16):
        self.model_dir = Path(model_dir).resolve()
        manifest_path = self.model_dir / 'model_manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Нет экспорта модели: {manifest_path}. Выполните tools/export_classifier.py.')
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        self._runtime = _load_runtime(self.model_dir, self.manifest)
        self._loaded = self._runtime.load_stacking(self.model_dir / 'stacking_bundle.joblib',
                                                  device=device, batch_size=batch_size)
        bundle = self._loaded['bundle']
        if bundle.get('meta_kind') != self.manifest['meta_kind']:
            raise ValueError('Тип модели не совпадает с описанием экспорта')
        if bundle.get('stack_feature_version') != self.manifest['stack_feature_version']:
            raise ValueError('Версия признаков не совпадает с описанием экспорта')
        self.classes = tuple(map(str, bundle['classes']))
        self.meta_columns = tuple(bundle['meta_columns'])
        if 'Вид запроса' in self.meta_columns:
            raise ValueError('Исходный вид запроса не должен быть признаком модели')
        self._lock = threading.RLock()

    def predict(self, request, current_kind=None):
        if not isinstance(request, Mapping):
            raise TypeError('Обращение должно быть словарём полей регистрации')
        description = _text(request.get('Описание 2')) or _text(request.get('Описание'))
        if not description:
            raise ValueError('Введите описание обращения')
        # Явный список полей исключает результат работ и исходную целевую категорию.
        model_input = {'Описание 2': description}
        model_input.update({column: request.get(column) for column in self.meta_columns})
        supplied = _text(current_kind if current_kind is not None else request.get('Вид запроса')) or None
        with self._lock:
            probabilities = self._runtime.predict_stacking_proba(self._loaded, [model_input])[0]
        if len(probabilities) != len(self.classes) or not np.isfinite(probabilities).all():
            raise ValueError('Модель вернула некорректные оценки')
        order = np.argsort(-probabilities, kind='stable')[:3]
        recommended = self.classes[order[0]]
        if recommended == 'Вне топ-15':
            comparison = 'outside_top15'
        elif supplied is None:
            comparison = 'not_supplied'
        else:
            comparison = 'matches' if supplied == recommended else 'differs'
        return {
            'recommended_kind': recommended,
            'top3': [{'label': self.classes[i], 'score': float(probabilities[i])} for i in order],
            'current_kind': supplied, 'comparison': comparison,
            'missing_fields': [name for name in self.meta_columns if not _text(model_input[name])],
            'model_variant': self.manifest['meta_kind'],
        }
