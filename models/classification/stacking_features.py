
"""Одинаковые признаки при обучении стекинга и при прогнозе новых обращений."""
import re
import unicodedata
import numpy as np
import pandas as pd

STACK_FEATURE_VERSION = 'confidence_agreement_v1'
CONFIDENCE_CAT_COLUMNS = ['tfidf_top_class', 'rubert_top_class']
CONFIDENCE_NUM_COLUMNS = [
    'tfidf_max_score', 'tfidf_margin', 'tfidf_entropy',
    'rubert_max_score', 'rubert_margin', 'rubert_entropy',
    'models_agree', 'top3_overlap',
]


def stack_feature_mode(kind):
    if kind in {'logreg', 'catboost'}:
        return 'base'
    if kind == 'catboost_confidence':
        return 'confidence'
    raise ValueError(f'Неизвестная мета-модель: {kind}')


def stack_categorical_columns(meta_columns, feature_mode):
    if feature_mode not in {'base', 'confidence'}:
        raise ValueError(f'Неизвестный режим признаков: {feature_mode}')
    return list(meta_columns) + (CONFIDENCE_CAT_COLUMNS if feature_mode == 'confidence' else [])


def _stack_normalize(value):
    if pd.isna(value):
        return ''
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value))).strip()


def align_stack_probabilities(proba, source_classes, target_classes):
    proba = np.asarray(proba, dtype=float)
    source_classes = list(map(str, source_classes))
    target_classes = list(map(str, target_classes))
    if proba.ndim != 2 or proba.shape[1] != len(source_classes):
        raise ValueError('Размер матрицы вероятностей не совпадает с порядком классов')
    if len(set(source_classes)) != len(source_classes) or len(set(target_classes)) != len(target_classes):
        raise ValueError('Имена классов должны быть уникальными')
    if set(source_classes) - set(target_classes):
        raise ValueError('В модели есть класс, отсутствующий в сохранённом порядке классов')
    out = np.zeros((len(proba), len(target_classes)), dtype=float)
    position = {name: i for i, name in enumerate(target_classes)}
    for j, name in enumerate(source_classes):
        out[:, position[name]] = proba[:, j]
    return out


def _check_stack_probabilities(proba, rows, class_count, name):
    proba = np.asarray(proba, dtype=float)
    if proba.shape != (rows, class_count):
        raise ValueError(f'{name}: неверные число строк или порядок классов')
    if not np.isfinite(proba).all() or np.any(proba < 0) or np.any(proba > 1):
        raise ValueError(f'{name}: вероятности должны быть конечными числами от 0 до 1')
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError(f'{name}: сумма вероятностей должна быть равна 1')
    return proba


def make_stack_features(frame, tfidf_proba, rubert_proba, *, classes, meta_columns,
                        feature_mode='base'):
    """Только оценки и разрешённые поля; целевая метка в расчётах не используется.

    Для meta-train оценки должны быть внутренними OOF. Функция не обучается
    на данных и одинаково применяется к meta-train, внешнему validation и inference.
    """
    classes = np.asarray(classes, dtype=str)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('Нужны как минимум два уникальных класса')
    if feature_mode not in {'base', 'confidence'}:
        raise ValueError(f'Неизвестный режим признаков: {feature_mode}')
    tfidf = _check_stack_probabilities(tfidf_proba, len(frame), len(classes), 'TF-IDF')
    rubert = _check_stack_probabilities(rubert_proba, len(frame), len(classes), 'RuBERT')
    out = {}
    for prefix, proba in [('tfidf', tfidf), ('rubert', rubert)]:
        for i in range(len(classes)):
            out[f'{prefix}_class_{i + 1:02d}_score'] = proba[:, i]
    for column in meta_columns:
        values = frame[column] if column in frame else pd.Series('__MISSING__', index=frame.index)
        out[column] = values.map(_stack_normalize).replace({'': '__MISSING__'}).astype(str).to_numpy()
    if feature_mode == 'confidence':
        orders = {}
        for prefix, proba in [('tfidf', tfidf), ('rubert', rubert)]:
            # При равных оценках выбираем первый класс в сохранённом порядке.
            order = np.argsort(-proba, axis=1, kind='stable')
            orders[prefix] = order
            scores = np.take_along_axis(proba, order[:, :2], axis=1)
            out[f'{prefix}_max_score'] = scores[:, 0]
            out[f'{prefix}_margin'] = scores[:, 0] - scores[:, 1]
            # Нормированная энтропия: 0 — один вариант, 1 — равномерные оценки.
            out[f'{prefix}_entropy'] = -(proba * np.log(np.clip(proba, 1e-15, 1.0))).sum(axis=1) / np.log(len(classes))
            out[f'{prefix}_top_class'] = classes[order[:, 0]]
        out['models_agree'] = (orders['tfidf'][:, 0] == orders['rubert'][:, 0]).astype('int8')
        k = min(3, len(classes))
        shared = orders['tfidf'][:, :k, None] == orders['rubert'][:, None, :k]
        out['top3_overlap'] = shared.any(axis=2).sum(axis=1).astype('int8')
    # Позиционный порядок сохраняется и при непоследовательных индексах исходных строк.
    return pd.DataFrame(out)


def check_stack_feature_schema(frame, bundle):
    mode = bundle.get('meta_feature_mode', 'base')
    if mode == 'confidence' and bundle.get('stack_feature_version') != STACK_FEATURE_VERSION:
        raise ValueError('Версия признаков сохранённой модели не поддерживается')
    expected = bundle.get('meta_feature_names')
    if expected is not None and list(frame.columns) != list(expected):
        raise ValueError('Признаки или их порядок отличаются от сохранённой модели')
