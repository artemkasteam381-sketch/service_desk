

import re, unicodedata, warnings
from functools import lru_cache
import pymorphy3
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

TEXT = 'Описание 2'
TARGET = 'Вид запроса'
OTHER = 'Вне топ-15'
FEATURES = {
    'service_user_type_component': ['Услуга', 'Пользователь', 'Тип запроса', 'Компонент услуги 1 уровня'],
}
ALL_META = FEATURES['service_user_type_component']
# В ноутбуке режим берётся из конфигурации; при прогнозе — из сохранённого bundle.
# Значение full сохраняет совместимость со старыми bundle без настройки текста.
TEXT_MODE = globals().get('TEXT_MODE', 'full')
ESSENCE_FALLBACK = globals().get('ESSENCE_FALLBACK', 'full')
CLEAN_TEXT = '_clean_text'
LEMMA_TEXT = '_lemma_text'
ESSENCE_TEXT = '_essence_text'
MODEL_TEXT = '_model_text'
MODEL_LEMMA_TEXT = '_model_lemma_text'
ESSENCE_FOUND = '_essence_found'
ESSENCE_MARKER = re.compile(r'суть\s+обращения\s*[:：\-–—]?\s*', re.IGNORECASE)
TEXT_BLOCK_COLUMNS = {
    'clean_word': MODEL_TEXT,
    'lemma_word': MODEL_LEMMA_TEXT,
    'clean_char': MODEL_TEXT,
}

# Вес каждого блока признаков при объединении. Метаданные имеют вес 1.0.
transformer_weights = {
    "clean_word": 1.0,
    "lemma_word": 1.0,
    "clean_char": 1.0,
}


@lru_cache(maxsize=1)
def morph_analyzer():
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=100_000)
def lemmatize_token(token):
    # Нормализуем русские слова; числа, латиницу и смешанные коды сохраняем.
    if re.fullmatch(r'[а-яё]+', token):
        return morph_analyzer().parse(token)[0].normal_form
    return token


def lemmatize(text):
    tokens = re.findall(r'(?u)\b\w+\b', text.lower())
    return ' '.join(lemmatize_token(token) for token in tokens)

def normalize(v):
    if pd.isna(v): return ''
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(v))).strip()

def has_essence_marker(text):
    return bool(ESSENCE_MARKER.search(text))

def extract_essence(text):
    # If the marker occurs more than once, use the last occurrence: in the
    # source data it is the actual user narrative rather than the form header.
    matches = list(ESSENCE_MARKER.finditer(text))
    if not matches:
        return ''
    return normalize(text[matches[-1].end():])

def select_model_text(frame, text_mode=None, essence_fallback=None):
    text_mode = TEXT_MODE if text_mode is None else text_mode
    essence_fallback = ESSENCE_FALLBACK if essence_fallback is None else essence_fallback
    full = frame[CLEAN_TEXT]
    essence = frame[ESSENCE_TEXT]
    if text_mode == 'full':
        return full
    if text_mode == 'essence':
        if essence_fallback == 'full':
            return essence.where(essence.ne(''), full)
        return essence
    if text_mode == 'full_plus_essence':
        return (full + ' ' + essence.where(essence.ne(''), '')).str.strip()
    raise ValueError('TEXT_MODE должен быть full, essence или full_plus_essence')


def essence_first_text(text):
    """Переносим последний непустой фрагмент сути; остальной текст не дублируем."""
    text = normalize(text)
    matches = list(ESSENCE_MARKER.finditer(text))
    if not matches:
        return text  # для RuBERT отсутствие маркера всегда означает полное описание
    marker = matches[-1]
    essence = normalize(text[marker.end():])
    if not essence:
        return text
    context = normalize(text[:marker.start()])
    # Исходный маркер тоже переносим вместе с фрагментом; сохраняем регистр,
    # пунктуацию, коды и весь текст, лемматизацию к RuBERT не применяем.
    return normalize(text[marker.start():] + ' ' + context)


def select_rubert_text(frame, text_mode='essence_first'):
    # Используем исходное описание, а не TF-IDF MODEL_TEXT с повторённой сутью.
    full = frame[TEXT].map(normalize)
    if text_mode == 'full':
        return full
    if text_mode == 'essence_first':
        return full.map(essence_first_text)
    raise ValueError("RuBERT text_mode: 'full' или 'essence_first'")


def prepare_input(frame, text_mode=None, essence_fallback=None):
    frame = frame.copy()
    if TEXT not in frame:
        if 'Описание' in frame: frame[TEXT] = frame['Описание']
        else: frame[TEXT] = ''
    frame[TEXT] = frame[TEXT].map(normalize)
    frame[CLEAN_TEXT] = frame[TEXT]
    frame[ESSENCE_TEXT] = frame[CLEAN_TEXT].map(extract_essence)
    frame[ESSENCE_FOUND] = frame[CLEAN_TEXT].map(has_essence_marker)
    frame[MODEL_TEXT] = select_model_text(frame, text_mode, essence_fallback)
    frame[MODEL_LEMMA_TEXT] = frame[MODEL_TEXT].map(lemmatize)
    for c in ALL_META:
        if c not in frame: frame[c] = '__MISSING__'
        frame[c] = frame[c].map(normalize).replace({'': '__MISSING__', '---)': '__MISSING__'})
    return frame

def group_key(text):
    text = normalize(text).casefold()
    text = re.sub(r'^(?:(?:re|fw|fwd)(?:\[[^\]]*\])?\s*:\s*)+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-zа-я]{2,}\b', ' EMAIL ', text)
    text = re.sub(r'\b\d{8,}\b', ' NUMBER ', text)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w]+', ' ', text)).strip()

def make_vectorizers():
    # Признаки
    clean_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=80_000,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )

    lemma_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=80_000,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )

    clean_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )
    return dict(clean_word=clean_word, lemma_word=lemma_word, clean_char=clean_char)


def fit_features(frame, columns):
    vectorizers = make_vectorizers()
    text_columns = TEXT_BLOCK_COLUMNS.copy()
    weights = transformer_weights.copy()
    # Словари и IDF обучаются только на переданном train.
    parts = [vectorizer.fit_transform(frame[text_columns[name]]) * weights[name]
             for name, vectorizer in vectorizers.items()]
    encoder = None
    if columns:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        parts.append(encoder.fit_transform(frame[columns]))
    fitted = dict(vectorizers=vectorizers, text_columns=text_columns,
                  transformer_weights=weights, encoder=encoder, columns=list(columns))
    return fitted, hstack(parts, format='csr')


def transform(frame, fitted):
    # Используем те же блоки, порядок и веса, которые сохранены при обучении.
    parts = [vectorizer.transform(frame[fitted['text_columns'][name]])
             * fitted['transformer_weights'][name]
             for name, vectorizer in fitted['vectorizers'].items()]
    if fitted['columns']:
        parts.append(fitted['encoder'].transform(frame[fitted['columns']]))
    return hstack(parts, format='csr')


def train(X, y, config):
    # liblinear обучает бинарные модели; OvR расширяет их на 16 классов.
    model = OneVsRestClassifier(LogisticRegression(**config), n_jobs=1)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X, y)
    return model, sum(issubclass(w.category, ConvergenceWarning) for w in caught)

def inner_splits(frame, n_splits, seed):
    if frame.groupby('_target')['_group'].nunique().min() < n_splits:
        raise ValueError('Недостаточно независимых групп для внутренней CV')
    splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(frame, frame['_target'], frame['_group']))
    for a,b in splits:
        assert not set(frame.iloc[a]['_group']) & set(frame.iloc[b]['_group'])
        if set(frame.iloc[a]['_target']) != set(frame['_target']): raise ValueError('Класс отсутствует во внутреннем train')
    return splits

def choose_config(frame, columns, configs, splits):
    if len(configs) == 1:
        return configs[0].copy(), [dict(**configs[0], inner_weighted_f1=None, convergence_warnings=0)]
    scores = [[] for _ in configs]
    nw = [0 for _ in configs]
    for a,b in splits:
        fit, X = fit_features(frame.iloc[a], columns)
        V = transform(frame.iloc[b], fit)
        for j,cfg in enumerate(configs):
            model, w = train(X, frame.iloc[a]['_target'], cfg)
            pred = model.predict(V)
            scores[j].append(f1_score(frame.iloc[b]['_target'], pred, average='weighted', zero_division=0))
            nw[j] += w
    winner = int(np.argmax([np.mean(x) for x in scores]))
    log = [dict(**c, inner_weighted_f1=float(np.mean(s)), convergence_warnings=w) for c,s,w in zip(configs,scores,nw)]
    return configs[winner].copy(), log

def fit_bundle(frame, columns, config, text_mode=None):
    text_mode = TEXT_MODE if text_mode is None else text_mode
    frame = prepare_input(frame, text_mode=text_mode)
    fitted, X = fit_features(frame, columns)
    model, nw = train(X, frame['_target'], config)
    return dict(features=fitted, model=model, columns=columns, config=config, convergence_warnings=nw,
                text_mode=text_mode, essence_fallback=ESSENCE_FALLBACK)

def predict_frame(bundle, frame):
    frame = prepare_input(frame, text_mode=bundle.get('text_mode', 'full'),
                          essence_fallback=bundle.get('essence_fallback', 'full'))
    if frame.empty: return np.empty((0, len(bundle['model'].classes_)))
    return bundle['model'].predict_proba(transform(frame, bundle['features']))

def recommend(bundle, description, service=None, user=None, request_type=None, component=None, supplied_label=None):
    row = {TEXT: description, 'Услуга': service, 'Пользователь': user,
           'Тип запроса': request_type, 'Компонент услуги 1 уровня': component}
    p = predict_frame(bundle, pd.DataFrame([row]))[0]
    classes = bundle['model'].classes_
    order = np.argsort(-p)[:3]
    return {'recommended': classes[order[0]], 'top3': [(classes[i], float(p[i])) for i in order],
            'supplied_label': supplied_label,
            'comparison': 'не передан' if supplied_label is None else ('совпадает' if supplied_label == classes[order[0]] else 'расхождение; требуется проверка'),
            'score_meaning': 'Некалиброванная вероятность модели; не гарантия правильности и не доказательство принадлежности к топ-15'}

def metrics(y, pred):
    return dict(weighted_f1=f1_score(y,pred,average='weighted',zero_division=0),
                weighted_precision=precision_score(y,pred,average='weighted',zero_division=0),
                macro_f1=f1_score(y,pred,average='macro',zero_division=0), accuracy=accuracy_score(y,pred))
