import re, unicodedata
import pandas as pd

PREPROCESS_VERSION = "classification_essence_first_v1"
FIELDS = ["Услуга", "Тип запроса", "Компонент услуги 1 уровня", "Пользователь"]
ESSENCE_MARKER = re.compile(r"суть\s+обращения\s*[:：\-–—]?\s*", re.IGNORECASE)

def normalize(v):
    if pd.isna(v): return ''
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(v))).strip()

def extract_essence(text):
    # If the marker occurs more than once, use the last occurrence: in the
    # source data it is the actual user narrative rather than the form header.
    matches = list(ESSENCE_MARKER.finditer(text))
    if not matches:
        return ''
    return normalize(text[matches[-1].end():])

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

def group_key(text):
    text = normalize(text).casefold()
    text = re.sub(r'^(?:(?:re|fw|fwd)(?:\[[^\]]*\])?\s*:\s*)+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-zа-я]{2,}\b', ' EMAIL ', text)
    text = re.sub(r'\b\d{8,}\b', ' NUMBER ', text)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w]+', ' ', text)).strip()

def normalize_metadata(value):
    value = normalize(value)
    return "__MISSING__" if value in {"", "---)"} else value

def parse_description(value):
    raw = normalize(value)
    text = essence_first_text(raw)
    return {"text_core": text, "text_with_subject": raw, "group": group_key(text), "text_normalized": raw}

def search_text(value):
    return essence_first_text(value)
