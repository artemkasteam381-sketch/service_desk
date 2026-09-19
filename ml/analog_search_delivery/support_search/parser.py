"""Snapshot of the existing notebook parser. Audit before production use.

Only parser definitions are copied; no notebook training code is imported.
"""
import re
import html
import unicodedata
import json
import numpy as np
from html.parser import HTMLParser
from urllib.parse import urlsplit

class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.depth = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}: self.depth += 1
        if tag in {"p", "br", "div", "li", "tr"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style"}: self.depth = max(0, self.depth - 1)
        if tag in {"p", "div", "li", "tr"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.depth: self.parts.append(data)

def normalize_space(text):
    return re.sub(r"\s+", " ", text).strip()

def plain_text(value):
    text = unicodedata.normalize("NFKC", html.unescape(str(value or "")))
    text = re.sub(r"[\u200b-\u200f\ufeff\x00-\x08]", "", text)
    if re.search(r"</?(?:html|body|div|p|br|span|table|a|script|style)(?:\s|>|/)", text, re.I):
        parser = VisibleHTML()
        parser.feed(text)
        text = "".join(parser.parts)
    return normalize_space(text)

LEAK_PATTERN = re.compile(r"Завершены\s+все\s+дочерние\s+запросы\s*:|"
    r"(?:Вид\s+запроса|Результат\s+работ|Описание\s+решения|Кем\s+решен\s*\(группа\)|"
    r"Фактическое\s+время\s+выполнения)\s*:", re.I)

QUOTE_PATTERN = re.compile(r"(?:-{2,}\s*)?(?:Original Message|Forwarded message|Пересылаемое сообщение)|"
    r"При\s+ответе\s+добавьте\s+комментарий\s+выше|"
    r"(?:From|От кого|Отправлено)\s*:[\s\S]{0,180}?(?:Sent|Date|Кому|Дата|To)\s*:", re.I)

SIGNATURE_PATTERN = re.compile(r"\bС\s+уважением\b|\bSent\s+from\b|"
    r"\bОтправлено\s+(?:из|с)\s+(?:Mail|Outlook|iPhone|Android|мобильного)", re.I)

FORM_RULES = {
    "issue": r"Суть\s+обращения",
    "steps": r"Последовательность\s+действий",
    "topic": r"Тематика",
    "channel": r"Канал\s+поступления\s+обращения",
    "intake_date": r"Дата\s+поступления\s+обращения",
    "phone": r"Номер\s+телефона\s+для\s+связи\s+с\s+ОПС|Телефон",
    "tracking": r"Номер\s+ШПИ|ШПИ/номер",
    "address": r"Адрес\s+заявителя|Адрес\s+почтомата",
    "post_index": r"Индекс\s+ДТИ\s*\(выделенный\s+участок\)|Индекс\s+ОПС",
    "window": r"Номер\s+окна",
    "account": r"Учетная\s+запись|Учётная\s+запись",
    "email": r"Электронная\s+почта\s+клиента|Электронная\s+почта",
    "error_time": r"Время\s+возникновения\s+ошибки",
    "extra": r"Дополнительная\s+информация",
    "platform": r"Платформа|Операционная\s+система|Версия\s+ОС",
    "version": r"Версия\s+приложения",
    "device": r"Модель\s+устройства|Устройство",
}

FORM_PATTERN = re.compile("|".join(f"(?P<{k}>{v})\\s*:" for k, v in FORM_RULES.items()), re.I)

def informative(text):
    return bool(re.search(r"[a-zа-яё]{2,}", text, re.I)) and not re.fullmatch(
        r"(?:нет|не указан[ао]?|отсутствует|да|описание отсутствует\s*\(пустое тело письма\)\.?|тема отсутствует\.?)", text.strip(), re.I)

def clean_for_model(text):
    text = normalize_space(text)
    text = re.sub(r"https?://[^\s<>]+", lambda m: " domain_" + (urlsplit(m.group()).hostname or "url") + " ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", " EMAIL ", text)
    text = re.sub(r"\b[A-Z]{2}\d{9}[A-Z]{2}\b|(?<!\d)\d{14}(?!\d)", " TRACK_ID ", text, flags=re.I)
    text = re.sub(r"(?<!\w)\+?\d[\d ()-]{9,}\d(?!\w)", " LONG_ID ", text)
    # Удаляется только остаток маскировки, не соседние слова или содержательные числа.
    text = re.sub(r"(?<!\w)-{3,}(?!\w)", " ", text)
    text = re.sub(r"^(?:здравствуйте|добрый\s+(?:день|вечер)|доброе\s+утро)[!.,\s]*", "", text, flags=re.I)
    text = re.sub(r"^(?:уважаемые\s+)?коллеги[!.,\s]*", "", text, flags=re.I)
    return normalize_space(text).lower()

def parse_description(value):
    raw = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = plain_text(raw)
    cuts = [(m.start(), kind, m.group()) for kind, pattern in [("leak", LEAK_PATTERN), ("quote", QUOTE_PATTERN)]
            if (m := pattern.search(text))]
    cuts.sort()
    cutoff = cuts[0][0] if cuts else len(text)
    safe, removed_tail = text[:cutoff].strip(), text[cutoff:].strip()
    leak_match, quote_match = LEAK_PATTERN.search(text), QUOTE_PATTERN.search(text)
    subject_match = re.search(r"Тема\s+письма\s*:\s*", safe, re.I)
    body_match = re.search(r"Текст\s+письма\s*:\s*", safe, re.I)
    subject = safe[subject_match.end():body_match.start() if body_match else len(safe)].strip() if subject_match else ""
    body = safe[body_match.end():].strip() if body_match else "" if subject_match else safe
    if not informative(subject): subject = ""
    signature_match = SIGNATURE_PATTERN.search(body)
    signature = body[signature_match.start():].strip() if signature_match else ""
    body_main = body[:signature_match.start()].strip() if signature_match else body
    fields, matches = {}, list(FORM_PATTERN.finditer(body_main))
    prefix = body_main[:matches[0].start()].strip() if matches else body_main
    for j, match in enumerate(matches):
        val = body_main[match.end():matches[j+1].start() if j+1 < len(matches) else len(body_main)].strip()
        fields.setdefault(match.lastgroup, []).append(val)
    get = lambda key: " ".join(v for v in fields.get(key, []) if informative(v))
    issue, steps = get("issue"), get("steps")
    # Маскированные названия email-полей иногда остаются в хвосте предыдущего поля.
    scrub = lambda x: re.sub(r"\bЭл(?:ект[а-яё ]*)?-{2,}.*$", "", x, flags=re.I).strip()
    issue, prefix, steps = scrub(issue), scrub(prefix), scrub(steps)
    core_parts = [issue if informative(issue) else prefix, steps, get("extra"), get("platform"), get("version"), get("device")]
    # Из-за повреждённых маркеров содержательный текст иногда попадает в поле email/телефона.
    # Сохраняем отдельно для ручного разбора, не добавляем автоматически: здесь встречаются и ответы поддержки.
    problem_hint = re.compile(r"не\s+(?:мог|работ|приход|получ|загру|откры|уда|доступ)|ошиб|проблем|код|уведом|опоздан|приложен", re.I)
    residual_fields = {k: vals for k, vals in fields.items() if k not in {"issue","steps","extra","platform","version","device","topic"}}
    residual = " ".join(v for vals in residual_fields.values() for v in vals
        if len(re.findall(r"[а-яёa-z]{2,}", v, re.I)) >= 7 and problem_hint.search(v))
    core = " ".join(dict.fromkeys(p for p in core_parts if informative(p)))
    subject_clean, core_clean = clean_for_model(subject), clean_for_model(core)
    with_subject = normalize_space(subject_clean + " " + core_clean)
    topic_clean = clean_for_model(get("topic"))
    warnings_list = []
    if leak_match: warnings_list.append("possible_post_registration_leak")
    if quote_match: warnings_list.append("quoted_history_removed")
    if not core_clean: warnings_list.append("empty_core")
    if len(with_subject) < 12: warnings_list.append("too_little_text")
    if len(text) > 2000: warnings_list.append("long_text")
    if len(with_subject) < 0.25 * max(len(text), 1): warnings_list.append("large_removal")
    if len(matches) > len(fields): warnings_list.append("repeated_form_fields")
    if residual: warnings_list.append("possible_narrative_in_metadata_field")
    return {
        "text_normalized": text, "subject": subject, "body_safe": body_main,
        "issue": issue, "steps": steps, "topic": get("topic"), "channel": get("channel"),
        "signature_removed": signature, "quoted_history": text[quote_match.start():] if quote_match else "",
        "leakage_quarantine": text[leak_match.start():] if leak_match else "",
        "removed_tail": removed_tail, "form_fields_json": json.dumps(fields, ensure_ascii=False),
        "form_residual_text": residual,
        "text_core": core_clean, "text_subject": subject_clean,
        "text_with_subject": with_subject, "text_with_topic": normalize_space(with_subject + " " + topic_clean),
        "text_safe_minimal": clean_for_model(safe),
        "has_subject": int(bool(subject)), "has_issue": int(bool(issue)),
        "has_form": int(bool(matches)), "has_quote": int(bool(quote_match)),
        "has_leakage_marker": int(bool(leak_match)), "raw_length": len(raw), "core_length": len(core_clean),
        "removal_fraction": 1 - len(with_subject) / max(len(text), 1),
        "review_required": bool(warnings_list), "parse_warnings": ";".join(warnings_list),
        "os_android": int(bool(re.search(r"\bandroid\b|андроид", core_clean, re.I))),
        "os_ios": int(bool(re.search(r"\bios\b|iphone|айфон", core_clean, re.I))),
        "os_windows": int(bool(re.search(r"\bwindows\b|виндовс", core_clean, re.I))),
        "os_linux": int(bool(re.search(r"\blinux\b|линукс", core_clean, re.I))),
    }

def search_text(text: str) -> str:
    return parse_description(text)["text_with_subject"]
