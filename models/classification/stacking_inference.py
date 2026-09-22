
"""Прогноз стекинга после распаковки ZIP, без запуска обучающих ячеек ноутбука."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .inference import TEXT, prepare_input, predict_frame, select_rubert_text
from .stacking_features import (
    make_stack_features, align_stack_probabilities, check_stack_feature_schema,
)


def load_stacking(bundle_path, device=None, batch_size=16):
    if batch_size < 1:
        raise ValueError('batch_size должен быть положительным')
    bundle = joblib.load(Path(bundle_path))
    if bundle.get('rubert_preprocess_version', 'essence_first_v1') != 'essence_first_v1':
        raise ValueError('Неизвестная версия подготовки текста RuBERT')
    return dict(bundle=bundle, device=device, batch_size=batch_size, encoder=None)


def _frozen_embeddings(loaded, texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    bundle = loaded['bundle']
    if loaded['encoder'] is None:
        device = torch.device(loaded['device'] or ('cuda' if torch.cuda.is_available() else 'cpu'))
        revision = bundle.get('rubert_revision', 'main')
        tokenizer = AutoTokenizer.from_pretrained(bundle['rubert_model'], revision=revision)
        tokenizer.padding_side = tokenizer.truncation_side = 'right'
        model = AutoModel.from_pretrained(bundle['rubert_model'], revision=revision)
        model.requires_grad_(False)
        model.to(device).eval()
        loaded['encoder'] = (tokenizer, model, device)
    tokenizer, model, device = loaded['encoder']
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), loaded['batch_size']):
            encoded = tokenizer(
                texts[start:start + loaded['batch_size']], padding=True, truncation=True,
                max_length=bundle['rubert_max_length'], return_tensors='pt',
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            chunks.append(pooled.cpu().numpy().astype('float32'))
    return np.vstack(chunks) if chunks else np.empty((0, model.config.hidden_size), dtype='float32')


def predict_stacking_proba(loaded, requests):
    bundle = loaded['bundle']
    classes = np.asarray(bundle['classes'])
    if isinstance(requests, dict):
        requests = [requests]
    frame = prepare_input(pd.DataFrame(requests), text_mode=bundle['text_mode'],
                          essence_fallback=bundle.get('essence_fallback', 'full'))
    if frame.empty:
        return np.empty((0, len(classes)))
    tfidf = align_stack_probabilities(
        predict_frame(bundle['tfidf_bundle'], frame), bundle['tfidf_bundle']['model'].classes_, classes,
    )
    embeddings = _frozen_embeddings(loaded, select_rubert_text(frame, bundle['rubert_text_mode']).tolist())
    rubert = align_stack_probabilities(bundle['rubert_head'].predict_proba(embeddings),
                                      bundle['rubert_head'].classes_, classes)
    features = make_stack_features(
        frame, tfidf, rubert, classes=classes, meta_columns=bundle['meta_columns'],
        feature_mode=bundle.get('meta_feature_mode', 'base'),
    )
    check_stack_feature_schema(features, bundle)
    model = bundle['meta_model']
    model_classes = model.named_steps['classifier'].classes_ if hasattr(model, 'named_steps') else model.classes_
    return align_stack_probabilities(model.predict_proba(features), model_classes, classes)


def recommend_stacking(loaded, description, **registration_fields):
    row = dict(registration_fields)
    row[TEXT] = description
    proba = predict_stacking_proba(loaded, [row])[0]
    classes = np.asarray(loaded['bundle']['classes'])
    order = np.argsort(-proba, kind='stable')[:3]
    return dict(
        recommended=str(classes[order[0]]),
        top3=[(str(classes[i]), float(proba[i])) for i in order],
        meta_model=loaded['bundle'].get('meta_kind'),
        feature_mode=loaded['bundle'].get('meta_feature_mode', 'base'),
    )
