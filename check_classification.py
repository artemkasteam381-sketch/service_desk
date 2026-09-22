"""Проверка загрузки и одного реального прогноза без Streamlit."""
import argparse
import json
from src.classification.service import RequestClassifier
from src.classification.settings import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', help='JSON-файл со всеми доступными полями одного обращения')
    args = parser.parse_args()
    settings = load_settings()
    model = RequestClassifier(settings.model_dir, settings.device, settings.batch_size)
    if args.request:
        with open(args.request, encoding='utf-8-sig') as source:
            request = json.load(source)
    else:
        request = {'Описание': 'Не приходит SMS-код для входа',
                   'Услуга': 'Мобильное приложение "Почта России"'}
    print(json.dumps(model.predict(request), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
