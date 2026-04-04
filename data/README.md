# Директория данных

Эта директория содержит данные проекта на разных этапах обработки.

## Структура

```
data/
├── raw/              # Сырые данные с парсинга
│   └── reviews_raw.json
├── processed/        # Размеченные данные
│   └── reviews_labeled.json
└── features/         # Признаковая матрица
    └── features.csv
```

## Описание файлов

### raw/reviews_raw.json
Сырые отзывы, собранные с Wildberries API.

Формат:
```json
[
  {
    "nmId": 123456789,
    "id": "abc123",
    "text": "Текст отзыва",
    "rating": 5,
    "date": "2026-04-03T12:00:00Z",
    "likes": 10,
    "dislikes": 2,
    "has_photo": true,
    "user_name": "Пользователь"
  }
]
```

### processed/reviews_labeled.json
Отзывы с разметкой (фейк/реальный).

Формат:
```json
[
  {
    "text": "Текст отзыва",
    "label": 1,
    "nmId": 123456789,
    "rating": 5,
    "date": "2026-04-03T12:00:00Z",
    "likes": 10,
    "dislikes": 2,
    "has_photo": true
  }
]
```

Где `label`:
- `1` - фейковый отзыв
- `0` - реальный отзыв

### features/features.csv
Признаковая матрица с 30+ признаками.

Содержит все признаки из `src/features.py`:
- Поведенческие (hour, is_night, likes_ratio, etc.)
- Текстовые (text_length, word_count, etc.)
- Лингвистические (unique_words_ratio, fake_keywords_count, etc.)
- Шаблонность (repetitive_ngrams, etc.)
- Рейтинг (is_max_rating, rating_text_mismatch, etc.)

## Как создать данные

1. Запустите парсер:
```bash
python src/parser.py
```

2. Разметьте данные:
```bash
python src/labeler.py
```

3. Создайте признаки:
```bash
python src/features.py
```

## Примечание

Файлы данных не включены в git (см. .gitignore).
Вам нужно собрать их самостоятельно, следуя инструкциям выше.
