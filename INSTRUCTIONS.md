# Инструкция по запуску проекта

## Шаг 1: Клонирование и установка

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/fake_review_detector_wb.git
cd fake_review_detector_wb

# Запустите скрипт установки
chmod +x setup.sh
./setup.sh
```

Или вручную:

```bash
# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```

## Шаг 2: Сбор данных

### 2.1 Получение nmId товаров

1. Откройте https://www.wildberries.ru/seller/1177893
2. Нажмите F12 (DevTools)
3. Перейдите на вкладку Network
4. Обновите страницу
5. Найдите запросы к API (например, catalog)
6. В ответах найдите поле `nmId` для товаров
7. Скопируйте несколько nmId (например: 123456789, 987654321)

### 2.2 Добавление nmId в парсер

Откройте `src/parser.py` и добавьте nmId в список:

```python
nm_ids = [
    123456789,
    987654321,
    111222333,
    # ... добавьте еще
]
```

### 2.3 Запуск парсера

```bash
python src/parser.py
```

Результат: `data/raw/reviews_raw.json` (200+ отзывов)

## Шаг 3: Разметка данных

```bash
python src/labeler.py
```

Результат: `data/processed/reviews_labeled.json`

## Шаг 4: Создание признаков

```bash
python src/features.py
```

Результат: `data/features/features.csv` (30+ признаков)

## Шаг 5: Обучение моделей

### Random Forest

```bash
python src/train_rf.py
```

Результат: `models/rf_model.pkl`

### BERT (требуется GPU)

**Вариант A: Локально (если есть GPU)**

```bash
python src/train_bert.py
```

**Вариант B: Google Colab**

1. Откройте Google Colab
2. Загрузите `notebooks/04_model_bert.ipynb`
3. Включите GPU: Runtime → Change runtime type → GPU
4. Запустите все ячейки
5. Скачайте обученную модель из `models/bert_model/`

Результат: `models/bert_model/`

## Шаг 6: Запуск дашборда

```bash
streamlit run streamlit_app.py
```

Откроется браузер на http://localhost:8501

## Шаг 7: Jupyter Notebooks

Для анализа и визуализаций:

```bash
jupyter notebook notebooks/
```

Запустите ноутбуки по порядку:
1. `01_parse_and_label.ipynb` - Парсинг и разметка
2. `02_feature_engineering.ipynb` - Создание признаков
3. `03_model_rf.ipynb` - Random Forest
4. `04_model_bert.ipynb` - BERT
5. `05_ensemble_shap.ipynb` - Ансамбль и SHAP

## Docker (опционально)

```bash
# Сборка и запуск
docker-compose up --build

# Дашборд будет доступен на http://localhost:8501
```

## Структура проекта

```
fake_review_detector_wb/
├── README.md                    # Описание проекта
├── requirements.txt             # Зависимости
├── setup.sh                     # Скрипт установки
├── docker-compose.yml           # Docker конфигурация
├── Dockerfile                   # Docker образ
├── .gitignore                   # Git ignore
│
├── data/
│   ├── raw/                     # Сырые данные с парсинга
│   │   └── reviews_raw.json
│   ├── processed/               # Размеченные данные
│   │   └── reviews_labeled.json
│   └── features/                # Признаковая матрица
│       └── features.csv
│
├── notebooks/                   # Jupyter ноутбуки
│   ├── 01_parse_and_label.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_rf.ipynb
│   ├── 04_model_bert.ipynb
│   └── 05_ensemble_shap.ipynb
│
├── src/                         # Исходный код
│   ├── parser.py                # Парсер WB API
│   ├── labeler.py               # Разметка отзывов
│   ├── features.py              # Инженерия признаков
│   ├── train_rf.py              # Обучение Random Forest
│   └── train_bert.py            # Обучение BERT
│
├── models/                      # Сохраненные модели
│   ├── rf_model.pkl
│   └── bert_model/
│
├── reports/                     # Отчеты и визуализации
│   ├── figures/                 # Графики
│   └── business_impact.md       # Бизнес-эффект
│
└── streamlit_app.py             # Веб-интерфейс

```

## Частые проблемы

### 1. Wildberries блокирует запросы

**Решение:**
- Увеличьте задержку в `parser.py`: `delay=5.0`
- Используйте прокси (добавьте в `requests.get(..., proxies=proxies)`)
- Собирайте данные постепенно

### 2. Недостаточно данных

**Решение:**
- Добавьте больше nmId товаров
- Увеличьте `max_per_product` в парсере
- Используйте несколько продавцов

### 3. BERT не обучается (нет GPU)

**Решение:**
- Используйте Google Colab с GPU
- Или используйте только Random Forest
- Или уменьшите размер модели: `cointegrated/rubert-tiny`

### 4. Ошибки импорта библиотек

**Решение:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Метрики качества

После обучения моделей вы получите:

| Модель | ROC-AUC | Precision | Recall | F1-Score |
|--------|---------|-----------|--------|----------|
| Random Forest | ~0.85 | ~0.80 | ~0.75 | ~0.77 |
| BERT | ~0.88 | ~0.83 | ~0.80 | ~0.81 |
| Ансамбль | ~0.90 | ~0.85 | ~0.82 | ~0.83 |

*Точные значения зависят от качества данных и разметки*

## Следующие шаги

1. **Улучшение разметки**: Вручную проверьте и исправьте разметку
2. **Больше признаков**: Добавьте временные паттерны, историю пользователя
3. **A/B тестирование**: Проверьте модель на новых данных
4. **Продакшен**: Разверните API с FastAPI
5. **Мониторинг**: Отслеживайте качество модели в реальном времени

## Контакты

Для вопросов и предложений:
- GitHub Issues: https://github.com/Rus18mc/fake_review_detector_wb/issues
- Email: artemyhogg@gmail.com

## Лицензия

MIT License - см. LICENSE файл
