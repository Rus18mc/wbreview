# Детектор накрученных отзывов на Wildberries

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-NLP%20%2B%20Ensemble-orange)

Система автоматического определения фейковых (накрученных) отзывов на маркетплейсе Wildberries с использованием ML (tabular + NLP + ensemble).

---

## 🚀 Ключевые особенности

* Детекция фейковых отзывов на основе **поведенческих + текстовых признаков**
* Комбинация моделей: **Random Forest + BERT + Ensemble**
* 30+ engineered features (behavioral + linguistic + template)
* Интерпретация моделей через **SHAP и LIME**
* End-to-end ML pipeline + Streamlit интерфейс

---

## 🎯 Бизнес-задача

Фейковые отзывы:

* искажают рейтинг товаров
* ухудшают пользовательский опыт
* приводят к возвратам и потерям

👉 Цель: автоматически выявлять подозрительные отзывы и снижать бизнес-риски.

---

## 📊 Данные

* **Источник**: Отзывы с товаров продавца ID 1177893 на Wildberries
* **Объем**: 200+ отзывов
* **API**: `https://feedbacks1.wb.ru/feedbacks/v1/{nmId}`

⚠️ Разметка выполнена эвристически (weak supervision)

---

## 🔬 Методология

### Разметка данных (weak labeling)

Поскольку реальной разметки нет, использованы эвристические правила:

**Фейк (1):**

* Рейтинг = 5
* Текст < 150 символов
* 2+ ключевых слова: "отлично", "супер", "рекомендую", "качество", "доставка", "быстро", "огонь", "топ"

**Реальный (0):**

* Рейтинг ≤ 3
* ИЛИ содержит конкретику (цифры, время использования, "сломался", "брак")
* ИЛИ длинный текст (>300 символов)

**Серые (исключены):** Остальные отзывы

---

### Признаки (30+)

#### A. Поведенческие

* `hour`, `is_night`, `is_weekend` — временные паттерны
* `likes_ratio`, `has_photo` — взаимодействие
* `text_length`, `word_count` — объем текста

#### B. Лингвистические

* `exclamation_count`, `uppercase_ratio` — эмоциональность
* `unique_words_ratio` — разнообразие лексики
* `fake_keywords_count`, `real_keywords_count` — ключевые слова

#### C. Шаблонность

* `repetitive_bigrams`, `repetitive_trigrams` — повторяемость
* `starts_with_positive`, `ends_with_recommend` — шаблон
* `rating_text_mismatch` — несоответствие

---

## 🤖 Модели

### 1. Random Forest

* 30+ engineered features
* GridSearchCV tuning
* Хорошо работает на табличных признаках

### 2. BERT (rubert-tiny2)

* Модель: `cointegrated/rubert-tiny2`
* Fine-tuning: 5 эпох
* Вход: текст отзывов

### 3. Ensemble

* Усреднение вероятностей RF + BERT
* Повышение устойчивости модели

---

## 📈 Результаты

| Модель        | ROC-AUC | Precision | Recall | F1  |
| ------------- | ------- | --------- | ------ | --- |
| Random Forest | TBD     | TBD       | TBD    | TBD |
| BERT          | TBD     | TBD       | TBD    | TBD |
| Ensemble      | TBD     | TBD       | TBD    | TBD |

👉 (метрики можно добавить после финального обучения)

---

## 🏗️ ML Pipeline

```
Parsing → Labeling → Feature Engineering → Modeling → Evaluation → Dashboard
```

---

## 💰 Бизнес-эффект

**Допущения:**

* 1 фейковый отзыв ≈ 500 руб. потерь

**Формула:**

```
Экономия = TP × 500 руб.
```

Пример:
100 выявленных фейков → **~50,000 руб. экономии**

---

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/yourusername/fake_review_detector_wb.git
cd fake_review_detector_wb
pip install -r requirements.txt
```

### 1. Сбор данных

```bash
python src/parser.py
```

### 2. Подготовка данных

```bash
python src/labeler.py
python src/features.py
```

### 3. Обучение моделей

```bash
python src/train_rf.py
python src/train_bert.py
```

### 4. Запуск интерфейса

```bash
streamlit run streamlit_app.py
```

---

## 📁 Структура проекта

```
fake_review_detector_wb/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
└── streamlit_app.py
```

---

## 🔍 Explainability (XAI)

**SHAP (Random Forest)**

* выявление важных признаков

**LIME (BERT)**

* объяснение текстовых предсказаний

---

## ⚠️ Ограничения

* эвристическая разметка
* небольшой датасет
* один продавец
* ограниченная обобщаемость

---

## 🛠️ Возможные улучшения

* [ ] Ручная разметка
* [ ] User-level features
* [ ] Использование LLM
* [ ] Production pipeline

---

## 📚 Технологии

* Python 3.9+
* scikit-learn, transformers, torch
* SHAP, LIME
* Streamlit

---

## 👤 Автор

Ruslan Saadetdinov
ML Engineer / Data Science

---

## 📄 Лицензия

MIT License

