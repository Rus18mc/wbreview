# 🚀 БЫСТРЫЙ СТАРТ

Проект готов к использованию! Следуйте этим шагам:

## 1️⃣ Перейдите в директорию проекта

```bash
cd /tmp/fake_review_detector_wb
```

## 2️⃣ Установите зависимости

```bash
./setup.sh
```

Или вручную:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3️⃣ Получите nmId товаров

1. Откройте https://www.wildberries.ru/seller/1177893
2. Нажмите F12 → Network → обновите страницу
3. Найдите запросы к API и скопируйте nmId товаров
4. Добавьте их в `src/parser.py`:

```python
nm_ids = [
    123456789,  # Замените на реальные nmId
    987654321,
    # ... добавьте еще
]
```

## 4️⃣ Запустите пайплайн

```bash
python src/parser.py      # Парсинг отзывов
python src/labeler.py     # Разметка данных
python src/features.py    # Создание признаков
python src/train_rf.py    # Обучение Random Forest
```

## 5️⃣ Запустите дашборд

```bash
streamlit run streamlit_app.py
```

Откроется браузер на http://localhost:8501

## 📚 Альтернативные варианты

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

### Docker
```bash
docker-compose up --build
```

## 📖 Документация

- **README.md** - полное описание проекта
- **INSTRUCTIONS.md** - детальная инструкция
- **PROJECT_SUMMARY.md** - краткое резюме
- **CHECKLIST.md** - чек-лист готовности

## ❓ Частые вопросы

**Q: Где взять nmId товаров?**  
A: См. шаг 3 выше или INSTRUCTIONS.md

**Q: Нужен ли GPU для обучения?**  
A: Random Forest работает на CPU. BERT требует GPU (используйте Google Colab)

**Q: Сколько нужно отзывов?**  
A: Минимум 200, рекомендуется 500+

**Q: Wildberries блокирует запросы?**  
A: Увеличьте delay в parser.py до 5 секунд

## 🎯 Что дальше?

1. Соберите данные
2. Обучите модели
3. Опубликуйте на GitHub
4. Добавьте в портфолио

---

**Проект готов к использованию!** 🎉
