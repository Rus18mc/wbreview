#!/bin/bash

# Скрипт для быстрого запуска проекта

echo "=== Детектор фейковых отзывов Wildberries ==="
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.9+"
    exit 1
fi

echo "✓ Python найден: $(python3 --version)"

# Создание виртуального окружения
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активация виртуального окружения
echo "Активация виртуального окружения..."
source venv/bin/activate

# Установка зависимостей
echo "Установка зависимостей..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✓ Окружение готово!"
echo ""
echo "Доступные команды:"
echo "  1. python src/parser.py          - Парсинг отзывов"
echo "  2. python src/labeler.py         - Разметка данных"
echo "  3. python src/features.py        - Создание признаков"
echo "  4. python src/train_rf.py        - Обучение Random Forest"
echo "  5. python src/train_bert.py      - Обучение BERT"
echo "  6. streamlit run streamlit_app.py - Запуск дашборда"
echo ""
echo "Для запуска Jupyter:"
echo "  jupyter notebook notebooks/"
echo ""
