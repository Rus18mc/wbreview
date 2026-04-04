"""
Streamlit дашборд для детекции фейковых отзывов
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import sys
sys.path.append('src')

from features import FeatureEngineering


# Настройка страницы
st.set_page_config(
    page_title="Детектор фейковых отзывов WB",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Детектор накрученных отзывов Wildberries")
st.markdown("Система автоматического определения фейковых отзывов на основе ML")


@st.cache_resource
def load_models():
    """Загрузка моделей"""
    
    # Random Forest
    try:
        with open('models/rf_model.pkl', 'rb') as f:
            rf_data = pickle.load(f)
        rf_model = rf_data['model']
        rf_features = rf_data['feature_columns']
    except:
        rf_model = None
        rf_features = None
        st.warning("Random Forest модель не найдена")
    
    # BERT
    try:
        bert_model = AutoModelForSequenceClassification.from_pretrained('models/bert_model')
        bert_tokenizer = AutoTokenizer.from_pretrained('models/bert_model')
    except:
        bert_model = None
        bert_tokenizer = None
        st.warning("BERT модель не найдена")
    
    return rf_model, rf_features, bert_model, bert_tokenizer


def predict_rf(text, rating, rf_model, rf_features):
    """Предсказание Random Forest"""
    
    if rf_model is None:
        return None, None
    
    # Создаем временный DataFrame
    temp_df = pd.DataFrame([{
        'text': text,
        'rating': rating,
        'date': pd.Timestamp.now(),
        'likes': 0,
        'dislikes': 0,
        'has_photo': False,
        'label': 0  # dummy
    }])
    
    # Создаем признаки
    fe = FeatureEngineering()
    temp_df = fe.create_all_features(temp_df)
    
    # Предсказание
    X = temp_df[rf_features].fillna(0)
    proba = rf_model.predict_proba(X)[0, 1]
    pred = rf_model.predict(X)[0]
    
    # Важность признаков для этого отзыва
    feature_values = X.iloc[0].to_dict()
    
    return pred, proba, feature_values


def predict_bert(text, bert_model, bert_tokenizer):
    """Предсказание BERT"""
    
    if bert_model is None or bert_tokenizer is None:
        return None, None
    
    # Токенизация
    encoding = bert_tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Предсказание
    with torch.no_grad():
        outputs = bert_model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)
        proba = probs[0, 1].item()
        pred = 1 if proba > 0.5 else 0
    
    return pred, proba


# Загрузка моделей
rf_model, rf_features, bert_model, bert_tokenizer = load_models()


# Интерфейс
st.sidebar.header("Настройки")
model_choice = st.sidebar.selectbox(
    "Выберите модель",
    ["Random Forest", "BERT", "Ансамбль"]
)

st.header("Введите отзыв для проверки")

col1, col2 = st.columns([3, 1])

with col1:
    review_text = st.text_area(
        "Текст отзыва",
        height=150,
        placeholder="Введите текст отзыва..."
    )

with col2:
    rating = st.slider("Рейтинг", 1, 5, 5)

if st.button("🔍 Проверить отзыв", type="primary"):
    
    if not review_text:
        st.error("Введите текст отзыва!")
    else:
        with st.spinner("Анализируем отзыв..."):
            
            results = {}
            
            # Random Forest
            if model_choice in ["Random Forest", "Ансамбль"] and rf_model:
                rf_pred, rf_proba, feature_vals = predict_rf(
                    review_text, rating, rf_model, rf_features
                )
                results['rf'] = {'pred': rf_pred, 'proba': rf_proba}
            
            # BERT
            if model_choice in ["BERT", "Ансамбль"] and bert_model:
                bert_pred, bert_proba = predict_bert(
                    review_text, bert_model, bert_tokenizer
                )
                results['bert'] = {'pred': bert_pred, 'proba': bert_proba}
            
            # Ансамбль
            if model_choice == "Ансамбль" and 'rf' in results and 'bert' in results:
                ensemble_proba = (results['rf']['proba'] + results['bert']['proba']) / 2
                ensemble_pred = 1 if ensemble_proba > 0.5 else 0
                results['ensemble'] = {'pred': ensemble_pred, 'proba': ensemble_proba}
            
            # Отображение результатов
            st.markdown("---")
            st.header("📊 Результаты анализа")
            
            # Основной результат
            if model_choice == "Ансамбль" and 'ensemble' in results:
                final_proba = results['ensemble']['proba']
                final_pred = results['ensemble']['pred']
            elif model_choice == "Random Forest" and 'rf' in results:
                final_proba = results['rf']['proba']
                final_pred = results['rf']['pred']
            elif model_choice == "BERT" and 'bert' in results:
                final_proba = results['bert']['proba']
                final_pred = results['bert']['pred']
            else:
                st.error("Модель не загружена")
                st.stop()
            
            # Визуализация
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Вероятность фейка",
                    f"{final_proba*100:.1f}%"
                )
            
            with col2:
                verdict = "🚨 ФЕЙК" if final_pred == 1 else "✅ РЕАЛЬНЫЙ"
                st.metric("Вердикт", verdict)
            
            with col3:
                confidence = "Высокая" if abs(final_proba - 0.5) > 0.3 else "Средняя"
                st.metric("Уверенность", confidence)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Вероятность фейка (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if final_proba > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Детали по моделям
            if model_choice == "Ансамбль":
                st.subheader("Детали по моделям")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'rf' in results:
                        st.info(f"**Random Forest**: {results['rf']['proba']*100:.1f}%")
                
                with col2:
                    if 'bert' in results:
                        st.info(f"**BERT**: {results['bert']['proba']*100:.1f}%")
            
            # Признаки (для RF)
            if model_choice in ["Random Forest", "Ансамбль"] and 'rf' in results:
                st.subheader("🔍 Ключевые признаки")
                
                # Топ-5 признаков
                top_features = {
                    'Длина текста': feature_vals.get('text_length', 0),
                    'Количество слов': feature_vals.get('word_count', 0),
                    'Восклицательные знаки': feature_vals.get('exclamation_count', 0),
                    'Ключевые слова фейков': feature_vals.get('fake_keywords_count', 0),
                    'Рейтинг': rating
                }
                
                df_features = pd.DataFrame(list(top_features.items()), 
                                          columns=['Признак', 'Значение'])
                st.dataframe(df_features, use_container_width=True)


# Боковая панель с информацией
st.sidebar.markdown("---")
st.sidebar.header("О проекте")
st.sidebar.info("""
Этот детектор использует машинное обучение для определения накрученных отзывов на Wildberries.

**Модели:**
- Random Forest (30+ признаков)
- BERT (rubert-tiny2)
- Ансамбль обеих моделей

**Признаки фейка:**
- Короткий текст (<150 символов)
- Максимальный рейтинг (5)
- Шаблонные фразы
- Отсутствие конкретики
""")

st.sidebar.markdown("---")
st.sidebar.caption("Проект для портфолио Data Science")
