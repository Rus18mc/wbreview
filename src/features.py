"""
Инженерия признаков для детекции фейковых отзывов
Создает 30+ признаков на основе текста и метаданных
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List, Dict
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Класс для создания признаков"""
    
    def __init__(self):
        self.fake_keywords = [
            'отлично', 'супер', 'рекомендую', 'качество', 
            'доставка', 'быстро', 'огонь', 'топ', 'класс',
            'прекрасно', 'замечательно', 'великолепно'
        ]
        
        self.real_keywords = [
            'сломался', 'брак', 'вернул', 'разочарован',
            'неделя', 'месяц', 'день', 'использую', 'пользуюсь'
        ]
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Поведенческие признаки на основе метаданных"""
        
        # Преобразуем дату
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Час отправки
        df['hour'] = df['date'].dt.hour
        df['is_night'] = (df['hour'] >= 0) & (df['hour'] <= 5)
        df['is_night'] = df['is_night'].astype(int)
        
        # День недели
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Метрики лайков
        df['likes_ratio'] = df['likes'] / (df['likes'] + df['dislikes'] + 1)
        df['has_likes'] = (df['likes'] > 0).astype(int)
        df['has_dislikes'] = (df['dislikes'] > 0).astype(int)
        
        # Фото
        df['has_photo'] = df['has_photo'].astype(int)
        
        return df
    
    def extract_text_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Базовые текстовые признаки"""
        
        # Длина текста
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
        
        # Пунктуация
        df['exclamation_count'] = df['text'].str.count('!')
        df['question_count'] = df['text'].str.count('\?')
        df['exclamation_ratio'] = df['exclamation_count'] / (df['text_length'] + 1)
        
        # Заглавные буквы
        df['uppercase_count'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
        df['uppercase_ratio'] = df['uppercase_count'] / (df['text_length'] + 1)
        
        # Цифры
        df['digit_count'] = df['text'].str.count(r'\d')
        df['has_digits'] = (df['digit_count'] > 0).astype(int)
        
        # Эмодзи и спецсимволы
        df['emoji_count'] = df['text'].apply(self._count_emojis)
        
        return df
    
    def extract_linguistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Лингвистические признаки"""
        
        # Уникальность слов
        df['unique_words_ratio'] = df['text'].apply(self._unique_words_ratio)
        
        # Повторяющиеся слова
        df['repetitive_words_ratio'] = df['text'].apply(self._repetitive_words_ratio)
        
        # Ключевые слова для фейков
        df['fake_keywords_count'] = df['text'].apply(
            lambda x: sum(1 for kw in self.fake_keywords if kw in x.lower())
        )
        df['fake_keywords_ratio'] = df['fake_keywords_count'] / (df['word_count'] + 1)
        
        # Ключевые слова для реальных отзывов
        df['real_keywords_count'] = df['text'].apply(
            lambda x: sum(1 for kw in self.real_keywords if kw in x.lower())
        )
        
        # Средняя длина предложений
        df['sentence_count'] = df['text'].str.count(r'[.!?]+')
        df['avg_sentence_length'] = df['word_count'] / (df['sentence_count'] + 1)
        
        return df
    
    def extract_template_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Признаки шаблонности"""
        
        # N-граммы
        df['repetitive_bigrams'] = df['text'].apply(self._repetitive_ngrams, n=2)
        df['repetitive_trigrams'] = df['text'].apply(self._repetitive_ngrams, n=3)
        
        # Начало и конец отзыва
        df['starts_with_positive'] = df['text'].apply(
            lambda x: any(x.lower().startswith(kw) for kw in self.fake_keywords)
        ).astype(int)
        
        df['ends_with_recommend'] = df['text'].str.lower().str.contains(
            'рекомендую|советую', regex=True
        ).astype(int)
        
        return df
    
    def extract_rating_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Признаки на основе рейтинга"""
        
        df['is_max_rating'] = (df['rating'] == 5).astype(int)
        df['is_min_rating'] = (df['rating'] == 1).astype(int)
        df['is_extreme_rating'] = ((df['rating'] == 1) | (df['rating'] == 5)).astype(int)
        
        # Несоответствие рейтинга и текста
        df['rating_text_mismatch'] = df.apply(self._rating_text_mismatch, axis=1)
        
        return df
    
    def _count_emojis(self, text: str) -> int:
        """Подсчет эмодзи"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        return len(emoji_pattern.findall(text))
    
    def _unique_words_ratio(self, text: str) -> float:
        """Доля уникальных слов"""
        words = text.lower().split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def _repetitive_words_ratio(self, text: str) -> float:
        """Доля повторяющихся слов"""
        words = text.lower().split()
        if len(words) == 0:
            return 0
        word_counts = Counter(words)
        repeated = sum(1 for count in word_counts.values() if count > 1)
        return repeated / len(word_counts)
    
    def _repetitive_ngrams(self, text: str, n: int = 2) -> float:
        """Доля повторяющихся n-грамм"""
        words = text.lower().split()
        if len(words) < n:
            return 0
        
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        if len(ngrams) == 0:
            return 0
        
        ngram_counts = Counter(ngrams)
        most_common_count = ngram_counts.most_common(1)[0][1] if ngram_counts else 0
        return most_common_count / len(ngrams)
    
    def _rating_text_mismatch(self, row) -> int:
        """Несоответствие рейтинга и тональности текста"""
        text_lower = row['text'].lower()
        
        # Высокий рейтинг, но негативные слова
        if row['rating'] >= 4:
            negative_words = ['плохо', 'ужасно', 'разочарован', 'не советую', 'брак']
            if any(word in text_lower for word in negative_words):
                return 1
        
        # Низкий рейтинг, но позитивные слова
        if row['rating'] <= 2:
            if any(word in text_lower for word in self.fake_keywords):
                return 1
        
        return 0
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создать все признаки"""
        logger.info("Создание признаков...")
        
        df = df.copy()
        
        # Заполняем пропуски
        df['text'] = df['text'].fillna('')
        df['likes'] = df['likes'].fillna(0)
        df['dislikes'] = df['dislikes'].fillna(0)
        
        # Создаем признаки
        df = self.extract_behavioral_features(df)
        df = self.extract_text_basic_features(df)
        df = self.extract_linguistic_features(df)
        df = self.extract_template_features(df)
        df = self.extract_rating_features(df)
        
        logger.info(f"Создано {len(df.columns)} признаков")
        
        return df


if __name__ == "__main__":
    # Пример использования
    import json
    
    # Загрузка данных
    with open('../data/processed/reviews_labeled.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Создание признаков
    fe = FeatureEngineering()
    df_features = fe.create_all_features(df)
    
    # Сохранение
    df_features.to_csv('../data/features/features.csv', index=False)
    logger.info(f"Признаки сохранены. Форма: {df_features.shape}")
