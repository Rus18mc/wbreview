"""
Разметка отзывов на фейковые и реальные
Использует эвристические правила
"""

import pandas as pd
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewLabeler:
    """Класс для разметки отзывов"""
    
    def __init__(self):
        self.fake_keywords = [
            'отлично', 'супер', 'рекомендую', 'качество', 
            'доставка', 'быстро', 'огонь', 'топ'
        ]
        
        self.real_indicators = [
            'сломался', 'брак', 'вернул', 'разочарован',
            'неделя', 'месяц', 'день', 'использую'
        ]
    
    def label_review(self, review: Dict) -> int:
        """
        Разметить один отзыв
        
        Returns:
            1 - фейк
            0 - реальный
            -1 - серый (исключить из обучения)
        """
        text = review.get('text', '').lower()
        rating = review.get('rating', 0)
        text_length = len(text)
        
        # Правило 1: Фейк
        # Рейтинг 5 + короткий текст + ключевые слова
        if rating == 5:
            keyword_count = sum(1 for kw in self.fake_keywords if kw in text)
            
            if keyword_count >= 2 and text_length < 150:
                return 1  # Фейк
            
            # Очень короткий восторженный отзыв
            if text_length < 50 and keyword_count >= 1:
                return 1  # Фейк
        
        # Правило 2: Реальный
        # Низкий рейтинг
        if rating <= 3:
            return 0  # Реальный
        
        # Содержит конкретику
        has_real_indicators = any(indicator in text for indicator in self.real_indicators)
        has_digits = any(char.isdigit() for char in text)
        
        if has_real_indicators or (has_digits and text_length > 100):
            return 0  # Реальный
        
        # Длинный детальный отзыв
        if text_length > 300:
            return 0  # Реальный
        
        # Правило 3: Серый (неопределенный)
        return -1
    
    def label_dataset(self, reviews: List[Dict]) -> pd.DataFrame:
        """
        Разметить весь датасет
        
        Returns:
            DataFrame с колонками: text, label, nmId, rating, date
        """
        logger.info(f"Начинаем разметку {len(reviews)} отзывов...")
        
        labeled_data = []
        
        for review in reviews:
            label = self.label_review(review)
            
            labeled_data.append({
                'text': review.get('text', ''),
                'label': label,
                'nmId': review.get('nmId'),
                'rating': review.get('rating'),
                'date': review.get('date'),
                'likes': review.get('likes', 0),
                'dislikes': review.get('dislikes', 0),
                'has_photo': review.get('has_photo', False)
            })
        
        df = pd.DataFrame(labeled_data)
        
        # Статистика
        fake_count = (df['label'] == 1).sum()
        real_count = (df['label'] == 0).sum()
        grey_count = (df['label'] == -1).sum()
        
        logger.info(f"Разметка завершена:")
        logger.info(f"  Фейковых: {fake_count} ({fake_count/len(df)*100:.1f}%)")
        logger.info(f"  Реальных: {real_count} ({real_count/len(df)*100:.1f}%)")
        logger.info(f"  Серых (исключены): {grey_count} ({grey_count/len(df)*100:.1f}%)")
        
        # Удаляем серые
        df_clean = df[df['label'] != -1].copy()
        logger.info(f"Итоговый датасет: {len(df_clean)} отзывов")
        
        return df_clean
    
    def save_labeled_data(self, df: pd.DataFrame, filepath: str):
        """Сохранить размеченные данные"""
        df.to_json(filepath, orient='records', force_ascii=False, indent=2)
        logger.info(f"Данные сохранены в {filepath}")


if __name__ == "__main__":
    # Загрузка сырых данных
    with open('../data/raw/reviews_raw.json', 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    # Разметка
    labeler = ReviewLabeler()
    df_labeled = labeler.label_dataset(reviews)
    
    # Сохранение
    labeler.save_labeled_data(df_labeled, '../data/processed/reviews_labeled.json')
