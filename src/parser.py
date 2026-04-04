"""
Парсер отзывов с Wildberries
Собирает отзывы через API для указанных товаров (nmId)
"""

import requests
import json
import time
from typing import List, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WBReviewParser:
    """Парсер отзывов с Wildberries"""
    
    def __init__(self, delay: float = 2.0):
        """
        Args:
            delay: Задержка между запросами в секундах
        """
        self.delay = delay
        self.base_url = "https://feedbacks1.wb.ru/feedbacks/v1"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
    
    def get_reviews_for_product(self, nm_id: int, max_reviews: int = 500) -> List[Dict]:
        """
        Получить отзывы для одного товара
        
        Args:
            nm_id: ID товара на WB
            max_reviews: Максимальное количество отзывов для сбора
            
        Returns:
            Список отзывов
        """
        reviews = []
        skip = 0
        take = 30  # Количество отзывов за один запрос
        
        logger.info(f"Начинаем сбор отзывов для товара {nm_id}")
        
        while len(reviews) < max_reviews:
            try:
                url = f"{self.base_url}/{nm_id}"
                params = {
                    'skip': skip,
                    'take': take
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"Статус {response.status_code} для товара {nm_id}")
                    break
                
                data = response.json()
                
                if 'feedbacks' not in data or not data['feedbacks']:
                    logger.info(f"Больше нет отзывов для товара {nm_id}")
                    break
                
                batch = data['feedbacks']
                
                for review in batch:
                    reviews.append({
                        'nmId': nm_id,
                        'id': review.get('id'),
                        'text': review.get('text', ''),
                        'rating': review.get('productValuation', 0),
                        'date': review.get('createdDate'),
                        'likes': review.get('feedbackValuation', {}).get('positive', 0),
                        'dislikes': review.get('feedbackValuation', {}).get('negative', 0),
                        'has_photo': len(review.get('photoLinks', [])) > 0,
                        'user_name': review.get('userName', ''),
                    })
                
                skip += take
                logger.info(f"Собрано {len(reviews)} отзывов для товара {nm_id}")
                
                # Задержка между запросами
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Ошибка при сборе отзывов для {nm_id}: {e}")
                break
        
        return reviews
    
    def parse_multiple_products(self, nm_ids: List[int], max_per_product: int = 200) -> List[Dict]:
        """
        Собрать отзывы для нескольких товаров
        
        Args:
            nm_ids: Список ID товаров
            max_per_product: Максимум отзывов на товар
            
        Returns:
            Список всех отзывов
        """
        all_reviews = []
        
        for nm_id in tqdm(nm_ids, desc="Парсинг товаров"):
            reviews = self.get_reviews_for_product(nm_id, max_per_product)
            all_reviews.extend(reviews)
            logger.info(f"Всего собрано: {len(all_reviews)} отзывов")
        
        return all_reviews
    
    def save_to_json(self, reviews: List[Dict], filepath: str):
        """Сохранить отзывы в JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        logger.info(f"Сохранено {len(reviews)} отзывов в {filepath}")


if __name__ == "__main__":
    # Пример использования
    # Эти nmId нужно получить со страницы продавца
    # https://www.wildberries.ru/seller/1177893
    
    # Для демо используем несколько товаров (нужно заменить на реальные)
    nm_ids = [
        # Здесь должны быть реальные nmId товаров продавца 1177893
        # Их можно найти через DevTools на странице продавца
    ]
    
    parser = WBReviewParser(delay=2.0)
    
    # Если nm_ids пустой, выводим инструкцию
    if not nm_ids:
        print("""
        Инструкция по получению nmId товаров:
        1. Откройте https://www.wildberries.ru/seller/1177893
        2. Нажмите F12 (DevTools)
        3. Перейдите на вкладку Network
        4. Обновите страницу
        5. Найдите запросы к API (catalog, products)
        6. В ответах найдите поле nmId для товаров
        7. Добавьте их в список nm_ids выше
        """)
    else:
        reviews = parser.parse_multiple_products(nm_ids, max_per_product=200)
        parser.save_to_json(reviews, '../data/raw/reviews_raw.json')
