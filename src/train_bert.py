"""
Обучение BERT модели для детекции фейковых отзывов
Используем rubert-tiny2 для русского языка
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewDataset(Dataset):
    """Dataset для отзывов"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTTrainer:
    """Класс для обучения BERT"""
    
    def __init__(self, model_name: str = 'cointegrated/rubert-tiny2'):
        """
        Args:
            model_name: Название модели из HuggingFace
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """Подготовка данных"""
        
        texts = df['text'].values
        labels = df['label'].values
        
        # Разделение на train/val
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        
        # Создание datasets
        train_dataset = ReviewDataset(X_train, y_train, self.tokenizer)
        val_dataset = ReviewDataset(X_val, y_val, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, pred):
        """Вычисление метрик"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        acc = accuracy_score(labels, preds)
        roc_auc = roc_auc_score(labels, probs)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def train(self, train_dataset, val_dataset, output_dir: str = '../models/bert'):
        """Обучение модели"""
        
        # Инициализация модели
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        # Параметры обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=100,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='roc_auc',
            greater_is_better=True,
            save_total_limit=2,
            seed=42
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        logger.info("Начинаем обучение BERT...")
        self.trainer.train()
        
        logger.info("Обучение завершено!")
    
    def evaluate(self, test_dataset):
        """Оценка модели"""
        
        if self.trainer is None:
            raise ValueError("Модель не обучена")
        
        logger.info("Оценка модели на тестовой выборке...")
        results = self.trainer.evaluate(test_dataset)
        
        logger.info("\n=== Результаты BERT ===")
        for key, value in results.items():
            logger.info(f"{key}: {value:.4f}")
        
        return results
    
    def predict(self, texts):
        """Предсказание для новых текстов"""
        
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        dataset = ReviewDataset(
            texts, 
            [0] * len(texts),  # Dummy labels
            self.tokenizer
        )
        
        predictions = self.trainer.predict(dataset)
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)
        
        return probs[:, 1].numpy()  # Вероятность класса "фейк"
    
    def save_model(self, path: str):
        """Сохранить модель"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Модель сохранена в {path}")
    
    def load_model(self, path: str):
        """Загрузить модель"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Модель загружена из {path}")


if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_json('../data/processed/reviews_labeled.json')
    
    logger.info(f"Загружено {len(df)} отзывов")
    logger.info(f"Распределение классов:\n{df['label'].value_counts()}")
    
    # Инициализация trainer
    bert_trainer = BERTTrainer(model_name='cointegrated/rubert-tiny2')
    
    # Подготовка данных
    train_dataset, val_dataset = bert_trainer.prepare_data(df, test_size=0.2)
    
    # Обучение
    bert_trainer.train(train_dataset, val_dataset)
    
    # Оценка
    results = bert_trainer.evaluate(val_dataset)
    
    # Сохранение
    bert_trainer.save_model('../models/bert_model')
