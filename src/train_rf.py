"""
Обучение Random Forest модели для детекции фейковых отзывов
"""

import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestTrainer:
    """Класс для обучения Random Forest"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.best_params = None
    
    def prepare_features(self, df: pd.DataFrame):
        """Подготовка признаков для обучения"""
        
        # Исключаем нецифровые колонки
        exclude_cols = ['text', 'nmId', 'date', 'label']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].fillna(0)
        y = df['label']
        
        logger.info(f"Подготовлено {len(feature_cols)} признаков")
        logger.info(f"Размер выборки: {len(X)}")
        logger.info(f"Распределение классов: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train(self, X_train, y_train, use_grid_search: bool = True):
        """Обучение модели"""
        
        if use_grid_search:
            logger.info("Запуск GridSearchCV...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced']
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            grid_search = GridSearchCV(
                rf, 
                param_grid, 
                cv=StratifiedKFold(n_splits=3),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            logger.info(f"Лучшие параметры: {self.best_params}")
            logger.info(f"Лучший ROC-AUC: {grid_search.best_score_:.4f}")
        else:
            logger.info("Обучение с параметрами по умолчанию...")
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Метрики
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        logger.info("\n=== Результаты Random Forest ===")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, 
                                         target_names=['Real', 'Fake']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def get_feature_importance(self, top_n: int = 20):
        """Получить важность признаков"""
        
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Сохранить модель"""
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузить модель"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.best_params = model_data.get('best_params')
        
        logger.info(f"Модель загружена из {filepath}")


if __name__ == "__main__":
    # Загрузка данных с признаками
    df = pd.read_csv('../data/features/features.csv')
    
    # Подготовка данных
    trainer = RandomForestTrainer()
    X, y = trainer.prepare_features(df)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Обучение
    trainer.train(X_train, y_train, use_grid_search=True)
    
    # Оценка
    metrics = trainer.evaluate(X_test, y_test)
    
    # Важность признаков
    importance = trainer.get_feature_importance(top_n=20)
    logger.info(f"\nTop 20 важных признаков:\n{importance}")
    
    # Сохранение
    trainer.save_model('../models/rf_model.pkl')
