# models/demand_predictor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
import catboost as cb
from datetime import datetime
import joblib

class DemandPredictor:
    """需求预测模型"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        初始化需求预测模型
        
        Args:
            model_type: 模型类型，可选 'xgboost', 'lightgbm', 'catboost', 
                       'random_forest', 'gradient_boosting', 'linear', 'ensemble'
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.scaler = None
        self.poly_features = None
        
        # 模型参数
        self.model_params = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                # 添加以下参数来避免警告
                'min_child_samples': 20,  # 增加叶子节点最小样本数
                'min_child_weight': 0.001,  # 最小叶子权重
                'min_split_gain': 0.0,  # 设置更小的分裂增益阈值
                'reg_alpha': 0.1,  # L1正则化
                'reg_lambda': 0.1,  # L2正则化
                'verbose': -1  # 不显示警告
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        }
    
    def prepare_training_data(self, transaction_data: pd.DataFrame,
                             product_code: str,
                             promotion_hours: Tuple[int, int] = (20, 22)) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备训练数据"""
        
        # 筛选特定商品的数据
        product_data = transaction_data[transaction_data['商品编码'] == product_code].copy()
        
        if product_data.empty:
            raise ValueError(f"商品 {product_code} 没有历史数据")
        
        # 确保有时间字段
        if '交易时间' not in product_data.columns and '日期' in product_data.columns:
            product_data['交易时间'] = product_data['日期']
        
        product_data['交易时间'] = pd.to_datetime(product_data['交易时间'])
        
        # 提取特征
        features = []
        targets = []
        
        # 按时间窗口聚合（例如每30分钟）
        product_data['时间窗口'] = product_data['交易时间'].dt.floor('30min')
        
        grouped = product_data.groupby('时间窗口').agg({
            '销售数量': 'sum',
            '售价': 'mean',
            '实际折扣率': 'mean' if '实际折扣率' in product_data.columns else None
        }).reset_index()
        
        for _, row in grouped.iterrows():
            # 时间特征
            timestamp = row['时间窗口']
            hour = timestamp.hour
            minute = timestamp.minute
            day_of_week = timestamp.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # 促销时间特征
            promo_start, promo_end = promotion_hours
            in_promotion = 1 if promo_start <= hour < promo_end else 0
            time_to_promo_end = max(0, promo_end - hour - minute/60)
            
            # 特征向量
            feature_vector = {
                'hour': hour,
                'minute': minute,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'in_promotion': in_promotion,
                'time_to_promo_end': time_to_promo_end,
                'price': row['售价'] if not pd.isna(row['售价']) else 100.0,
                'discount_rate': row['实际折扣率'] if '实际折扣率' in row and not pd.isna(row['实际折扣率']) else 1.0
            }
            
            features.append(feature_vector)
            targets.append(row['销售数量'])
        
        features_df = pd.DataFrame(features)
        targets_df = pd.Series(targets, name='sales_quantity')
        
        return features_df, targets_df
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """训练模型"""
        self.feature_columns = X.columns.tolist()
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**self.model_params['xgboost'])
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**self.model_params['lightgbm'])
        elif self.model_type == 'catboost':
            self.model = cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.model_params['random_forest'])
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self.model_params['gradient_boosting'])
        elif self.model_type == 'linear':
            # 添加多项式特征
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = self.poly_features.fit_transform(X)
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_poly, y)
            return
        elif self.model_type == 'ensemble':
            # 集成模型
            self.models = {
                'xgboost': xgb.XGBRegressor(**self.model_params['xgboost']),
                'lightgbm': lgb.LGBMRegressor(**self.model_params['lightgbm']),
                'random_forest': RandomForestRegressor(**self.model_params['random_forest'])
            }
            for name, model in self.models.items():
                model.fit(X, y)
            self.model = None
            return
        
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_type == 'ensemble':
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)
            # 平均预测
            return np.mean(predictions, axis=0)
        elif self.model_type == 'linear':
            X_poly = self.poly_features.transform(X)
            return self.model.predict(X_poly)
        else:
            return self.model.predict(X)
    
    def predict_demand(self, features: Dict, 
                      discount_rate: float,
                      time_to_close: float,
                      current_stock: int,
                      base_demand: float = 10.0) -> float:
        """预测需求量"""
        
        # 如果模型未训练，使用启发式模型
        if self.model is None and self.model_type != 'ensemble':
            return self._heuristic_demand_prediction(
                discount_rate, time_to_close, current_stock, base_demand
            )
        
        # 准备特征向量
        feature_vector = self._create_feature_vector(features, discount_rate, time_to_close, current_stock)
        
        # 转换为DataFrame
        feature_df = pd.DataFrame([feature_vector])
        
        # 确保特征顺序
        if self.feature_columns:
            feature_df = feature_df[self.feature_columns]
        
        # 预测
        if self.model_type == 'ensemble':
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(feature_df)
                predictions.append(pred[0])
            prediction = np.mean(predictions)
        elif self.model_type == 'linear':
            feature_df_poly = self.poly_features.transform(feature_df)
            prediction = self.model.predict(feature_df_poly)[0]
        else:
            prediction = self.model.predict(feature_df)[0]
        
        # 应用约束：不能超过当前库存，不能为负
        prediction = max(0, min(prediction, current_stock))
        
        return prediction
    
    def _heuristic_demand_prediction(self, discount_rate: float,
                                   time_to_close: float,
                                   current_stock: int,
                                   base_demand: float) -> float:
        """启发式需求预测（当没有足够历史数据时使用）"""
        
        # 价格弹性效应
        price_factor = (1.0 / discount_rate) ** 1.2  # 假设价格弹性为1.2
        
        # 时间压力效应（越接近关店时间，需求刺激越大）
        time_factor = 1.0 + (1.0 - time_to_close) * 2.0
        
        # 库存压力效应（库存越多，打折效果越好）
        stock_factor = min(1.0 + current_stock / 100.0, 2.0)
        
        # 综合预测
        predicted_demand = base_demand * price_factor * time_factor * stock_factor
        
        # 添加随机波动
        predicted_demand *= np.random.uniform(0.8, 1.2)
        
        # 约束：不能超过当前库存，不能为负
        predicted_demand = max(0, min(predicted_demand, current_stock))
        
        return predicted_demand
    
    def _create_feature_vector(self, features: Dict,
                              discount_rate: float,
                              time_to_close: float,
                              current_stock: int) -> Dict:
        """创建特征向量"""
        
        # 基础特征
        feature_vector = features.copy()
        
        # 添加动态特征
        feature_vector.update({
            'discount_rate': discount_rate,
            'time_to_close': time_to_close,
            'current_stock': current_stock,
            'stock_ratio': current_stock / max(features.get('hist_avg_sales', 100), 1),
            'price_elasticity_effect': (1.0 / discount_rate) ** features.get('price_elasticity', 1.2),
            'urgency_factor': 1.0 / (time_to_close + 0.1)  # 避免除零
        })
        
        return feature_vector
    
    def save_model(self, path: str):
        """保存模型"""
        save_data = {
            'model_type': self.model_type,
            'model': self.model,
            'feature_columns': self.feature_columns,
            'poly_features': self.poly_features,
            'model_params': self.model_params
        }
        
        if self.model_type == 'ensemble':
            save_data['models'] = self.models
        
        joblib.dump(save_data, path)
    
    def load_model(self, path: str):
        """加载模型"""
        loaded_data = joblib.load(path)
        
        self.model_type = loaded_data['model_type']
        self.model = loaded_data['model']
        self.feature_columns = loaded_data['feature_columns']
        self.poly_features = loaded_data['poly_features']
        self.model_params = loaded_data['model_params']
        
        if self.model_type == 'ensemble':
            self.models = loaded_data['models']