# models/demand_predictor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
import catboost as cb
from datetime import datetime, timedelta
import joblib
from dataclasses import dataclass


@dataclass
class ProductInfo:
    """商品信息"""
    product_code: str
    product_name: str
    category: str
    price: float
    cost: float
    weight: float
    is_fresh: bool
    shelf_life_hours: int


class EnhancedDemandPredictor:
    """增强版需求预测模型 - 修复groupby错误"""

    def __init__(self, model_type: str = 'ensemble', config=None):
        """
        初始化需求预测模型
        """
        self.model_type = model_type
        self.config = config
        self.model = None
        self.feature_columns = []
        self.scaler = None
        self.poly_features = None
        self.product_cache = {}

        # 模型参数
        self.model_params = {
            'xgboost': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            },
            'lightgbm': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'random_state': 42
            }
        }

    def prepare_training_data_from_transactions(self, transaction_data: pd.DataFrame,
                                                product_code: str,
                                                promotion_hours: Tuple[int, int] = (20, 22),
                                                store_code: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """从交易数据准备训练数据 - 修复groupby错误"""

        # 筛选特定商品的数据
        if store_code:
            product_data = transaction_data[
                (transaction_data['商品编码'] == product_code) &
                (transaction_data['门店编码'] == store_code)
                ].copy()
        else:
            product_data = transaction_data[transaction_data['商品编码'] == product_code].copy()

        if product_data.empty:
            raise ValueError(f"商品 {product_code} 没有历史数据")

        # 确保有时间字段
        if '交易时间' not in product_data.columns:
            if '日期' in product_data.columns:
                product_data['交易时间'] = pd.to_datetime(product_data['日期'])
            else:
                raise ValueError("没有时间字段用于训练")

        # 确保交易时间是datetime类型
        # if not pd.api.types.is_datetime64_any_dtype(product_data['交易时间']):
        #     product_data['交易时间'] = pd.to_datetime(product_data['交易时间'])

        # 按时间窗口聚合（每30分钟）
        product_data['时间窗口'] = product_data['交易时间'].dt.floor('30min')

        # 修复groupby聚合 - 避免nested renamer错误
        agg_dict = {
            '销售数量': 'sum',
            '平均售价': 'mean'
        }

        # 添加可选列
        if '实际折扣率' in product_data.columns:
            agg_dict['实际折扣率'] = 'mean'
        else:
            # 如果没有实际折扣率列，创建一个默认列
            product_data['实际折扣率'] = 1.0
            agg_dict['实际折扣率'] = 'mean'

        if '是否折扣' in product_data.columns:
            agg_dict['是否折扣'] = 'mean'
        else:
            # 创建默认的是否折扣列
            product_data['是否折扣'] = 0.0
            agg_dict['是否折扣'] = 'mean'

        try:
            # 使用修复的agg字典
            grouped = product_data.groupby('时间窗口').agg(agg_dict).reset_index()
        except Exception as e:
            # 如果仍然出错，使用更简单的聚合方式
            print(f"警告: 标准groupby失败, 使用替代方法: {e}")
            grouped = pd.DataFrame({
                '时间窗口': product_data['时间窗口'].unique(),
                '销售数量': product_data.groupby('时间窗口')['销售数量'].sum().values,
                '平均售价': product_data.groupby('时间窗口')['平均售价'].mean().values
            })

            if '实际折扣率' in product_data.columns:
                grouped['实际折扣率'] = product_data.groupby('时间窗口')['实际折扣率'].mean().values
            else:
                grouped['实际折扣率'] = 1.0

            if '是否折扣' in product_data.columns:
                grouped['是否折扣'] = product_data.groupby('时间窗口')['是否折扣'].mean().values
            else:
                grouped['是否折扣'] = 0.0

        # 提取特征
        features = []
        targets = []

        for _, row in grouped.iterrows():
            timestamp = row['时间窗口']

            # 基础特征
            feature_vector = self._extract_features_from_timestamp(
                timestamp, row, promotion_hours
            )

            features.append(feature_vector)
            targets.append(row['销售数量'])

        features_df = pd.DataFrame(features)
        targets_series = pd.Series(targets, name='sales_quantity')

        return features_df, targets_series

    def _extract_features_from_timestamp(self, timestamp: pd.Timestamp,
                                         row: pd.Series,
                                         promotion_hours: Tuple[int, int]) -> Dict:
        """从时间戳提取特征"""

        # 时间特征
        year = timestamp.year
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()
        month = timestamp.month
        day_of_month = timestamp.day
        is_weekend = 1 if day_of_week >= 5 else 0

        # 促销时间特征
        promo_start, promo_end = promotion_hours
        in_promotion = 1 if promo_start <= hour < promo_end else 0
        time_to_promo_end = max(0, promo_end - hour - minute / 60)

        # 周期性特征
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        # 是否为特殊时段
        is_morning_rush = 1 if 7 <= hour < 9 else 0
        is_lunch_time = 1 if 11 <= hour < 13 else 0
        is_evening_rush = 1 if 17 <= hour < 19 else 0
        is_night = 1 if hour >= 22 or hour < 6 else 0

        # 价格特征
        price = row.get('平均售价', 100.0)
        discount_rate = row.get('实际折扣率', 1.0)
        has_discount = row.get('是否折扣', 0.0)

        feature_vector = {
            "year": year,
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'month': month,
            'day_of_month': day_of_month,
            'weekend': is_weekend,
            'in_promotion': in_promotion,
            'time_to_promo_end': time_to_promo_end,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'is_morning_rush': is_morning_rush,
            'is_lunch_time': is_lunch_time,
            'is_evening_rush': is_evening_rush,
            'is_night': is_night,
            'price': price,
            'discount_rate': discount_rate,
            'has_discount': has_discount,
            'price_ratio': 1.0 / discount_rate if discount_rate > 0 else 1.0
        }

        return feature_vector

    def train(self, X: pd.DataFrame, y: pd.Series,
              product_info: Optional[ProductInfo] = None):
        """训练模型"""
        self.feature_columns = X.columns.tolist()

        if product_info:
            self.product_cache[product_info.product_code] = product_info

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
        X.to_csv("data/X.csv", encoding="utf-8")
        self.model.fit(X, y)
    def predict_train_set(self,  X: pd.DataFrame):
        """预测训练集"""
        # 确保X有正确的列顺序
        if hasattr(self, 'feature_columns'):
            # 确保X包含所有训练时的特征列
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0  # 或使用其他合适的默认值

            # 重新排序列以匹配训练时的顺序
            X = X[self.feature_columns]

        if self.model_type == 'ensemble':
            # 集成模型预测
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)

            # 计算平均预测值
            if len(predictions) > 0:
                # 对每个模型的预测结果进行平均
                avg_predictions = np.mean(predictions, axis=0)
            else:
                avg_predictions = np.array([])

            # 处理最终预测结果
            if len(avg_predictions) == 1:
                prediction = max(0.1, avg_predictions[0])
            else:
                prediction = np.maximum(0.1, avg_predictions)
        else:
            # 单个模型预测
            if self.model is None:
                raise ValueError("模型尚未训练")

            pred_result = self.model.predict(X)

            # 根据输出形状处理
            if pred_result.ndim == 0 or len(pred_result) == 1:
                prediction = max(0.1, pred_result if pred_result.ndim == 0 else pred_result[0])
            else:
                prediction = np.maximum(0.1, pred_result)

        print(f"预测结果形状: {prediction.shape if hasattr(prediction, 'shape') else 'scalar'}")
        return prediction
    def predict_demand(self, features: Dict,
                       discount_rate: float,
                       time_to_close: float,
                       current_stock: int,
                       product_info: Optional[ProductInfo] = None,
                       weather_features: Optional[Dict] = None,
                       calendar_features: Optional[Dict] = None) -> float:
        """预测需求量"""

        # 如果模型未训练，使用启发式模型
        if self.model is None and self.model_type != 'ensemble':
            return self._advanced_heuristic_prediction(
                features, discount_rate, time_to_close, current_stock,
                product_info, weather_features, calendar_features
            )
            # 可用于计算当日已售库存
        # today_saled_stock = product_info[
        #     (product_info['日期'] == current_time.date())]['销售数量'].sum()
        # # self.product_cache
        # total_stock = current_stock + today_saled_stock
        # 准备特征向量
        feature_vector = self._create_complete_feature_vector(
            features, discount_rate, time_to_close, current_stock,
            product_info, weather_features, calendar_features,
        )

        # 转换为DataFrame
        feature_df = pd.DataFrame([feature_vector])

        # 确保特征顺序
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(feature_df.columns)
            for col in missing_cols:
                feature_df[col] = 0
            feature_df = feature_df[self.feature_columns]
        print("预测销量的特征信息：", feature_df)
        # 预测
        if self.model_type == 'ensemble':
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(feature_df)
                predictions.append(pred[0])
            prediction = np.mean(predictions)
        else:
            prediction = self.model.predict(feature_df)[0]

        # 应用约束：不能超过当前库存，不能为负
        # prediction = max(0.1, min(prediction, current_stock * 0.8))
        prediction = max(0.1, prediction)
        print("预测销量：", prediction)
        return prediction

    def _advanced_heuristic_prediction(self, features: Dict,
                                       discount_rate: float,
                                       time_to_close: float,
                                       current_stock: int,
                                       product_info: Optional[ProductInfo],
                                       weather_features: Optional[Dict],
                                       calendar_features: Optional[Dict]) -> float:
        """高级启发式需求预测"""

        # 基础需求
        base_demand = features.get('hist_avg_sales', 10.0)

        # 1. 价格弹性效应
        price_elasticity = features.get('price_elasticity', 1.2)
        price_factor = (1.0 / discount_rate) ** price_elasticity

        # 2. 时间压力效应（越接近关店时间，需求刺激越大）
        # 使用指数衰减函数
        time_factor = 1.0 + np.exp(3 * (1.0 - time_to_close)) - 1

        # 3. 库存压力效应（库存越多，打折效果越好）
        # 使用S型函数
        stock_ratio = current_stock / max(features.get('hist_avg_sales', 10) * 3, 10)
        stock_factor = 1.0 + 2.0 / (1.0 + np.exp(-5 * (stock_ratio - 0.5)))

        # 4. 促销时段效应
        promo_ratio = features.get('hist_promo_sales_ratio', 1.2)
        promo_factor = 1.0 + (promo_ratio - 1.0) * (1.0 - time_to_close)

        # 5. 天气影响
        weather_factor = 1.0
        if weather_features:
            # 温度影响
            temp = weather_features.get('temperature', 25)
            if temp > 30:
                weather_factor *= 0.9  # 太热减少需求
            elif 20 <= temp <= 25:
                weather_factor *= 1.1  # 舒适温度增加需求

            # 雨天影响
            if weather_features.get('is_rainy', 0):
                if product_info and product_info.is_fresh:
                    weather_factor *= 0.8  # 生鲜雨天需求减少
                else:
                    weather_factor *= 1.2  # 其他商品雨天需求增加

        # 6. 日历影响
        calendar_factor = 1.0
        if calendar_features:
            if calendar_features.get('is_holiday', 0):
                calendar_factor *= 1.3  # 节假日增加需求
            if calendar_features.get('is_weekend', 0):
                calendar_factor *= 1.2  # 周末增加需求
            if calendar_features.get('is_shopping_day', 0):
                calendar_factor *= 1.5  # 购物节大幅增加需求

        # 7. 商品特性影响
        product_factor = 1.0
        if product_info:
            if product_info.is_fresh:
                # 生鲜商品时间敏感性更强
                product_factor *= 1.0 + (1.0 - time_to_close) * 2.0
            if product_info.shelf_life_hours < 24:
                # 短保质期商品更敏感
                product_factor *= 1.5

        # 8. 销售趋势影响
        sales_trend = features.get('sales_trend', 0)
        trend_factor = 1.0 + sales_trend

        # 综合预测
        predicted_demand = (
                base_demand *
                price_factor *
                time_factor *
                stock_factor *
                promo_factor *
                weather_factor *
                calendar_factor *
                product_factor *
                trend_factor
        )

        # 添加随机波动（实际中会有不确定性）
        predicted_demand *= np.random.uniform(0.8, 1.2)

        # 约束：不能超过当前库存，不能为负
        predicted_demand = max(0.1, min(predicted_demand, current_stock * 0.8))

        return predicted_demand

    def _create_complete_feature_vector(self, features: Dict,
                                        discount_rate: float,
                                        time_to_close: float,
                                        current_stock: int,
                                        product_info: Optional[ProductInfo],
                                        weather_features: Optional[Dict],
                                        calendar_features: Optional[Dict],
                                        total_stock: int = 100) -> Dict:
        """创建完整的特征向量"""
        # 推测折扣销量用
        # 基础特征
        feature_vector = features.copy()

        # 添加动态特征
        feature_vector.update({
            'discount_rate': discount_rate,
            'time_to_close': time_to_close,
            'current_stock': current_stock,
            'stock_ratio': current_stock / total_stock,# max(features.get('hist_avg_sales', 100), 1),
            'price_elasticity_effect': (1.0 / discount_rate) ** features.get('price_elasticity', 1.2),
            'urgency_factor': 1.0 / (time_to_close + 0.1),  # 避免除零
            'is_high_discount': 1 if discount_rate < 0.7 else 0,
            'is_low_discount': 1 if discount_rate > 0.9 else 0
        })

        # 添加商品特征
        if product_info:
            feature_vector.update({
                'product_is_fresh': 1 if product_info.is_fresh else 0,
                'product_weight': product_info.weight,
                'shelf_life_hours': product_info.shelf_life_hours,
                'price_cost_ratio': product_info.price / max(product_info.cost, 0.1)
            })

        # 添加天气特征
        if weather_features:
            feature_vector.update(weather_features)

        # 添加日历特征
        if calendar_features:
            feature_vector.update(calendar_features)

        # 创建交互特征
        feature_vector['discount_time_interaction'] = discount_rate * time_to_close
        feature_vector['discount_stock_interaction'] = discount_rate * current_stock
        feature_vector['time_stock_interaction'] = time_to_close * current_stock

        return feature_vector