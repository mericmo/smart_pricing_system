# data/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
class PricingFeatureEngineer:
    """定价特征工程"""
    
    def __init__(self):
        self.scalers = {}
        self.pca_models = {}
        self.feature_columns = []
    
    def create_features(self, transaction_data: pd.DataFrame, 
                       product_code: str, 
                       promotion_hours: Tuple[int, int],
                       current_time: datetime) -> Dict:
        """创建定价特征"""
        
        # 基础特征
        features = {}
        
        # 1. 时间特征
        current_hour = current_time.hour
        current_minute = current_time.minute
        day_of_week = current_time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        month = current_time.month
        hour_of_day = current_hour
        
        features.update({
            'hour_of_day': hour_of_day,
            'minute_of_hour': current_minute,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'month': month,
            'quarter': (month - 1) // 3 + 1
        })
        
        # 2. 促销时间特征
        promo_start, promo_end = promotion_hours
        time_to_promo_start = self._calculate_time_to_event(current_hour, promo_start)
        time_to_promo_end = self._calculate_time_to_event(current_hour, promo_end)
        in_promotion_time = 1 if promo_start <= current_hour < promo_end else 0
        
        features.update({
            'time_to_promo_start': time_to_promo_start,
            'time_to_promo_end': time_to_promo_end,
            'in_promotion_time': in_promotion_time,
            'promo_duration_hours': (promo_end - promo_start) % 24
        })
        
        # 3. 历史销售特征
        historical_features = self._extract_historical_features(
            transaction_data, product_code, promotion_hours, current_time
        )
        features.update(historical_features)
        
        # 4. 库存特征（从外部传入）
        # 这些将在外部添加
        
        # 5. 价格特征
        price_features = self._extract_price_features(transaction_data, product_code)
        features.update(price_features)
        
        # 6. 竞争环境特征（简化版）
        features.update({
            'store_traffic_index': 1.0,  # 可扩展：基于门店人流
            'competitive_pressure': 0.5,  # 可扩展：基于竞品价格
            'weather_factor': 1.0,  # 可扩展：天气影响
            'special_day': 0.0  # 可扩展：节假日
        })
        
        return features
    
    def _calculate_time_to_event(self, current_hour: int, event_hour: int) -> float:
        """计算到事件的时间距离"""
        if event_hour >= current_hour:
            return event_hour - current_hour
        else:
            return (24 - current_hour) + event_hour
    
    def _extract_historical_features(self, transaction_data: pd.DataFrame,
                                   product_code: str,
                                   promotion_hours: Tuple[int, int],
                                   current_time: datetime) -> Dict:
        """提取历史销售特征"""
        
        # 筛选商品数据
        product_data = transaction_data[transaction_data['商品编码'] == product_code].copy()
        
        if product_data.empty:
            return {
                'hist_avg_sales': 5.0,
                'hist_sales_std': 2.0,
                'hist_promo_sales_ratio': 1.2,
                'sales_trend': 0.0,
                'last_week_sales': 0.0
            }
        
        # 转换时间列
        if '交易时间' not in product_data.columns and '日期' in product_data.columns:
            product_data['交易时间'] = product_data['日期']
        
        if '交易时间' in product_data.columns:
            product_data['交易时间'] = pd.to_datetime(product_data['交易时间'])
        
        # 计算历史平均销量（最近30天）
        recent_days = 30
        cutoff_date = current_time - timedelta(days=recent_days)
        recent_data = product_data[product_data['交易时间'] >= cutoff_date]
        
        if not recent_data.empty:
            hist_avg_sales = recent_data['销售数量'].mean()
            hist_sales_std = recent_data['销售数量'].std()
        else:
            hist_avg_sales = product_data['销售数量'].mean()
            hist_sales_std = product_data['销售数量'].std()
        
        # 促销时段销售表现
        promo_start, promo_end = promotion_hours
        promo_data = product_data[
            (product_data['交易时间'].dt.hour >= promo_start) & 
            (product_data['交易时间'].dt.hour < promo_end)
        ]
        
        non_promo_data = product_data[
            (product_data['交易时间'].dt.hour < promo_start) | 
            (product_data['交易时间'].dt.hour >= promo_end)
        ]
        
        if not promo_data.empty and not non_promo_data.empty:
            promo_avg = promo_data['销售数量'].mean()
            non_promo_avg = non_promo_data['销售数量'].mean()
            hist_promo_sales_ratio = promo_avg / max(non_promo_avg, 0.1)
        else:
            hist_promo_sales_ratio = 1.2  # 默认促销提升20%
        
        # 销售趋势（最近7天 vs 前7天）
        week_ago = current_time - timedelta(days=7)
        two_weeks_ago = current_time - timedelta(days=14)
        
        recent_week = product_data[
            (product_data['交易时间'] >= week_ago) & 
            (product_data['交易时间'] < current_time)
        ]
        
        previous_week = product_data[
            (product_data['交易时间'] >= two_weeks_ago) & 
            (product_data['交易时间'] < week_ago)
        ]
        
        if not recent_week.empty and not previous_week.empty:
            recent_sales = recent_week['销售数量'].sum()
            previous_sales = previous_week['销售数量'].sum()
            sales_trend = (recent_sales - previous_sales) / max(previous_sales, 1)
        else:
            sales_trend = 0.0
        
        # 上周同期销量
        last_week_data = product_data[
            (product_data['交易时间'] >= week_ago - timedelta(days=1)) & 
            (product_data['交易时间'] < current_time - timedelta(days=7))
        ]
        last_week_sales = last_week_data['销售数量'].sum() if not last_week_data.empty else 0.0
        
        return {
            'hist_avg_sales': float(hist_avg_sales),
            'hist_sales_std': float(hist_sales_std),
            'hist_promo_sales_ratio': float(hist_promo_sales_ratio),
            'sales_trend': float(sales_trend),
            'last_week_sales': float(last_week_sales)
        }
    
    def _extract_price_features(self, transaction_data: pd.DataFrame, 
                               product_code: str) -> Dict:
        """提取价格特征"""
        
        product_data = transaction_data[transaction_data['商品编码'] == product_code].copy()
        
        if product_data.empty:
            return {
                'avg_price': 100.0,
                'price_std': 10.0,
                'min_price': 80.0,
                'max_price': 120.0,
                'avg_discount': 0.85,
                'price_elasticity': 1.2
            }
        
        # 计算价格统计
        if '售价' in product_data.columns:
            avg_price = product_data['售价'].mean()
            price_std = product_data['售价'].std()
            min_price = product_data['售价'].min()
            max_price = product_data['售价'].max()
        else:
            avg_price = 100.0
            price_std = 10.0
            min_price = 80.0
            max_price = 120.0
        
        # 计算平均折扣
        if '实际折扣率' in product_data.columns:
            avg_discount = product_data['实际折扣率'].mean()
        else:
            avg_discount = 0.85
        
        # 估算价格弹性（简化版）
        if len(product_data) >= 20 and '实际折扣率' in product_data.columns:
            # 按折扣率分组
            product_data['折扣分组'] = pd.qcut(product_data['实际折扣率'], q=4, duplicates='drop')
            grouped = product_data.groupby('折扣分组').agg({
                '实际折扣率': 'mean',
                '销售数量': 'mean'
            }).reset_index()
            
            if len(grouped) >= 2:
                # 计算弹性系数
                price_changes = grouped['实际折扣率'].diff().dropna()
                quantity_changes = grouped['销售数量'].pct_change().dropna()
                
                if len(price_changes) > 0 and len(quantity_changes) > 0:
                    elasticities = quantity_changes.abs() / price_changes.abs()
                    price_elasticity = elasticities.mean()
                else:
                    price_elasticity = 1.2
            else:
                price_elasticity = 1.2
        else:
            price_elasticity = 1.2
        
        return {
            'avg_price': float(avg_price),
            'price_std': float(price_std),
            'min_price': float(min_price),
            'max_price': float(max_price),
            'avg_discount': float(avg_discount),
            'price_elasticity': float(price_elasticity)
        }
    
    def normalize_features(self, features: Dict, feature_set: str = 'default') -> Dict:
        """特征归一化"""
        if feature_set not in self.scalers:
            self.scalers[feature_set] = MinMaxScaler()
            # 这里需要先拟合，实际使用中应该在初始化时用训练数据拟合
        
        # 将特征转换为数组
        feature_names = list(features.keys())
        feature_values = np.array(list(features.values())).reshape(1, -1)
        
        # 归一化
        if hasattr(self.scalers[feature_set], 'fit'):
            # 如果还没有拟合，先拟合
            self.scalers[feature_set].fit(feature_values)
        
        normalized_values = self.scalers[feature_set].transform(feature_values)
        
        # 转换回字典
        normalized_features = dict(zip(feature_names, normalized_values[0]))
        
        return normalized_features
    
    def reduce_dimensionality(self, features: Dict, n_components: int = 10) -> Dict:
        """降维"""
        feature_names = list(features.keys())
        feature_values = np.array(list(features.values())).reshape(1, -1)
        
        if 'pricing_pca' not in self.pca_models:
            self.pca_models['pricing_pca'] = PCA(n_components=min(n_components, len(feature_names)))
            # 这里需要先拟合，实际使用中应该在初始化时用训练数据拟合
        
        if hasattr(self.pca_models['pricing_pca'], 'fit'):
            self.pca_models['pricing_pca'].fit(feature_values)
        
        reduced_values = self.pca_models['pricing_pca'].transform(feature_values)
        
        # 创建新的特征字典
        reduced_features = {
            f'pca_component_{i}': float(value)
            for i, value in enumerate(reduced_values[0])
        }
        
        return reduced_features