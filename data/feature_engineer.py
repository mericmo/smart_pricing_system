# data/feature_engineer.py (简化版)
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import re


class PricingFeatureEngineer:
    """定价特征工程 -（专注于核心特征）"""

    def __init__(self, config=None):
        self.scalers = {}
        self.pca_models = {}
        self.feature_columns = []
        self.config = config

    def create_features(self, transaction_data: pd.DataFrame,
                        calendar_data: pd.DataFrame,
                        weather_data: pd.DataFrame,
                        product_code: str,
                        store_code: str,
                        promotion_hours: Tuple[int, int],
                        current_time: Any,
                        external_features: Optional[Dict] = None) -> Dict:
        """创建定价特征 - 简化版"""

        # 确保current_time是pandas Timestamp
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.Timestamp(current_time)
        # 数据预先处理
        transaction_data = self._filter_store_products(transaction_data, store_code, product_code)
        # 基础时间特征
        features = self._create_base_features(current_time, promotion_hours)

        # 商品历史特征
        historical_features = self._extract_historical_features(
            transaction_data, promotion_hours, current_time
        )
        features.update(historical_features)

        # 价格特征
        price_features = self._extract_price_features(transaction_data, product_code)
        features.update(price_features)

        # 天气特征
        weather_features = self._extract_weather_features(
            transaction_data, weather_data, product_code, current_time
        )
        features.update(weather_features)

        # 日历特征
        calendar_features = self._extract_calendar_features(calendar_data, current_time)
        features.update(calendar_features)

        # 商品基础信息特征
        product_features = self._extract_product_features(transaction_data, product_code)
        features.update(product_features)

        # 外部特征
        if external_features:
            features.update(external_features)

        return features

    def _filter_store_products(self, transaction_data: pd.DataFrame, store_code: str,
                               product_code: str) -> pd.DataFrame:
        # 注释掉的数据预处理代码
        transaction_data = transaction_data[
            (transaction_data['门店编码'] == store_code) & (transaction_data['商品编码'] == product_code) & (
                        transaction_data['销售数量'] > 0) & (transaction_data["销售金额"] > 0) & (transaction_data["售价"] > 0)]
        if "销售净额" in transaction_data.columns and "销售金额" in transaction_data.columns:
            transaction_data['平均售价'] = transaction_data['销售净额'] / transaction_data['销售净额'] * \
                                           transaction_data['销售数量']
            # 确保销售数量为数值（若有非数字或空值会变为 NaN）
            transaction_data["销售数量"] = pd.to_numeric(transaction_data["销售数量"], errors="coerce")
            # 确保金额列为数值
            transaction_data['销售净额'] = pd.to_numeric(transaction_data['销售净额'], errors="coerce")
            transaction_data["平均售价"] = np.where(
                transaction_data["销售数量"] > 0,
                transaction_data['销售净额'] / transaction_data["销售数量"],
                np.nan
            )
        return transaction_data

    def _create_base_features(self, current_time: pd.Timestamp,
                              promotion_hours: Tuple[int, int]) -> Dict:
        """创建基础时间特征"""
        current_year = current_time.year
        current_hour = current_time.hour
        current_minute = current_time.minute
        day_of_week = current_time.dayofweek
        day_of_month = current_time.day
        month = current_time.month
        is_weekend = 1 if day_of_week >= 5 else 0

        # 促销时间特征
        promo_start, promo_end = promotion_hours
        time_to_promo_start = self._calculate_time_distance(current_hour, promo_start)
        time_to_promo_end = self._calculate_time_distance(current_hour, promo_end)
        in_promotion_time = 1 if promo_start <= current_hour < promo_end else 0

        # 时间周期特征
        hour_sin = np.sin(2 * np.pi * current_hour / 24)
        hour_cos = np.cos(2 * np.pi * current_hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        # 是否为特殊时段
        is_morning_rush = 1 if 7 <= current_hour < 9 else 0
        is_lunch_time = 1 if 11 <= current_hour < 13 else 0
        is_evening_rush = 1 if 17 <= current_hour < 19 else 0
        is_night = 1 if current_hour >= 22 or current_hour < 6 else 0

        # 是否为月末/月初
        is_month_end = 1 if day_of_month >= 25 else 0
        is_month_start = 1 if day_of_month <= 5 else 0

        return {
            "year": current_year,
            'hour_of_day': current_hour,
            'minute_of_hour': current_minute,
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month,
            'weekend': is_weekend,
            'quarter': (month - 1) // 3 + 1,
            'time_to_promo_start': time_to_promo_start,
            'time_to_promo_end': time_to_promo_end,
            'in_promotion_time': in_promotion_time,
            'promo_duration_hours': (promo_end - promo_start) % 24,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'is_morning_rush': is_morning_rush,
            'is_lunch_time': is_lunch_time,
            'is_evening_rush': is_evening_rush,
            'is_night': is_night,
            'is_month_end': is_month_end,
            'is_month_start': is_month_start
        }

    def _calculate_time_distance(self, current_hour: int, target_hour: int) -> float:
        """计算时间距离"""
        if target_hour >= current_hour:
            return target_hour - current_hour
        else:
            return (24 - current_hour) + target_hour

    def _extract_historical_features(self, transaction_data: pd.DataFrame,
                                     promotion_hours: Tuple[int, int],
                                     current_time: pd.Timestamp) -> Dict:
        """提取历史销售特征"""

        # 筛选商品数据
        # product_data = transaction_data[transaction_data['商品编码'] == product_code].copy()
        product_data = transaction_data.copy()

        if product_data.empty:
            # return self._get_default_historical_features()
            raise ValueError("数据缺失。")
        # 确保时间列是pandas Timestamp
        # if '交易时间' in product_data.columns:
        #     if not pd.api.types.is_datetime64_any_dtype(product_data['交易时间']):
        #         product_data['交易时间'] = pd.to_datetime(product_data['交易时间'])
        # else:
        #     # return self._get_default_historical_features()
        #     raise ValueError("数据缺失交易时间字段。")

        # 计算历史平均销量（最近30天）
        recent_days = 30
        cutoff_date = current_time - pd.Timedelta(days=recent_days)
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
            hist_promo_sales_ratio = 1.2

        # 销售趋势（最近7天 vs 前7天）
        week_ago = current_time - pd.Timedelta(days=7)
        two_weeks_ago = current_time - pd.Timedelta(days=14)

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
            (product_data['交易时间'] >= week_ago - pd.Timedelta(days=1)) &
            (product_data['交易时间'] < current_time - pd.Timedelta(days=7))
            ]
        last_week_sales = last_week_data['销售数量'].sum() if not last_week_data.empty else 0.0

        # 最近3小时销量
        three_hours_ago = current_time - pd.Timedelta(hours=3)
        recent_3h_data = product_data[product_data['交易时间'] >= three_hours_ago]
        recent_3h_sales = recent_3h_data['销售数量'].sum() if not recent_3h_data.empty else 0.0

        return {
            'hist_avg_sales': float(hist_avg_sales),
            'hist_sales_std': float(hist_sales_std),
            'hist_promo_sales_ratio': float(hist_promo_sales_ratio),
            'sales_trend': float(sales_trend),
            'last_week_sales': float(last_week_sales),
            'recent_3h_sales': float(recent_3h_sales)
        }

    def _get_default_historical_features(self) -> Dict:
        """获取默认历史特征"""
        return {
            'hist_avg_sales': 10.0,
            'hist_sales_std': 5.0,
            'hist_promo_sales_ratio': 1.2,
            'sales_trend': 0.0,
            'last_week_sales': 0.0,
            'recent_3h_sales': 0.0
        }

    def _extract_price_features(self, transaction_data: pd.DataFrame,
                                product_code: str) -> Dict:
        """提取价格特征"""

        product_data = transaction_data[transaction_data['商品编码'] == product_code].copy()

        if product_data.empty:
            # 这里考虑直接退出
            return self._get_default_price_features()

        # 计算价格统计
        if '售价' in product_data.columns:
            avg_price = product_data['售价'].mean()
            price_std = product_data['售价'].std()
            min_price = product_data['售价'].min()
            max_price = product_data['售价'].max()
            median_price = product_data['售价'].median()
        else:
            avg_price = 100.0
            price_std = 10.0
            min_price = 80.0
            max_price = 120.0
            median_price = 100.0

        # 计算折扣特征
        if '实际折扣率' in product_data.columns:
            avg_discount = product_data['实际折扣率'].mean()
            discount_std = product_data['实际折扣率'].std()
            min_discount = product_data['实际折扣率'].min()
            max_discount = product_data['实际折扣率'].max()
        else:
            avg_discount = 1.0
            discount_std = 0
            min_discount = 1.0
            max_discount = 1.0

        # 估算价格弹性
        price_elasticity = self._estimate_price_elasticity(product_data)

        # 折扣频率
        if '是否折扣' in product_data.columns:
            discount_frequency = product_data['是否折扣'].mean()
        else:
            discount_frequency = 0.0

        return {
            'avg_price': float(avg_price),
            'price_std': float(price_std),
            'min_price': float(min_price),
            'max_price': float(max_price),
            'median_price': float(median_price),
            'avg_discount': float(avg_discount),
            'discount_std': float(discount_std),
            'min_discount': float(min_discount),
            'max_discount': float(max_discount),
            'price_elasticity': float(price_elasticity),
            'discount_frequency': float(discount_frequency)
        }

    def _estimate_price_elasticity(self, product_data: pd.DataFrame) -> float:
        """估算价格弹性"""
        if len(product_data) < 20 or '实际折扣率' not in product_data.columns:
            return 1.2

        try:
            # 按折扣率分组
            product_data['折扣分组'] = pd.qcut(
                product_data['实际折扣率'],
                q=4,
                duplicates='drop'
            )

            grouped = product_data.groupby('折扣分组').agg({
                '实际折扣率': 'mean',
                '销售数量': 'sum'
            }).reset_index()

            if len(grouped) < 2:
                return 1.2

            # 计算弹性系数
            elasticities = []
            for i in range(1, len(grouped)):
                price_change = (grouped.loc[i, '实际折扣率'] - grouped.loc[i - 1, '实际折扣率']) / grouped.loc[
                    i - 1, '实际折扣率']
                quantity_change = (grouped.loc[i, '销售数量'] - grouped.loc[i - 1, '销售数量']) / max(
                    grouped.loc[i - 1, '销售数量'], 1)

                if price_change != 0:
                    elasticity = abs(quantity_change / price_change)
                    elasticities.append(elasticity)

            if elasticities:
                return np.mean(elasticities)
        except:
            pass

        return 1.2

    def _get_default_price_features(self) -> Dict:
        """获取默认价格特征"""
        return {
            'avg_price': 100.0,
            'price_std': 10.0,
            'min_price': 80.0,
            'max_price': 120.0,
            'median_price': 100.0,
            'avg_discount': 1.0,
            'discount_std': 0,
            'min_discount': 1.0,
            'max_discount': 1.0,
            'price_elasticity': 1.2,
            'discount_frequency': 0.0
        }

    def _extract_weather_features(self, transaction_data: pd.DataFrame,
                                  weather_data: pd.DataFrame,
                                  product_code: str,
                                  current_time: pd.Timestamp) -> Dict:
        """提取天气特征"""

        # 获取当前日期的天气数据
        current_date = current_time.date()

        # 查找当天的天气数据
        if weather_data is not None and 'date' in weather_data.columns and 'high' in weather_data.columns:
            # 从交易数据中提取
            # today_weather = transaction_data[
            #     (transaction_data['日期'] == pd.Timestamp(current_date)) &
            #     (transaction_data['商品编码'] == product_code)
            #     ]
            weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date
            today_weather = weather_data[
                weather_data['date'] == current_date
                ]
            if not today_weather.empty:
                # 取第一条记录的天气数据
                row = today_weather.iloc[0]

                features = {
                    'temperature': float(row.get('high', 25)) if pd.notna(row.get('high')) else 25.0,
                    'temp_low': float(row.get('low', 20)) if pd.notna(row.get('low')) else 20.0,
                    'temp_range': float(row.get('温差', 5)) if pd.notna(row.get('温差')) else 5.0,
                    'weather_severity': float(row.get('天气严重程度', 0)) if pd.notna(row.get('天气严重程度')) else 0
                }

                # 天气类型编码
                if 'code_day' in row and pd.notna(row['code_day']):
                    code = str(row['code_day']).zfill(2)
                    features.update({
                        'is_sunny': 1 if code == '01' else 0,
                        'is_cloudy': 1 if code == '02' else 0,
                        'is_rainy': 1 if code in ['04', '07', '08', '09', '10'] else 0,
                        'is_snowy': 1 if code in ['13', '14', '15', '16'] else 0,
                        'is_foggy': 1 if code == '18' else 0,
                        'is_haze': 1 if code == '53' else 0
                    })
                else:
                    features.update({
                        'is_sunny': 0,
                        'is_cloudy': 0,
                        'is_rainy': 0,
                        'is_snowy': 0,
                        'is_foggy': 0,
                        'is_haze': 0
                    })

                return features

        # 如果没有天气数据，返回默认值
        return self._get_default_weather_features()

    def _get_default_weather_features(self) -> Dict:
        """获取默认天气特征"""
        return {
            'temperature': 25.0,
            'temp_low': 20.0,
            'temp_range': 5.0,
            'weather_severity': 0,
            'is_sunny': 0,
            'is_cloudy': 0,
            'is_rainy': 0,
            'is_snowy': 0,
            'is_foggy': 0,
            'is_haze': 0
        }

    def _extract_calendar_features(self, calendar_data: pd.DataFrame, current_time: pd.Timestamp) -> Dict:
        """提取日历特征"""
        current_date = current_time.date()
        day_of_week = current_time.dayofweek
        day_of_month = current_time.day
        month = current_time.month

        # 是否为节假日（简化判断）
        is_holiday = 0
        holiday_impact = 0

        # 重要节假日判断（简化版）
        # if month == 1 and 1 <= day_of_month <= 3:  # 元旦
        #     is_holiday = 1
        #     holiday_impact = 2
        # elif month == 2 and 10 <= day_of_month <= 17:  # 春节（简化）
        #     is_holiday = 1
        #     holiday_impact = 3
        # elif month == 5 and 1 <= day_of_month <= 3:  # 劳动节
        #     is_holiday = 1
        #     holiday_impact = 2
        # elif month == 10 and 1 <= day_of_month <= 7:  # 国庆节
        #     is_holiday = 1
        #     holiday_impact = 3
        if not calendar_data.empty:
            calendar_data['date'] = pd.to_datetime(calendar_data['date']).dt.date
            today_row = calendar_data[calendar_data['date'] == current_date]
            if not today_row.empty:
                # 检查节假日字段
                if 'holiday_legal' in today_row.columns:
                    holiday_value = today_row.iloc[0]['holiday_legal']
                    # 正确判断是否为节假日
                    # 假设holiday_legal是1/0或True/False表示节假日
                    if pd.notna(holiday_value):
                        is_holiday = int(holiday_value)  # 转换为整数

                # 检查节假日影响力字段
                if '节假日影响力' in today_row.columns:
                    impact_value = today_row.iloc[0]['节假日影响力']
                    if pd.notna(impact_value):
                        holiday_impact = float(impact_value)
        # 是否为特殊购物日
        is_shopping_day = 0
        if month == 11 and day_of_month == 11:  # 双11
            is_shopping_day = 1
        elif month == 12 and day_of_month == 12:  # 双12
            is_shopping_day = 1
        elif month == 6 and day_of_month == 18:  # 618
            is_shopping_day = 1

        # 是否为发薪日（假设每月15日和30日）
        is_payday = 1 if day_of_month in [15, 30] else 0

        return {
            'is_holiday': is_holiday,
            'holiday_impact': holiday_impact,
            'is_shopping_day': is_shopping_day,
            'is_payday': is_payday,
            'day_of_week': day_of_week,#current_time.dayofyear,
            'week_of_year': current_time.isocalendar()[1]
        }

    def _extract_product_features(self, transaction_data: pd.DataFrame,
                                  product_code: str) -> Dict:
        """提取商品基础信息特征"""

        # product_data = transaction_data[transaction_data['商品编码'] == product_code]
        product_data = transaction_data.copy()
        if product_data.empty:
            return self._get_default_product_features()

        # 获取第一条记录的商品信息
        row = product_data.iloc[0]

        # 商品类别
        if '小类编码' in row and pd.notna(row['小类编码']):
            category_code = str(row['小类编码'])
            main_category = category_code[:2]
            sub_category = category_code[:4]
        else:
            main_category = '00'
            sub_category = '0000'

        # 商品名称分析
        product_name = str(row.get('商品名称', '')) if '商品名称' in row else ''

        # 商品规格提取
        weight = self._extract_weight_from_name(product_name)

        return {
            'main_category': main_category,
            'sub_category': sub_category,
            'product_weight': weight,
            'name_length': len(product_name)
        }

    def _extract_weight_from_name(self, product_name: str) -> float:
        """从商品名称中提取重量"""
        if not product_name:
            return 0.0

        # 匹配重量/容量
        patterns = [
            r'(\d+(\.\d+)?)\s*[kK]?[gG]',  # 重量，如380g
            r'(\d+(\.\d+)?)\s*[mM]?[lL]',  # 容量，如500ml
            r'(\d+(\.\d+)?)\s*[cC][mM]',  # 尺寸，如15cm
            r'(\d+(\.\d+)?)\s*个',  # 个数，如6个
            r'(\d+(\.\d+)?)\s*包',  # 包数，如12包
        ]

        for pattern in patterns:
            match = re.search(pattern, product_name)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue

        return 0.0

    def _get_default_product_features(self) -> Dict:
        """获取默认商品特征"""
        return {
            'main_category': '00',
            'sub_category': '0000',
            'product_weight': 0.0,
            'name_length': 0
        }
