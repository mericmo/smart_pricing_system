# data/data_processor.py (更新版)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import warnings
warnings.filterwarnings('ignore')
import os
'''
时间特征：hour, minute, day_of_week, month, day_of_month, is_weekend, quarter

促销时间特征：in_promotion, time_to_promo_end, promo_duration_hours

周期特征：hour_sin, hour_cos, day_sin, day_cos

时段特征：is_morning_rush, is_lunch_time, is_evening_rush, is_night

价格特征：price, discount_rate, has_discount, price_ratio

历史销售特征：hist_avg_sales, hist_sales_std, hist_promo_sales_ratio, sales_trend, last_week_sales, recent_3h_sales

价格统计特征：avg_price, price_std, min_price, max_price, median_price

折扣特征：avg_discount, discount_std, min_discount, max_discount, price_elasticity, discount_frequency

天气特征：temperature, temp_low, temp_range, weather_severity, 天气类型

日历特征：is_holiday, holiday_impact, is_shopping_day, is_payday, day_of_year, week_of_year

商品特征：main_category, sub_category, product_weight, name_length
'''
class TransactionDataProcessor:
    """交易数据处理器 - 针对您的数据格式优化"""
    
    def __init__(self, transaction_data: pd.DataFrame, 
                 weather_data: pd.DataFrame = None,
                 calendar_data: pd.DataFrame = None,
                 start_time = 20,
                 end_time = 22
                 ):
        """
        初始化处理器
        
        Args:
            transaction_data: 交易数据DataFrame
            weather_data: 天气数据DataFrame
            calendar_data: 日历数据DataFrame
        """
        self.transaction_data = transaction_data.copy()
        self.weather_data = weather_data.copy() if weather_data is not None else None
        self.calendar_data = calendar_data.copy() if calendar_data is not None else None
        self.start_time = start_time
        self.end_time = end_time
        self._preprocess_data()
    
    def _preprocess_data(self):
        """数据预处理 - 针对您的数据格式"""
        # 处理交易数据
        self._preprocess_transaction_data()
        
        # 处理天气数据
        if self.weather_data is not None:
            self._preprocess_weather_data()
        
        # 处理日历数据
        if self.calendar_data is not None:
            self._preprocess_calendar_data()
        
        # 合并外部数据
        if self.weather_data is not None or self.calendar_data is not None:
            self._merge_external_data()
    
    def _preprocess_transaction_data(self):
        """预处理交易数据"""
        # 1. 去除异常数据
        self.transaction_data = self.transaction_data[(self.transaction_data["销售金额"] > 0) & (self.transaction_data["销售数量"] > 0)]

        # 2. 处理时间字段
        if '交易时间' in self.transaction_data.columns:
            # 统一时间格式
            self.transaction_data['交易时间'] = pd.to_datetime(
                self.transaction_data['交易时间'], 
                errors='coerce',
                format='%Y/%m/%d %H:%M'
            )
        elif '日期' in self.transaction_data.columns:
            # 如果只有日期字段，创建交易时间（假设交易发生在中午12点）
            self.transaction_data['交易时间'] = pd.to_datetime(
                self.transaction_data['日期'] + ' 12:00',
                errors='coerce',
                format='%Y/%m/%d %H:%M'
            )
        
        # 3. 处理折扣类型
        if '折扣类型' in self.transaction_data.columns:
            # 解析折扣类型
            self.transaction_data['是否折扣'] = self.transaction_data['折扣类型'].apply(
                lambda x: 0 if str(x).startswith('n-无折扣') else 1
            )
            
            # 提取折扣力度（如果有）
            self.transaction_data['折扣力度'] = self.transaction_data.apply(
                self._extract_discount_rate, axis=1
            )
        
        # 4. 计算实际折扣率
        if all(col in self.transaction_data.columns for col in ['销售金额', '售价', '销售数量']):
            self.transaction_data['实际折扣率'] = self.transaction_data.apply(
                # lambda row: row['销售金额'] / (row['售价'] * row['销售数量'])
                lambda row: row['销售净额'] / row['销售金额']
                if row['售价'] * row['销售数量'] > 0 else 1.0,
                axis=1
            )
        # 5. 计算均价
        if "销售净额" in self.transaction_data.columns and "销售金额" in self.transaction_data.columns:
            # self.transaction_data['平均售价'] = self.transaction_data['销售净额'] / self.transaction_data['销售净额'] * \
            #                                self.transaction_data['销售数量']
            # 确保销售数量为数值（若有非数字或空值会变为 NaN）
            self.transaction_data["销售数量"] = pd.to_numeric(self.transaction_data["销售数量"], errors="coerce")
            # 确保金额列为数值
            self.transaction_data['销售净额'] = pd.to_numeric(self.transaction_data['销售净额'], errors="coerce")
            self.transaction_data["平均售价"] = np.where(
                self.transaction_data["销售数量"] > 0,
                self.transaction_data['销售净额'] / self.transaction_data["销售数量"],
                np.nan
            )
        # 6. 提取时间特征
        self._extract_transaction_time_features()
        
        # 7. 处理商品信息
        self._process_product_info()
    
    def _extract_discount_rate(self, row):
        """从折扣类型中提取折扣率"""
        discount_type = str(row['折扣类型'])
        discount_amount = float(row['折扣金额'])
        # 常见折扣类型解析
        if 'n-无折扣促销' in discount_type or discount_amount == 0:
            return 1.0
        elif discount_amount != 0:
            sale_amount = float(row['销售金额'])
            sale_num = float(row['销售数量'])
            return discount_amount * sale_num / sale_amount
        # elif '折' in discount_type:
        #     # 提取数字，如"8折" -> 0.8
        #     import re
        #     match = re.search(r'(\d+(\.\d+)?)折', discount_type)
        #     if match:
        #         return float(match.group(1)) / 10
        # elif '减' in discount_type or '满减' in discount_type:
        #     # 满减活动，粗略估计折扣率
        #     return 0.8  # 假设满减平均8折
        
        # 默认为原价
        return 1.0
    
    def _extract_transaction_time_features(self):
        """提取交易时间特征"""
        if '交易时间' not in self.transaction_data.columns:
            return
        
        df = self.transaction_data
        
        # 基本时间特征
        df['年'] = df['交易时间'].dt.year
        df['月'] = df['交易时间'].dt.month
        df['日'] = df['交易时间'].dt.day
        df['小时'] = df['交易时间'].dt.hour
        df['分钟'] = df['交易时间'].dt.minute
        df['星期几'] = df['交易时间'].dt.dayofweek  # 0-6，0=周一
        df['是否周末'] = df['星期几'].isin([5, 6]).astype(int)
        
        # 时间段划分
        df['时间段'] = pd.cut(
            df['小时'],
            bins=[0, 6, 9, 11, 14, 17, 20, 24],
            labels=['深夜', '早高峰', '上午', '中午', '下午', '晚高峰', '晚间'],
            right=False
        )
        
        # 是否促销时段（20:00-22:00）
        df['是否促销时段'] = ((df['小时'] >= self.start_time) & (df['小时'] < self.end_time)).astype(int)
        
        # 季度
        df['季度'] = df['交易时间'].dt.quarter
        
        # 是否为月末（最后5天）
        df['是否月末'] = (df['日'] >= 26).astype(int)
        
        # 是否为月初（前5天）
        df['是否月初'] = (df['日'] <= 5).astype(int)
    
    def _process_product_info(self):
        """处理商品信息"""
        if '商品编码' in self.transaction_data.columns:
            # 商品分类（根据小类编码）
            if '小类编码' in self.transaction_data.columns:
                self.transaction_data['商品大类'] = self.transaction_data['小类编码'].apply(
                    lambda x: str(x)[:2] if pd.notna(x) else '00'
                )
            
            # 商品名称处理
            if '商品名称' in self.transaction_data.columns:
                # 提取商品规格
                self.transaction_data['商品规格'] = self.transaction_data['商品名称'].apply(
                    self._extract_product_spec
                )
    
    def _extract_product_spec(self, product_name):
        """提取商品规格"""
        if not isinstance(product_name, str):
            return ''
        
        # 提取重量/容量信息
        import re
        patterns = [
            r'(\d+(\.\d+)?)[kK]?[gG]',  # 重量，如380g
            r'(\d+(\.\d+)?)[mM]?[lL]',  # 容量，如500ml
            r'(\d+(\.\d+)?)[cC][mM]',   # 尺寸，如15cm
            r'(\d+(\.\d+)?)个',         # 个数，如6个
            r'(\d+(\.\d+)?)包',         # 包数，如12包
        ]
        
        for pattern in patterns:
            match = re.search(pattern, product_name)
            if match:
                return match.group(1)
        
        return ''
    
    def _preprocess_weather_data(self):
        """预处理天气数据"""
        if self.weather_data is None:
            return
        
        # 统一日期格式
        if 'date' in self.weather_data.columns:
            self.weather_data['date'] = pd.to_datetime(
                self.weather_data['date'],
                errors='coerce',
                format='%Y/%m/%d'
            )
        
        # 创建天气编码映射
        # weather_code_map = {
        #     '01': '晴', '02': '多云', '03': '阴', '04': '阵雨',
        #     '05': '雷阵雨', '06': '雨夹雪', '07': '小雨', '08': '中雨',
        #     '09': '大雨', '10': '暴雨', '11': '大暴雨', '12': '特大暴雨',
        #     '13': '阵雪', '14': '小雪', '15': '中雪', '16': '大雪',
        #     '17': '暴雪', '18': '雾', '19': '冻雨', '20': '沙尘暴',
        #     '21': '小到中雨', '22': '中到大雨', '23': '大到暴雨',
        #     '24': '暴雨到大暴雨', '25': '大暴雨到特大暴雨',
        #     '26': '小到中雪', '27': '中到大雪', '28': '大到暴雪',
        #     '29': '浮尘', '30': '扬沙', '31': '强沙尘暴', '53': '霾'
        # }
        
        # 添加天气描述
        if 'code_day' in self.weather_data.columns:
        #     self.weather_data['天气描述'] = self.weather_data['code_day'].astype(str).map(
        #         lambda x: weather_code_map.get(x, '未知')
        #     )
            self.weather_data['天气描述'] = self.weather_data['code_day'].astype(str)
        # 计算温差
        if all(col in self.weather_data.columns for col in ['high', 'low']):
            self.weather_data['温差'] = self.weather_data['high'] - self.weather_data['low']
        
        # 天气严重程度评分
        def get_weather_severity(code):
            if pd.isna(code):
                return 0
            
            code_str = str(code).zfill(2)
            severity_map = {
                '01': 0,  # 晴
                '02': 1,  # 多云
                '03': 2,  # 阴
                '04': 3,  # 阵雨
                '07': 4,  # 小雨
                '08': 5,  # 中雨
                '09': 6,  # 大雨
                '10': 7,  # 暴雨
                '13': 4,  # 阵雪
                '14': 5,  # 小雪
                '15': 6,  # 中雪
                '16': 7,  # 大雪
                '18': 3,  # 雾
                '53': 4,  # 霾
            }
            return severity_map.get(code_str, 0)
        
        if 'code_day' in self.weather_data.columns:
            self.weather_data['天气严重程度'] = self.weather_data['code_day'].apply(get_weather_severity)
    
    def _preprocess_calendar_data(self):
        """预处理日历数据"""
        if self.calendar_data is None:
            return
        # 统一日期格式
        if 'date' in self.calendar_data.columns:
            self.calendar_data['date'] = pd.to_datetime(
                self.calendar_data['date'],
                errors='coerce',
                format='%Y/%m/%d'
            )
        
        # 如果没有周末字段，根据日期计算
        # if 'is_weekend' not in self.calendar_data.columns:
        #     self.calendar_data['星期几'] = self.calendar_data['date'].dt.dayofweek
        #     self.calendar_data['is_weekend'] = self.calendar_data['星期几'].isin([5, 6]).astype(int)

        # 节假日影响力评分
        if 'holiday_name' in self.calendar_data.columns and 'holiday_type' in self.calendar_data.columns:

            # 确保日期列是datetime类型
            # if 'ds' not in self.calendar_data.columns:
            #     # 如果没有ds列，从date列创建
            #     self.calendar_data['ds'] = pd.to_datetime(self.calendar_data['date'].astype(str), format='%Y%m%d')

            def get_holiday_impact(row):
                """
                计算节假日影响力评分

                评分规则：
                0: 无节假日影响
                1: 普通节假日或特殊日期
                2: 电商大促日（双11、双12、618）
                3: 调休日
                4: 法定节假日期间
                """
                holiday_name = row['holiday_name']
                holiday_type = row['holiday_type']
                date = row['date']

                # 如果没有节假日名称，返回0
                if pd.isna(holiday_name) or holiday_name == '':
                    return 0
                # 法定节假日期间 - 最高影响力
                if holiday_type == 'holiday':
                    return 4
                # 调休日 - 较高影响力
                elif holiday_type == 'recess':
                    return 3

                # 电商大促日
                major_holidays = [[11, 11], [12, 12], [6, 18]]
                for holiday in major_holidays:
                    if date.month == holiday[0] and date.day == holiday[1]:
                        return 2

                # 其他有节日名称的日子（节假日窗口期但不是法定节假日当天）
                # 例如：春节前几天，国庆节前后等
                return 1

            # 应用函数
            self.calendar_data['节假日影响力'] = self.calendar_data.apply(get_holiday_impact, axis=1)
    
    def _merge_external_data(self):
        """合并外部数据到交易数据"""
        # 确保交易数据有日期列
        if '日期' not in self.transaction_data.columns:
            return

        # 确保列是datetime类型
        # if not pd.api.types.is_datetime64_any_dtype(self.transaction_data['日期']):
        #     print("正在将日期列转换为datetime类型...")
        #     self.transaction_data['日期'] = pd.to_datetime(self.transaction_data['日期'], errors='coerce')
        #     # 检查转换结果
        #     null_count = self.transaction_data['日期'].isna().sum()
        #     if null_count > 0:
        #         print(f"警告: 有 {null_count} 个日期转换失败，将被设为NaT")
        # 提取交易日期
        self.transaction_data['日期'] = self.transaction_data['日期'].dt.date
        
        # 合并天气数据
        if self.weather_data is not None and 'date' in self.weather_data.columns:
            # 转换日期格式以便合并
            self.weather_data['date_date'] = self.weather_data['date'].dt.date
            
            # 选择需要的天气列
            weather_cols = ['date_date']
            for col in ['high', 'low', '温差', '天气描述', '天气严重程度', 'code_day', 'text_day', 'text_night', 'code_night']:
                if col in self.weather_data.columns:
                    weather_cols.append(col)
            
            weather_subset = self.weather_data[weather_cols].copy()
            
            # 合并
            self.transaction_data = pd.merge(
                self.transaction_data,
                weather_subset,
                left_on='日期',
                right_on='date_date',
                how='left'
            )
            
            # 清理临时列
            if 'date_date' in self.transaction_data.columns:
                self.transaction_data = self.transaction_data.drop('date_date', axis=1)
        
        # 合并日历数据
        if self.calendar_data is not None and 'date' in self.calendar_data.columns:
            # 转换日期格式以便合并
            self.calendar_data['date_date'] = self.calendar_data['date'].dt.date
            
            # 选择需要的日历列
            calendar_cols = ['date_date']
            for col in ['holiday_legal', 'weekend', 'workday', 'holiday_name', '节假日影响力', 'special_event']:
                if col in self.calendar_data.columns:
                    calendar_cols.append(col)
            calendar_subset = self.calendar_data[calendar_cols].copy()
            
            # 合并
            self.transaction_data = pd.merge(
                self.transaction_data,
                calendar_subset,
                left_on='日期',
                right_on='date_date',
                how='left'
            )
            
            # 清理临时列
            if 'date_date' in self.transaction_data.columns:
                self.transaction_data = self.transaction_data.drop('date_date', axis=1)
    
    def get_product_sales_pattern(self, product_code: str, 
                                 store_code: Optional[str] = None,
                                 lookback_days: int = 90) -> Dict[str, Any]:
        """获取商品销售模式 - 针对数据优化"""
        
        # 筛选数据
        product_data = self.filter_by_product(product_code, store_code)
        
        if product_data.empty:
            return self._get_default_sales_pattern()
        
        # 限制时间范围
        if '交易时间' in product_data.columns:
            cutoff_date = product_data['交易时间'].max() - timedelta(days=lookback_days)
            recent_data = product_data[product_data['交易时间'] >= cutoff_date]
        else:
            recent_data = product_data
        
        if recent_data.empty:
            recent_data = product_data
        
        # 按小时分析销售模式
        hourly_sales = {}
        if '小时' in recent_data.columns:
            for hour in range(24):
                hour_data = recent_data[recent_data['小时'] == hour]
                if not hour_data.empty:
                    hourly_sales[f'hour_{hour}'] = {
                        'avg_sales': hour_data['销售数量'].mean(),
                        'total_sales': hour_data['销售数量'].sum(),
                        'transactions': len(hour_data),
                        'avg_discount': hour_data['实际折扣率'].mean() if '实际折扣率' in hour_data.columns else 1.0
                    }
        
        # 按星期分析
        weekday_sales = {}
        if '星期几' in recent_data.columns:
            for day in range(7):
                day_data = recent_data[recent_data['星期几'] == day]
                if not day_data.empty:
                    weekday_sales[f'weekday_{day}'] = {
                        'avg_sales': day_data['销售数量'].mean(),
                        'total_sales': day_data['销售数量'].sum(),
                        'transactions': len(day_data)
                    }
        
        # 促销时段分析
        promo_performance = {}
        if '是否促销时段' in recent_data.columns:
            promo_data = recent_data[recent_data['是否促销时段'] == 1]
            non_promo_data = recent_data[recent_data['是否促销时段'] == 0]
            
            if not promo_data.empty and not non_promo_data.empty:
                promo_performance = {
                    'promo_avg_sales': promo_data['销售数量'].mean(),
                    'non_promo_avg_sales': non_promo_data['销售数量'].mean(),
                    'promo_ratio': promo_data['销售数量'].sum() / non_promo_data['销售数量'].sum() 
                    if non_promo_data['销售数量'].sum() > 0 else 1.0
                }
        
        # 价格弹性分析
        price_elasticity = self._calculate_price_elasticity(recent_data)
        
        # 天气影响分析
        weather_impact = self._analyze_weather_impact(recent_data)
        
        # 节假日影响分析
        holiday_impact = self._analyze_holiday_impact(recent_data)
        sale_num = recent_data['销售数量'].sum()
        sale_amount =  recent_data['销售金额'].sum()
        # 综合结果
        result = {
            'product_code': product_code,
            'store_code': store_code,
            'total_sales': sale_num,#recent_data['销售数量'].sum(),
            'total_revenue': sale_amount, #recent_data['销售金额'].sum(),
            # 'avg_price': recent_data['售价'].mean() if '售价' in recent_data.columns else 0,
            'avg_price': recent_data['平均售价'] if '平均售价' in recent_data.columns else 0,
            'avg_discount': recent_data['实际折扣率'].mean() if '实际折扣率' in recent_data.columns else 1.0,
            'discount_frequency': recent_data['是否折扣'].mean() if '是否折扣' in recent_data.columns else 0,
            'hourly_pattern': hourly_sales,
            'weekday_pattern': weekday_sales,
            'promo_performance': promo_performance,
            'price_elasticity': price_elasticity,
            'weather_impact': weather_impact,
            'holiday_impact': holiday_impact,
            # 计算销售趋势
            'sales_trend': self._calculate_sales_trend(recent_data),
            # 获取最佳销售时段
            'best_selling_hours': self._get_best_selling_hours(hourly_sales),
            # 获取最差销售时段
            'worst_selling_hours': self._get_worst_selling_hours(hourly_sales)
        }

        return result
    
    def _calculate_price_elasticity(self, product_data: pd.DataFrame) -> float:
        """计算价格弹性"""
        if len(product_data) < 10 or '实际折扣率' not in product_data.columns:
            return 1.2  # 默认值
        
        # 按折扣率分组
        product_data['折扣分组'] = pd.qcut(
            product_data['实际折扣率'], 
            q=5, 
            duplicates='drop'
        )
        
        grouped = product_data.groupby('折扣分组').agg({
            '实际折扣率': 'mean',
            '销售数量': 'sum'
        }).reset_index()
        
        if len(grouped) < 2:
            return 1.2
        
        # 计算弹性（价格变化百分比 vs 销量变化百分比）
        elasticity_values = []
        for i in range(1, len(grouped)):
            price_change = (grouped.loc[i, '实际折扣率'] - grouped.loc[i-1, '实际折扣率']) / grouped.loc[i-1, '实际折扣率']
            quantity_change = (grouped.loc[i, '销售数量'] - grouped.loc[i-1, '销售数量']) / max(grouped.loc[i-1, '销售数量'], 1)
            
            if price_change != 0:
                elasticity = abs(quantity_change / price_change)
                elasticity_values.append(elasticity)
        
        if elasticity_values:
            return np.mean(elasticity_values)
        
        return 1.2
    
    def _analyze_weather_impact(self, product_data: pd.DataFrame) -> Dict[str, float]:
        """分析天气影响"""
        result = {
            'rain_impact': 1.0,
            'hot_impact': 1.0,
            'cold_impact': 1.0
        }
        
        if '天气描述' not in product_data.columns or 'code_day' not in product_data.columns:
            return result
        
        # 雨天影响
        rainy_days = product_data[product_data['code_day'].astype(str).str.contains('04|07|08|09|10')]
        sunny_days = product_data[product_data['code_day'].astype(str).str.contains('01')]
        
        if not rainy_days.empty and not sunny_days.empty:
            rainy_sales = rainy_days['销售数量'].sum() / len(rainy_days)
            sunny_sales = sunny_days['销售数量'].sum() / len(sunny_days)
            if sunny_sales > 0:
                result['rain_impact'] = rainy_sales / sunny_sales
        
        # 温度影响
        if 'high' in product_data.columns:
            hot_days = product_data[product_data['high'] >= 30]
            mild_days = product_data[(product_data['high'] >= 20) & (product_data['high'] < 30)]
            
            if not hot_days.empty and not mild_days.empty:
                hot_sales = hot_days['销售数量'].sum() / len(hot_days)
                mild_sales = mild_days['销售数量'].sum() / len(mild_days)
                if mild_sales > 0:
                    result['hot_impact'] = hot_sales / mild_sales
        
        return result
    
    def _analyze_holiday_impact(self, product_data: pd.DataFrame) -> Dict[str, float]:
        """分析节假日影响"""
        result = {
            'holiday_boost': 1.0,
            'weekend_boost': 1.0
        }
        
        if 'holiday_type' not in product_data.columns or 'holiday_name' not in product_data.columns:
            return result
        
        # 节假日影响
        holiday_days = product_data[product_data['holiday_type'] is not None or product_data['holiday_type'] is not None]
        weekday_days = product_data[product_data['is_holiday'] is None]
        
        if not holiday_days.empty and not weekday_days.empty:
            holiday_sales = holiday_days['销售数量'].sum() / len(holiday_days)
            weekday_sales = weekday_days['销售数量'].sum() / len(weekday_days)
            if weekday_sales > 0:
                result['holiday_boost'] = holiday_sales / weekday_sales
        
        # 周末影响
        weekend_days = product_data[product_data['weekend'] == 1]
        work_days = product_data[product_data['weekend'] == 0]
        
        if not weekend_days.empty and not work_days.empty:
            weekend_sales = weekend_days['销售数量'].sum() / len(weekend_days)
            work_sales = work_days['销售数量'].sum() / len(work_days)
            if work_sales > 0:
                result['weekend_boost'] = weekend_sales / work_sales
        
        return result
    
    def _calculate_sales_trend(self, product_data: pd.DataFrame) -> float:
        """计算销售趋势"""
        if '交易时间' not in product_data.columns or len(product_data) < 14:
            return 0.0
        
        # 按天聚合
        # product_data['交易日期'] = product_data['交易时间'].dt.date
        # daily_sales = product_data.groupby('交易日期')['销售数量'].sum().reset_index()
        daily_sales = product_data.groupby('日期')['销售数量'].sum().reset_index()

        if len(daily_sales) < 14:
            return 0.0
        
        # 计算最近7天 vs 前7天的趋势
        recent_7_days = daily_sales.tail(7)['销售数量'].sum()
        previous_7_days = daily_sales.tail(14).head(7)['销售数量'].sum()
        
        if previous_7_days > 0:
            trend = (recent_7_days - previous_7_days) / previous_7_days
        else:
            trend = 0.0
        
        return trend
    
    def _get_best_selling_hours(self, hourly_sales: Dict) -> List[int]:
        """获取最佳销售时段"""
        if not hourly_sales:
            # return [20, 21]  # 默认促销时段
            return [self.start_time, self.end_time]  # 默认促销时段
        # 找到平均销量最高的时段
        hour_avg_sales = []
        for hour_str, data in hourly_sales.items():
            hour = int(hour_str.split('_')[1])
            hour_avg_sales.append((hour, data['avg_sales']))
        
        # 按平均销量降序排序
        hour_avg_sales.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前3个时段
        return [hour for hour, _ in hour_avg_sales[:3]]
    
    def _get_worst_selling_hours(self, hourly_sales: Dict) -> List[int]:
        """获取最差销售时段"""
        if not hourly_sales:
            return [2, 3, 4]  # 默认凌晨时段
        
        # 找到平均销量最低的时段
        hour_avg_sales = []
        for hour_str, data in hourly_sales.items():
            hour = int(hour_str.split('_')[1])
            hour_avg_sales.append((hour, data['avg_sales']))
        
        # 按平均销量升序排序
        hour_avg_sales.sort(key=lambda x: x[1])
        
        # 返回前3个时段
        return [hour for hour, _ in hour_avg_sales[:3]]
    
    def _get_default_sales_pattern(self) -> Dict[str, Any]:
        """获取默认销售模式"""
        return {
            'product_code': 'unknown',
            'store_code': 'unknown',
            'total_sales': 0,
            'total_revenue': 0,
            'avg_price': 0,
            'avg_discount': 1.0,
            'discount_frequency': 0,
            'hourly_pattern': {},
            'weekday_pattern': {},
            'promo_performance': {},
            'price_elasticity': 1.2,
            'weather_impact': {'rain_impact': 1.0, 'hot_impact': 1.0, 'cold_impact': 1.0},
            'holiday_impact': {'holiday_boost': 1.0, 'weekend_boost': 1.0},
            'sales_trend': 0.0,
            'best_selling_hours': [20, 21],
            'worst_selling_hours': [2, 3, 4]
        }
    
    def filter_by_product(self, product_code: str, store_code: Optional[str] = None) -> pd.DataFrame:
        """按商品编码筛选数据"""
        filtered = self.transaction_data[
            self.transaction_data['商品编码'] == product_code
        ]
        
        if store_code:
            filtered = filtered[filtered['门店编码'] == store_code]
        
        return filtered.copy()
    
    def filter_by_time_range(self, start_hour: int, end_hour: int) -> pd.DataFrame:
        """按时间段筛选数据"""
        if '小时' not in self.transaction_data.columns:
            return self.transaction_data.copy()
        
        if start_hour <= end_hour:
            mask = (self.transaction_data['小时'] >= start_hour) & (self.transaction_data['小时'] < end_hour)
        else:
            # 跨天时间段
            mask = (self.transaction_data['小时'] >= start_hour) | (self.transaction_data['小时'] < end_hour)
        
        return self.transaction_data[mask].copy()
    
    def get_product_summary(self, product_code: str, store_code: Optional[str] = None) -> Dict:
        """获取商品汇总信息"""
        product_data = self.filter_by_product(product_code, store_code)
        
        if product_data.empty:
            return self._get_default_product_summary(product_code)
        
        # 计算基本统计
        total_sales = product_data['销售数量'].sum()
        total_revenue = product_data['销售金额'].sum()
        # avg_price = product_data['销售净额'].mean() if '销售净额' in product_data.columns else 0
        sale_amount = product_data['销售净额'].sum()
        avg_price = sale_amount/total_sales if sale_amount != 0 and total_sales != 0 else 0
        
        # 折扣统计
        if '实际折扣率' in product_data.columns:
            avg_discount = product_data['实际折扣率'].mean()
            discount_std = product_data['实际折扣率'].std()
        else:
            avg_discount = 1.0
            discount_std = 0
        
        # 销售趋势
        sales_trend = self._calculate_sales_trend(product_data)
        
        # 获取销售模式
        sales_pattern = self.get_product_sales_pattern(product_code, store_code)
        
        # 综合结果
        summary = {
            '商品编码': product_code,
            '门店编码': store_code,
            '总销量': int(total_sales),
            '总销售额': float(total_revenue),
            '平均售价': float(avg_price),
            '平均折扣率': float(avg_discount),
            '折扣标准差': float(discount_std),
            '促销频率': float(sales_pattern.get('discount_frequency', 0)),
            '价格弹性': float(sales_pattern.get('price_elasticity', 1.2)),
            '促销敏感度': float(sales_pattern.get('promo_performance', {}).get('promo_ratio', 1.2)),
            '销售趋势': float(sales_trend),
            '最佳销售时段': sales_pattern.get('best_selling_hours', [20, 21]),
            '最差销售时段': sales_pattern.get('worst_selling_hours', [2, 3, 4]),
            '天气影响': sales_pattern.get('weather_impact', {}),
            '节假日影响': sales_pattern.get('holiday_impact', {})
        }
        
        return summary
    
    def _get_default_product_summary(self, product_code: str) -> Dict:
        """获取默认商品汇总信息"""
        return {
            '商品编码': product_code,
            '门店编码': None,
            '总销量': 0,
            '总销售额': 0,
            '平均售价': 0,
            '平均折扣率': 1.0,
            '折扣标准差': 0,
            '促销频率': 0,
            '价格弹性': 1.2,
            '促销敏感度': 1.2,
            '销售趋势': 0,
            '最佳销售时段': [20, 21],
            '最差销售时段': [2, 3, 4],
            '天气影响': {'rain_impact': 1.0, 'hot_impact': 1.0, 'cold_impact': 1.0},
            '节假日影响': {'holiday_boost': 1.0, 'weekend_boost': 1.0}
        }