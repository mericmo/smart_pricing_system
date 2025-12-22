# 导入必要的库
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from IPython.display import display
warnings.filterwarnings('ignore')

# 特征仓库系统 (feature_store.py)
# FeatureStore类：

# 基础特征：销售数量、销售金额、售价等
# 时间特征：星期、月份、是否周末、是否月末等
# 滞后特征：过去1、3、7、14、30天的销量
# 滚动统计特征：7天/30天均值、标准差、最大值、最小值
# 趋势特征：7天/30天趋势、同比上周/上月
# 季节性特征：按星期/月份的季节性模式
# 天气特征：温度、降雨量、是否恶劣天气
# 节假日特征：是否节假日、节假日类型、节假日前/后一天
# 交互特征：促销与周末/月末交互、天气与促销交互
# 目标变量：未来7/14/30天销量

class FeatureStore:
    """
    负责所有需要复杂计算的特征工程
    包括时间序列特征、滞后特征、滚动特征等
    """
    
    def create_time_features(self, df):
        """创建时间相关特征"""
        df['星期'] = df['日期'].dt.dayofweek
        df['月份'] = df['日期'].dt.month
        df['是否周末'] = (df['日期'].dt.dayofweek >= 5).astype(int)
        df['是否月末'] = (df['日期'].dt.day >= 25).astype(int)
        df['季度'] = df['日期'].dt.quarter
        df['年份'] = df['日期'].dt.year
        df['日期序号'] = (df['日期'] - df['日期'].min()).dt.days
        
        # 季节性特征
        df['季节'] = df['日期'].dt.month.apply(self._get_season)
        
        return df
    
    def create_lag_features(self, df, lags=[1, 3, 7, 14, 30]):
        """创建滞后特征 - 需要按商品单独计算"""
        df = df.sort_values(['商品编码', '日期'])
        # 生成前面第x天的销量信息
        for product_id in df['商品编码'].unique():
            product_mask = df['商品编码'] == product_id
            for lag in lags:
                df.loc[product_mask, f'销量_滞后{lag}天'] = (
                    df.loc[product_mask, '销售数量'].shift(lag)
                )
        
        return df

    def create_rolling_features(self, df, windows=[7, 14, 30]):
        """创建滚动特征 - 需要按商品单独计算"""
        # 1. 首先按商品编码和日期排序（与滞后特征相同）
        df = df.sort_values(['商品编码', '日期'])

        # 2. 遍历每个商品
        for product_id in df['商品编码'].unique():
            # 创建布尔掩码，筛选出当前商品的所有数据
            product_mask = df['商品编码'] == product_id

            # 3. 为每个窗口大小创建滚动特征
            for window in windows:  # windows = [7, 30]
                # 3.1 滚动均值（移动平均）
                df.loc[product_mask, f'销量_{window}天均值'] = (
                    # rolling(window, min_periods=1) 创建滚动窗口
                    # window=7 表示使用最近7天的数据
                    # min_periods=1 表示至少需要1个数据点（避免NaN）
                    df.loc[product_mask, '销售数量'].rolling(window, min_periods=1).mean()
                )

                # 3.2 滚动标准差（波动性指标）
                df.loc[product_mask, f'销量_{window}天标准差'] = (
                    df.loc[product_mask, '销售数量'].rolling(window, min_periods=1).std()
                )
        
        return df
    
    def create_trend_features(self, df):
        """创建趋势特征"""
        # 1. 首先按商品编码和日期排序
        df = df.sort_values(['商品编码', '日期'])

        # 2. 遍历每个商品（确保按商品独立计算）
        for product_id in df['商品编码'].unique():
            product_mask = df['商品编码'] == product_id

            # 3. 创建7天趋势特征
            # 销量_7天趋势 = (当前7天均值 - 7天前的7天均值) / 7天前的7天均值
            #             = 7天均值在7天内的相对变化率
            df.loc[product_mask, '销量_7天趋势'] = (
                # pct_change(periods=7) 计算7天前的百分比变化
                df.loc[product_mask, '销量_7天均值'].pct_change(periods=7)
            )

            # 4. 创建同比上周特征 销量_同比上周 = 7天前（同星期几）的实际销量
            df.loc[product_mask, '销量_同比上周'] = (
                # shift(7) 获取7天前的销售数量
                df.loc[product_mask, '销售数量'].shift(7)
            )
        
        return df
    
    def create_weather_enhanced_features(self, df, weather_df):
        """创建增强的天气特征"""
        # 按日期合并天气数据
        df = df.merge(weather_df, on='日期', how='left')
        
        # 天气分段特征 - 使用平均温度
        df['温度等级'] = pd.cut(df['温度'], 
                              bins=[-np.inf, 0, 10, 20, 30, np.inf],
                              labels=['严寒', '寒冷', '凉爽', '温暖', '炎热'])
        
        # 温度差特征
        if '最高温度' in df.columns and '最低温度' in df.columns:
            df['温度差'] = df['最高温度'] - df['最低温度']
            df['温度差等级'] = pd.cut(df['温度差'],
                                  bins=[-1, 5, 10, 15, np.inf],
                                  labels=['小温差', '中温差', '大温差', '极大温差'])
        
        # 是否恶劣天气 - 调整阈值以适应深圳气候
        df['是否恶劣天气'] = (
            (df['温度'] < 10) | (df['温度'] > 35) | (df['降雨量'] > 25)
        ).astype(int)
        
        return df
    
    def create_calendar_enhanced_features(self, df, calendar_df):
        """创建增强的日历特征"""
        # 按日期合并日历数据
        df = df.merge(calendar_df, on='日期', how='left')
        print('\n合并后数据示例:')
        display(df.head(1))
        # 节假日前后的特征
        df['节假日前1天'] = df['是否节假日'].shift(1).fillna(0)
        df['节假日后1天'] = df['是否节假日'].shift(-1).fillna(0)
        
        # 节假日连续天数
        df['节假日连续天数'] = self._calculate_holiday_streak(df)
        
        return df
    
    def create_interaction_features(self, df):
        """创建交互特征"""
        # 促销与时间的交互
        df['促销_周末交互'] = np.where(
            df['实际折扣率'] != 0,  # 条件
            df['是否周末'] * df['销售数量'] / df['实际折扣率'],  # 条件为真时的值
            0  # 条件为假时的值
        )
        df['促销_月末交互'] = np.where(
            df['实际折扣率'] != 0,  # 条件
            df['是否月末'] * df['销售数量'] / df['实际折扣率'],  # 条件为真时的值
            0  # 条件为假时的值
        )


        # 天气与促销的交互
        if '是否恶劣天气' in df.columns:
            df['天气_促销交互'] = np.where(
                df['实际折扣率'] != 0,  # 条件
                df['是否恶劣天气'] * df['销售数量'] / df['实际折扣率'],  # 条件为真时的值
                0  # 条件为假时的值
            )
        
        return df
    
    def _get_season(self, month):
        """获取季节"""
        if month in [12, 1, 2]:
            return '冬季'
        elif month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        else:
            return '秋季'
    
    def _calculate_holiday_streak(self, df):
        """计算节假日连续天数"""
        streak = 0
        streaks = []
        
        for is_holiday in df['是否节假日']:
            if is_holiday == 1:
                streak += 1
            else:
                streak = 0
            streaks.append(streak)
        
        return streaks