# utils/data_utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import json

def load_transaction_data(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """加载交易数据"""
    try:
        df = pd.read_csv(filepath, encoding=encoding)
        print(f"数据加载成功: {len(df)} 条记录")
        return df
    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return pd.DataFrame()

def validate_transaction_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """验证交易数据"""
    if df.empty:
        print("数据框为空")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"缺少必要列: {missing_columns}")
        return False
    
    # 检查关键列的缺失值
    critical_columns = ['商品编码', '销售数量', '销售金额']
    for col in critical_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"列 '{col}' 有 {missing_count} 个缺失值")
    
    return True

def prepare_time_features(df: pd.DataFrame, time_column: str = '交易时间') -> pd.DataFrame:
    """准备时间特征"""
    if df.empty or time_column not in df.columns:
        return df
    
    df_copy = df.copy()
    
    try:
        # 转换时间列
        df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        
        # 提取时间特征
        df_copy['year'] = df_copy[time_column].dt.year
        df_copy['month'] = df_copy[time_column].dt.month
        df_copy['day'] = df_copy[time_column].dt.day
        df_copy['hour'] = df_copy[time_column].dt.hour
        df_copy['minute'] = df_copy[time_column].dt.minute
        df_copy['day_of_week'] = df_copy[time_column].dt.dayofweek
        df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
        df_copy['quarter'] = df_copy[time_column].dt.quarter
        
        # 计算是否促销时段（20:00-22:00）
        df_copy['is_clearance_time'] = ((df_copy['hour'] >= 20) & (df_copy['hour'] < 22)).astype(int)
        
        print("时间特征提取完成")
        
    except Exception as e:
        print(f"提取时间特征时出错: {e}")
    
    return df_copy

def calculate_price_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """计算价格指标"""
    if df.empty or '售价' not in df.columns:
        return {}
    
    metrics = {
        'avg_price': df['售价'].mean(),
        'median_price': df['售价'].median(),
        'min_price': df['售价'].min(),
        'max_price': df['售价'].max(),
        'price_std': df['售价'].std(),
        'price_cv': df['售价'].std() / df['售价'].mean() if df['售价'].mean() > 0 else 0
    }
    
    return {k: float(v) for k, v in metrics.items() if not pd.isna(v)}

def calculate_sales_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """计算销售指标"""
    if df.empty or '销售数量' not in df.columns:
        return {}
    
    metrics = {
        'total_quantity': df['销售数量'].sum(),
        'avg_quantity': df['销售数量'].mean(),
        'median_quantity': df['销售数量'].median(),
        'quantity_std': df['销售数量'].std(),
        'total_transactions': len(df),
        'avg_transaction_value': df['销售金额'].mean() if '销售金额' in df.columns else 0
    }
    
    return {k: float(v) for k, v in metrics.items() if not pd.isna(v)}

def filter_by_date_range(df: pd.DataFrame, 
                        start_date: str, 
                        end_date: str,
                        date_column: str = '交易时间') -> pd.DataFrame:
    """按日期范围筛选数据"""
    if df.empty or date_column not in df.columns:
        return df
    
    try:
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # 转换输入日期
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # 筛选数据
        mask = (df[date_column] >= start) & (df[date_column] <= end)
        filtered_df = df[mask].copy()
        
        print(f"日期范围筛选: {start_date} 到 {end_date}, 保留 {len(filtered_df)} 条记录")
        
        return filtered_df
    
    except Exception as e:
        print(f"按日期范围筛选时出错: {e}")
        return df

def aggregate_by_time_unit(df: pd.DataFrame, 
                          time_unit: str = 'hour',
                          value_column: str = '销售数量') -> pd.DataFrame:
    """按时间单位聚合数据"""
    if df.empty or '交易时间' not in df.columns:
        return pd.DataFrame()
    
    try:
        # 确保时间列是datetime类型
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['交易时间']):
            df_copy['交易时间'] = pd.to_datetime(df_copy['交易时间'])
        
        # 设置时间索引
        df_copy.set_index('交易时间', inplace=True)
        
        # 按时间单位重采样
        if time_unit == 'hour':
            resampled = df_copy.resample('H')[value_column].sum()
        elif time_unit == 'day':
            resampled = df_copy.resample('D')[value_column].sum()
        elif time_unit == 'week':
            resampled = df_copy.resample('W')[value_column].sum()
        elif time_unit == 'month':
            resampled = df_copy.resample('M')[value_column].sum()
        else:
            raise ValueError(f"不支持的时间单位: {time_unit}")
        
        # 重置索引
        result = resampled.reset_index()
        result.columns = ['时间', f'{value_column}_总和']
        
        return result
    
    except Exception as e:
        print(f"按时间单位聚合时出错: {e}")
        return pd.DataFrame()

def detect_seasonal_patterns(df: pd.DataFrame, 
                            value_column: str = '销售数量',
                            freq: str = 'D') -> Dict[str, Any]:
    """检测季节性模式"""
    if df.empty or '交易时间' not in df.columns:
        return {}
    
    try:
        # 准备时间序列数据
        df_copy = df.copy()
        df_copy['交易时间'] = pd.to_datetime(df_copy['交易时间'])
        df_copy.set_index('交易时间', inplace=True)
        
        # 重采样
        if freq == 'H':
            ts = df_copy.resample('H')[value_column].sum()
        elif freq == 'D':
            ts = df_copy.resample('D')[value_column].sum()
        else:
            ts = df_copy[value_column]
        
        # 计算基本统计
        stats = {
            'mean': float(ts.mean()),
            'std': float(ts.std()),
            'min': float(ts.min()),
            'max': float(ts.max()),
            'autocorrelation': {}
        }
        
        # 计算自相关（滞后1-7）
        for lag in range(1, 8):
            if len(ts) > lag:
                autocorr = ts.autocorr(lag=lag)
                if not pd.isna(autocorr):
                    stats['autocorrelation'][f'lag_{lag}'] = float(autocorr)
        
        return stats
    
    except Exception as e:
        print(f"检测季节性模式时出错: {e}")
        return {}

def save_processed_data(df: pd.DataFrame, filepath: str):
    """保存处理后的数据"""
    try:
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存数据
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False, encoding='utf-8')
        elif filepath.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        elif filepath.endswith('.feather'):
            df.to_feather(filepath)
        else:
            print(f"不支持的文件格式: {filepath}")
            return False
        
        print(f"数据已保存到: {filepath}")
        return True
    
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return False