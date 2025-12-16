# data/data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TransactionDataProcessor:
    """交易数据处理器"""
    
    def __init__(self, transaction_data: pd.DataFrame):
        """
        初始化处理器
        transaction_data: 交易数据DataFrame，包含字段：
          日期,门店编码,流水单号,会员id,交易时间,渠道名称,平台触点名称,
          小类编码,商品编码,商品名称,售价,折扣类型,税率,销售数量,
          销售金额,销售净额,折扣金额
        """
        self.transaction_data = transaction_data.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """数据预处理"""
        # 确保时间字段格式正确
        if '交易时间' in self.transaction_data.columns:
            self.transaction_data['交易时间'] = pd.to_datetime(
                self.transaction_data['交易时间']
            )
        
        if '日期' in self.transaction_data.columns:
            self.transaction_data['日期'] = pd.to_datetime(
                self.transaction_data['日期']
            )
        
        # 计算实际折扣率
        if '销售金额' in self.transaction_data.columns and '售价' in self.transaction_data.columns:
            self.transaction_data['实际折扣率'] = self.transaction_data.apply(
                self._calculate_actual_discount, axis=1
            )
        
        # 提取时间特征
        self._extract_time_features()
    
    def _calculate_actual_discount(self, row):
        """计算实际折扣率"""
        try:
            if row['售价'] > 0:
                # 销售金额 / (售价 * 销售数量)
                return row['销售金额'] / (row['售价'] * row['销售数量'])
            return 1.0
        except:
            return 1.0
    
    def _extract_time_features(self):
        """提取时间特征"""
        if '交易时间' in self.transaction_data.columns:
            self.transaction_data['小时'] = self.transaction_data['交易时间'].dt.hour
            self.transaction_data['分钟'] = self.transaction_data['交易时间'].dt.minute
            self.transaction_data['星期几'] = self.transaction_data['交易时间'].dt.dayofweek
            self.transaction_data['是否周末'] = self.transaction_data['星期几'].isin([5, 6]).astype(int)
            self.transaction_data['月份'] = self.transaction_data['交易时间'].dt.month
            self.transaction_data['季度'] = self.transaction_data['交易时间'].dt.quarter
    
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
        if start_hour <= end_hour:
            mask = (self.transaction_data['小时'] >= start_hour) & (self.transaction_data['小时'] < end_hour)
        else:
            # 跨天时间段
            mask = (self.transaction_data['小时'] >= start_hour) | (self.transaction_data['小时'] < end_hour)
        
        return self.transaction_data[mask].copy()
    
    def get_product_summary(self, product_code: str) -> Dict:
        """获取商品汇总信息"""
        product_data = self.filter_by_product(product_code)
        
        if product_data.empty:
            return {
                '商品编码': product_code,
                '总销量': 0,
                '总销售额': 0,
                '平均折扣率': 1.0,
                '价格弹性': 1.2,  # 默认值
                '促销敏感度': 1.0
            }
        
        summary = {
            '商品编码': product_code,
            '总销量': product_data['销售数量'].sum(),
            '总销售额': product_data['销售金额'].sum(),
            '平均售价': product_data['售价'].mean(),
            '平均折扣率': product_data['实际折扣率'].mean() if '实际折扣率' in product_data.columns else 1.0,
            '促销频率': self._calculate_promotion_frequency(product_data),
            '销量标准差': product_data['销售数量'].std(),
            '价格弹性': self._estimate_price_elasticity(product_data),
            '促销敏感度': self._calculate_promotion_sensitivity(product_data)
        }
        
        return summary
    
    def _calculate_promotion_frequency(self, product_data: pd.DataFrame) -> float:
        """计算促销频率"""
        if '折扣类型' in product_data.columns:
            # 统计有折扣的交易比例
            has_discount = product_data['折扣类型'].notna() & (product_data['折扣类型'] != '无折扣')
            return has_discount.mean()
        elif '实际折扣率' in product_data.columns:
            # 折扣率小于1视为促销
            has_discount = product_data['实际折扣率'] < 0.99
            return has_discount.mean()
        return 0.0
    
    def _estimate_price_elasticity(self, product_data: pd.DataFrame) -> float:
        """估算价格弹性"""
        if '实际折扣率' not in product_data.columns or len(product_data) < 10:
            return 1.2  # 默认值
        
        # 按折扣率分组计算平均销量
        product_data['折扣区间'] = pd.cut(
            product_data['实际折扣率'], 
            bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0],
            labels=['5折以下', '5-7折', '7-85折', '85-95折', '原价']
        )
        
        grouped = product_data.groupby('折扣区间')['销售数量'].mean()
        
        # 计算弹性系数（销量变化率/价格变化率）
        if len(grouped) >= 2:
            # 找到有折扣和无折扣的对比
            if '原价' in grouped.index and '7-85折' in grouped.index:
                price_change = 0.85 - 1.0  # 价格变化
                quantity_change = (grouped['7-85折'] - grouped['原价']) / grouped['原价']  # 销量变化率
                if price_change != 0:
                    elasticity = abs(quantity_change / price_change)
                    return max(0.5, min(elasticity, 3.0))  # 限制在合理范围
        
        return 1.2
    
    def _calculate_promotion_sensitivity(self, product_data: pd.DataFrame) -> float:
        """计算促销敏感度"""
        if '实际折扣率' not in product_data.columns:
            return 1.0
        
        # 促销日 vs 非促销日的销量对比
        product_data['是否促销'] = product_data['实际折扣率'] < 0.99
        
        if product_data['是否促销'].nunique() == 2:
            sensitivity = product_data.groupby('是否促销')['销售数量'].mean()
            if False in sensitivity.index and True in sensitivity.index:
                return sensitivity[True] / max(sensitivity[False], 1)
        
        return 1.0
    
    def get_time_based_sales_pattern(self, product_code: str, 
                                    time_segments: List[Tuple[int, int]]) -> Dict:
        """获取基于时间段的销售模式"""
        product_data = self.filter_by_product(product_code)
        
        if product_data.empty:
            # 返回默认模式
            return {
                segment: {'平均销量': 5, '平均折扣率': 0.85, '交易次数': 10}
                for segment in time_segments
            }
        
        patterns = {}
        for i, (start_hour, end_hour) in enumerate(time_segments):
            # 筛选该时间段的数据
            if start_hour <= end_hour:
                segment_data = product_data[
                    (product_data['小时'] >= start_hour) & 
                    (product_data['小时'] < end_hour)
                ]
            else:
                segment_data = product_data[
                    (product_data['小时'] >= start_hour) | 
                    (product_data['小时'] < end_hour)
                ]
            
            if not segment_data.empty:
                patterns[f"时段{i+1}"] = {
                    '平均销量': segment_data['销售数量'].mean(),
                    '平均折扣率': segment_data['实际折扣率'].mean() if '实际折扣率' in segment_data.columns else 1.0,
                    '交易次数': len(segment_data),
                    '销售总量': segment_data['销售数量'].sum(),
                    '时间段': f"{start_hour}:00-{end_hour}:00"
                }
            else:
                patterns[f"时段{i+1}"] = {
                    '平均销量': 0,
                    '平均折扣率': 1.0,
                    '交易次数': 0,
                    '销售总量': 0,
                    '时间段': f"{start_hour}:00-{end_hour}:00"
                }
        
        return patterns
    
    def get_customer_price_sensitivity(self, product_code: str) -> Dict:
        """分析顾客价格敏感性"""
        product_data = self.filter_by_product(product_code)
        
        if product_data.empty or '会员id' not in product_data.columns:
            return {
                'price_sensitive_customers': 0.3,  # 价格敏感顾客比例
                'loyal_customers': 0.2,  # 忠诚顾客比例
                'average_basket_size': 1.5,  # 平均购买数量
                'repeat_purchase_rate': 0.1  # 重复购买率
            }
        
        # 分析会员购买行为
        member_data = product_data[product_data['会员id'].notna()]
        
        if not member_data.empty:
            # 价格敏感度：购买折扣商品的会员比例
            if '实际折扣率' in member_data.columns:
                price_sensitive = member_data[member_data['实际折扣率'] < 0.95]
                price_sensitive_ratio = len(price_sensitive) / len(member_data)
            else:
                price_sensitive_ratio = 0.3
            
            # 忠诚顾客：多次购买的比例
            purchase_counts = member_data['会员id'].value_counts()
            loyal_customers = (purchase_counts >= 2).sum() / len(purchase_counts) if len(purchase_counts) > 0 else 0
            
            # 平均购买数量
            avg_basket_size = member_data['销售数量'].mean()
            
            # 重复购买率
            repeat_rate = (purchase_counts > 1).sum() / len(purchase_counts) if len(purchase_counts) > 0 else 0
            
            return {
                'price_sensitive_customers': min(price_sensitive_ratio, 0.8),
                'loyal_customers': min(loyal_customers, 0.5),
                'average_basket_size': max(avg_basket_size, 1.0),
                'repeat_purchase_rate': min(repeat_rate, 0.3)
            }
        
        return {
            'price_sensitive_customers': 0.3,
            'loyal_customers': 0.2,
            'average_basket_size': 1.5,
            'repeat_purchase_rate': 0.1
        }