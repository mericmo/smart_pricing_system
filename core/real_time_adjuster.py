# core/real_time_adjuster.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# from core.pricing_strategy_generator import PricingStrategy
from core.pricing_strategy_generator import EnhancedPricingStrategy as PricingStrategy
@dataclass
class SalesUpdate:
    """销售更新数据"""
    timestamp: datetime
    product_code: str
    quantity_sold: int
    actual_price: float
    discount_applied: float
    remaining_stock: int

class RealTimeAdjuster:
    """实时调整器"""
    
    def __init__(self, strategy_generator):
        """
        初始化实时调整器
        
        Args:
            strategy_generator: 定价策略生成器实例
        """
        self.strategy_generator = strategy_generator
        self.sales_updates = {}  # strategy_id -> List[SalesUpdate]
        self.adjustment_history = {}  # strategy_id -> List[调整记录]
    
    def record_sales(self, strategy_id: str, sales_update: SalesUpdate):
        """记录销售数据"""
        
        if strategy_id not in self.sales_updates:
            self.sales_updates[strategy_id] = []
        
        self.sales_updates[strategy_id].append(sales_update)
        
        # 按时间排序
        self.sales_updates[strategy_id].sort(key=lambda x: x.timestamp)
    
    def check_and_adjust(self, strategy: PricingStrategy,
                        current_time: Optional[datetime] = None) -> Optional[PricingStrategy]:
        """检查并调整策略"""
        
        if current_time is None:
            current_time = datetime.now()
        
        strategy_id = strategy.strategy_id
        if not strategy_id or strategy_id not in self.sales_updates:
            return None  # 没有销售数据，无法调整
        
        sales_data = self.sales_updates[strategy_id]
        if not sales_data:
            return None
        
        # 分析销售表现
        analysis = self._analyze_sales_performance(strategy, sales_data, current_time)
        
        # 检查是否需要调整
        if not self._needs_adjustment(analysis):
            return None
        
        # 生成调整后的策略
        adjusted_strategy = self._generate_adjusted_strategy(strategy, analysis, current_time)
        
        # 记录调整
        self._record_adjustment(strategy_id, analysis, adjusted_strategy)
        
        return adjusted_strategy
    
    def _analyze_sales_performance(self, strategy: PricingStrategy,
                                 sales_data: List[SalesUpdate],
                                 current_time: datetime) -> Dict:
        """分析销售表现"""
        
        # 计算预期 vs 实际
        total_expected = 0
        total_actual = 0
        
        # 解析促销时间
        promo_start = datetime.strptime(strategy.promotion_start, "%H:%M").time()
        promo_end = datetime.strptime(strategy.promotion_end, "%H:%M").time()
        
        # 获取当前促销阶段
        current_stage = self._get_current_stage(strategy, current_time)
        
        # 计算当前阶段的表现
        stage_start_time = self._parse_time_string(current_stage['start_time'])
        stage_end_time = self._parse_time_string(current_stage['end_time'])
        
        stage_sales = []
        for sale in sales_data:
            sale_time = sale.timestamp.time()
            if stage_start_time <= sale_time < stage_end_time:
                stage_sales.append(sale)
        
        # 计算偏差
        if stage_sales:
            actual_sales = sum(s.quantity_sold for s in stage_sales)
            expected_sales = current_stage['expected_sales']
            
            if expected_sales > 0:
                deviation = (actual_sales - expected_sales) / expected_sales
            else:
                deviation = 0
            
            # 计算销售速率
            if len(stage_sales) >= 2:
                first_sale = min(stage_sales, key=lambda x: x.timestamp)
                last_sale = max(stage_sales, key=lambda x: x.timestamp)
                
                time_diff = (last_sale.timestamp - first_sale.timestamp).total_seconds() / 3600  # 小时
                if time_diff > 0:
                    sales_rate = actual_sales / time_diff
                else:
                    sales_rate = 0
            else:
                sales_rate = 0
        else:
            actual_sales = 0
            expected_sales = current_stage['expected_sales']
            deviation = -1.0 if expected_sales > 0 else 0
            sales_rate = 0
        
        # 计算剩余库存和时间
        if sales_data:
            latest_sale = max(sales_data, key=lambda x: x.timestamp)
            remaining_stock = latest_sale.remaining_stock
        else:
            remaining_stock = strategy.initial_stock
        
        # 计算剩余时间比例
        current_hour = current_time.hour + current_time.minute / 60
        promo_start_hour = promo_start.hour + promo_start.minute / 60
        promo_end_hour = promo_end.hour + promo_end.minute / 60
        
        if promo_end_hour <= promo_start_hour:
            promo_end_hour += 24
            if current_hour < promo_start_hour:
                current_hour += 24
        
        elapsed_time = current_hour - promo_start_hour
        total_time = promo_end_hour - promo_start_hour
        time_remaining_ratio = max(0, 1 - elapsed_time / total_time)
        
        return {
            'current_stage': current_stage,
            'actual_sales': actual_sales,
            'expected_sales': expected_sales,
            'deviation': deviation,
            'sales_rate': sales_rate,
            'remaining_stock': remaining_stock,
            'time_remaining_ratio': time_remaining_ratio,
            'elapsed_time_ratio': 1 - time_remaining_ratio,
            'on_track_score': self._calculate_on_track_score(actual_sales, expected_sales, time_remaining_ratio)
        }
    
    def _get_current_stage(self, strategy: PricingStrategy, current_time: datetime) -> Dict:
        """获取当前阶段"""
        
        current_time_str = current_time.strftime("%H:%M")
        
        for stage in strategy.pricing_schedule:
            if stage['start_time'] <= current_time_str < stage['end_time']:
                return stage
        
        # 如果不在任何阶段内，返回最后一个阶段
        return strategy.pricing_schedule[-1]
    
    def _parse_time_string(self, time_str: str) -> datetime.time:
        """解析时间字符串"""
        return datetime.strptime(time_str, "%H:%M").time()
    
    def _calculate_on_track_score(self, actual_sales: int, expected_sales: int,
                                 time_remaining_ratio: float) -> float:
        """计算是否在正轨上的分数"""
        
        if expected_sales <= 0:
            return 0.5
        
        # 计算预期进度
        expected_progress = 1 - time_remaining_ratio
        actual_progress = actual_sales / expected_sales if expected_sales > 0 else 0
        
        # 计算分数
        if actual_progress >= expected_progress:
            # 超前或正常
            score = 0.5 + 0.5 * min(actual_progress / max(expected_progress, 0.1), 2.0)
        else:
            # 落后
            score = 0.5 * (actual_progress / max(expected_progress, 0.1))
        
        return min(max(score, 0), 1)
    
    def _needs_adjustment(self, analysis: Dict) -> bool:
        """判断是否需要调整"""
        
        on_track_score = analysis['on_track_score']
        deviation = analysis['deviation']
        
        # 如果严重落后或超前，需要调整
        if on_track_score < 0.3 or on_track_score > 1.5:
            return True
        
        # 如果偏差超过阈值，需要调整
        if abs(deviation) > 0.3:  # 30%偏差
            return True
        
        # 如果销售速率为0但应该有销售，需要调整
        if analysis['sales_rate'] == 0 and analysis['expected_sales'] > 0:
            return True
        
        return False
    
    def _generate_adjusted_strategy(self, original_strategy: PricingStrategy,
                                  analysis: Dict,
                                  current_time: datetime) -> PricingStrategy:
        """生成调整后的策略"""
        
        # 计算剩余需要销售的库存
        remaining_stock = analysis['remaining_stock']
        
        if remaining_stock <= 0:
            # 已经售罄，不需要调整
            return original_strategy
        
        # 计算剩余时间
        current_time_str = current_time.strftime("%H:%M")
        
        # 找到当前及后续的阶段
        remaining_stages = []
        found_current = False
        
        for stage in original_strategy.pricing_schedule:
            if stage['start_time'] <= current_time_str or not remaining_stages:
                if not found_current and stage['start_time'] <= current_time_str <= stage['end_time']:
                    found_current = True
                    # 调整当前阶段
                    adjusted_stage = self._adjust_current_stage(stage, analysis)
                    remaining_stages.append(adjusted_stage)
                elif found_current or stage['start_time'] > current_time_str:
                    remaining_stages.append(stage)
        
        # 重新优化剩余阶段的定价
        # 这里简化处理，实际应该重新运行优化算法
        
        # 调整折扣力度
        adjustment_factor = self._calculate_adjustment_factor(analysis)
        
        for stage in remaining_stages:
            # 根据调整因子调整折扣
            if adjustment_factor > 1.1:  # 销售太慢
                # 加大折扣
                stage['discount'] = max(
                    original_strategy.min_discount,
                    stage['discount'] * 0.9  # 增加10%折扣力度
                )
            elif adjustment_factor < 0.9:  # 销售太快
                # 减少折扣
                stage['discount'] = min(
                    original_strategy.max_discount,
                    stage['discount'] * 1.1  # 减少10%折扣力度
                )
            
            # 更新价格
            stage['price'] = original_strategy.original_price * stage['discount']
        
        # 重新计算预期销量（简化）
        total_remaining_time = sum(self._calculate_time_duration(s['start_time'], s['end_time']) 
                                  for s in remaining_stages)
        
        if total_remaining_time > 0:
            avg_sales_rate = remaining_stock / total_remaining_time
            for stage in remaining_stages:
                stage_duration = self._calculate_time_duration(stage['start_time'], stage['end_time'])
                stage['expected_sales'] = int(avg_sales_rate * stage_duration * 
                                            (1 / stage['discount']))  # 考虑折扣影响
        
        # 创建新的策略（简化，实际应该重新生成完整策略）
        adjusted_strategy = PricingStrategy(
            product_code=original_strategy.product_code,
            product_name=original_strategy.product_name,
            original_price=original_strategy.original_price,
            cost_price=original_strategy.cost_price,
            initial_stock=remaining_stock,
            promotion_start=current_time_str,
            promotion_end=original_strategy.promotion_end,
            min_discount=original_strategy.min_discount,
            max_discount=original_strategy.max_discount,
            time_segments=len(remaining_stages),
            pricing_schedule=remaining_stages,
            evaluation=self._generate_adjusted_evaluation(original_strategy, remaining_stages, remaining_stock),
            generated_time=current_time.isoformat(),
            strategy_id=f"{original_strategy.strategy_id}_ADJ"
        )
        
        return adjusted_strategy
    
    def _adjust_current_stage(self, stage: Dict, analysis: Dict) -> Dict:
        """调整当前阶段"""
        
        adjusted_stage = stage.copy()
        
        # 根据偏差调整预期销量
        if analysis['deviation'] < -0.2:  # 销售落后20%以上
            # 减少预期销量
            adjusted_stage['expected_sales'] = int(stage['expected_sales'] * 0.8)
        elif analysis['deviation'] > 0.2:  # 销售超前20%以上
            # 增加预期销量
            adjusted_stage['expected_sales'] = int(stage['expected_sales'] * 1.2)
        
        return adjusted_stage
    
    def _calculate_adjustment_factor(self, analysis: Dict) -> float:
        """计算调整因子"""
        
        on_track_score = analysis['on_track_score']
        
        if on_track_score < 0.3:
            return 1.5  # 需要大幅增加折扣
        elif on_track_score < 0.7:
            return 1.2  # 需要增加折扣
        elif on_track_score > 1.5:
            return 0.7  # 需要减少折扣
        elif on_track_score > 1.2:
            return 0.9  # 需要稍微减少折扣
        else:
            return 1.0  # 不需要调整
    
    def _calculate_time_duration(self, start_time: str, end_time: str) -> float:
        """计算时间段长度（小时）"""
        
        start = datetime.strptime(start_time, "%H:%M")
        end = datetime.strptime(end_time, "%H:%M")
        
        if end <= start:
            end = datetime.strptime("23:59", "%H:%M")
            duration1 = (end - start).seconds / 3600
            duration2 = datetime.strptime(end_time, "%H:%M").hour / 24
            return duration1 + duration2
        else:
            return (end - start).seconds / 3600
    
    def _generate_adjusted_evaluation(self, original_strategy: PricingStrategy,
                                    adjusted_schedule: List[Dict],
                                    remaining_stock: int) -> Dict:
        """生成调整后的评估"""
        
        total_expected_sales = sum(stage['expected_sales'] for stage in adjusted_schedule)
        total_revenue = sum(stage['expected_revenue'] for stage in adjusted_schedule)
        total_profit = sum(stage['expected_profit'] for stage in adjusted_schedule)
        
        sell_out_probability = min(1.0, total_expected_sales / remaining_stock) if remaining_stock > 0 else 0
        
        return {
            'total_expected_sales': total_expected_sales,
            'total_revenue': round(total_revenue, 2),
            'total_profit': round(total_profit, 2),
            'remaining_stock': remaining_stock,
            'sell_out_probability': round(sell_out_probability, 3),
            'adjustment_reason': '基于实时销售表现调整',
            'original_strategy_id': original_strategy.strategy_id
        }
    
    def _record_adjustment(self, strategy_id: str, analysis: Dict, adjusted_strategy: PricingStrategy):
        """记录调整"""
        
        if strategy_id not in self.adjustment_history:
            self.adjustment_history[strategy_id] = []
        
        adjustment_record = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'adjusted_strategy_id': adjusted_strategy.strategy_id,
            'reason': '销售偏离预期' if abs(analysis['deviation']) > 0.2 else '进度不匹配'
        }
        
        self.adjustment_history[strategy_id].append(adjustment_record)
    
    def get_adjustment_summary(self, strategy_id: str) -> Dict:
        """获取调整摘要"""
        
        if strategy_id not in self.adjustment_history:
            return {'adjustment_count': 0, 'adjustments': []}
        
        adjustments = self.adjustment_history[strategy_id]
        
        return {
            'adjustment_count': len(adjustments),
            'last_adjustment': adjustments[-1] if adjustments else None,
            'adjustments': adjustments[-5:]  # 最近5次调整
        }