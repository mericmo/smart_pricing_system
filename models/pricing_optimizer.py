# models/pricing_optimizer.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

@dataclass
class PricingSegment:
    """定价时段"""
    start_time: str  # 格式: "HH:MM"
    end_time: str    # 格式: "HH:MM"
    discount: float  # 折扣率，0.4表示4折
    expected_sales: int  # 预期销量
    price: float     # 实际价格
    revenue: float   # 预期收入
    profit: float    # 预期利润

class PricingOptimizer:
    """定价优化器"""
    
    def __init__(self, demand_predictor, cost_price: float, original_price: float):
        """
        初始化定价优化器
        """
        self.demand_predictor = demand_predictor
        self.cost_price = cost_price
        self.original_price = original_price
        
        # 计算最低可接受折扣
        self.min_discount = max(cost_price / original_price, 0.4)  # 不低于成本，且不低于4折
    
    def optimize_staged_pricing(self, initial_stock: int,
                               promotion_start: str,
                               promotion_end: str,
                               min_discount: float = 0.4,
                               max_discount: float = 0.9,
                               time_segments: int = 4,
                               features: Dict = None) -> List[PricingSegment]:
        """
        优化阶梯定价
        """
        
        # 解析时间
        start_hour, start_minute = map(int, promotion_start.split(':'))
        end_hour, end_minute = map(int, promotion_end.split(':'))
        
        # 计算总促销时长（小时）
        start_time_minutes = start_hour * 60 + start_minute
        end_time_minutes = end_hour * 60 + end_minute
        if end_time_minutes <= start_time_minutes:
            end_time_minutes += 24 * 60  # 跨天
        
        total_minutes = end_time_minutes - start_time_minutes
        segment_minutes = total_minutes // time_segments
        
        # 动态规划优化
        optimal_schedule = self._dynamic_programming_optimization(
            initial_stock=initial_stock,
            total_minutes=total_minutes,
            segment_minutes=segment_minutes,
            min_discount=min_discount,
            max_discount=max_discount,
            features=features,
            start_hour=start_hour,
            start_minute=start_minute
        )
        
        return optimal_schedule
    
    def _dynamic_programming_optimization(self, initial_stock: int,
                                        total_minutes: int,
                                        segment_minutes: int,
                                        min_discount: float,
                                        max_discount: float,
                                        features: Dict,
                                        start_hour: int,
                                        start_minute: int) -> List[PricingSegment]:
        """动态规划优化"""
        
        # 离散化折扣空间
        discount_levels = np.linspace(min_discount, max_discount, num=5) #num=20
        
        # 计算时段数量
        num_segments = total_minutes // segment_minutes
        if total_minutes % segment_minutes != 0:
            num_segments += 1
        
        # 初始化DP表
        dp = np.full((num_segments + 1, initial_stock + 1), -np.inf)
        dp[0, initial_stock] = 0  # 初始状态：剩余库存为initial_stock，利润为0
        
        # 决策记录表
        decisions = np.full((num_segments, initial_stock + 1), -1, dtype=int)
        
        # 动态规划
        for t in range(num_segments):
            for s in range(initial_stock + 1):  # 剩余库存
                if dp[t, s] == -np.inf:
                    continue
                
                # 当前时段剩余时间比例
                time_remaining = 1.0 - (t * segment_minutes) / total_minutes
                
                # 尝试所有可能的折扣
                for i, discount in enumerate(discount_levels):
                    # 预测销售量
                    predicted_sales = self.demand_predictor.predict_demand(
                        features=features,
                        discount_rate=discount,
                        time_to_close=time_remaining,
                        current_stock=s
                    )  # 移除了base_demand参数
                    
                    # 实际销售量不能超过库存
                    actual_sales = min(round(predicted_sales), s)
                    
                    # 计算利润
                    price = self.original_price * discount
                    profit_per_unit = price - self.cost_price
                    segment_profit = profit_per_unit * actual_sales
                    
                    # 更新状态
                    new_stock = s - actual_sales
                    new_profit = dp[t, s] + segment_profit
                    
                    if new_profit > dp[t + 1, new_stock]:
                        dp[t + 1, new_stock] = new_profit
                        decisions[t, s] = i
        
        # 回溯找到最优解
        # 找到最终状态（利润最高的状态）
        final_segment = num_segments
        final_stock = np.argmax(dp[final_segment, :])
        max_profit = dp[final_segment, final_stock]
        
        # 回溯构建定价方案
        schedule = []
        current_stock = initial_stock
        
        for t in range(num_segments):
            if decisions[t, current_stock] == -1:
                break
            
            discount_idx = decisions[t, current_stock]
            discount = discount_levels[discount_idx]
            
            # 计算时间
            segment_start_minutes = start_minute + t * segment_minutes
            segment_start_hour = start_hour + segment_start_minutes // 60
            segment_start_minute = segment_start_minutes % 60
            
            segment_end_minutes = segment_start_minutes + segment_minutes
            segment_end_hour = start_hour + segment_end_minutes // 60
            segment_end_minute = segment_end_minutes % 60
            
            # 格式化时间
            start_time_str = f"{segment_start_hour % 24:02d}:{segment_start_minute:02d}"
            end_time_str = f"{segment_end_hour % 24:02d}:{segment_end_minute:02d}"
            
            # 预测销售量
            time_remaining = 1.0 - (t * segment_minutes) / total_minutes
            predicted_sales = self.demand_predictor.predict_demand(
                features=features,
                discount_rate=discount,
                time_to_close=time_remaining,
                current_stock=current_stock
            )
            actual_sales = min(int(predicted_sales), current_stock)
            
            # 计算收入利润
            price = self.original_price * discount
            revenue = price * actual_sales
            profit = (price - self.cost_price) * actual_sales
            
            # 创建定价时段
            segment = PricingSegment(
                start_time=start_time_str,
                end_time=end_time_str,
                discount=discount,
                expected_sales=actual_sales,
                price=round(price, 2),
                revenue=round(revenue, 2),
                profit=round(profit, 2)
            )
            
            schedule.append(segment)
            current_stock -= actual_sales
            
            if current_stock <= 0:
                break
        
        return schedule
    
    def _evaluate_discount_profit(self, discount: float,
                                 initial_stock: int,
                                 promotion_hours: Tuple[int, int],
                                 features: Dict) -> float:
        """评估折扣的预期利润 - 修复方法调用"""
        
        # start_hour, end_hour = promotion_hours
        # total_hours = (end_hour - start_hour) % 24
        
        # 预测总销量
        predicted_sales = 0
        remaining_stock = initial_stock
        
        # 将促销时间分为若干小段进行预测
        num_subsegments = 8
        for i in range(num_subsegments):
            time_elapsed = i / num_subsegments
            time_remaining = 1.0 - time_elapsed
            
            segment_sales = self.demand_predictor.predict_demand(
                features=features,
                discount_rate=discount,
                time_to_close=time_remaining,
                current_stock=remaining_stock
            )  # 移除了base_demand参数
            
            actual_sales = min(int(segment_sales / num_subsegments), remaining_stock)
            predicted_sales += actual_sales
            remaining_stock -= actual_sales
            
            if remaining_stock <= 0:
                break
        
        # 计算利润
        price = self.original_price * discount
        profit = (price - self.cost_price) * predicted_sales
        
        # 添加库存剩余惩罚
        if remaining_stock > 0:
            profit -= remaining_stock * self.cost_price * 0.3  # 未售出损失成本的30%
        
        return profit

    def evaluate_pricing_schedule(self, schedule: List[PricingSegment],
                                  initial_stock: int) -> Dict:
        """评估定价方案"""

        total_expected_sales = sum(segment.expected_sales for segment in schedule)
        total_revenue = sum(segment.revenue for segment in schedule)
        total_profit = sum(segment.profit for segment in schedule)
        remaining_stock = max(0, initial_stock - total_expected_sales)

        # 计算售罄概率
        sell_out_probability = min(1.0, total_expected_sales / initial_stock) if initial_stock > 0 else 0

        # 计算利润率
        profit_margin = total_profit / total_revenue if total_revenue > 0 else 0

        # 计算折扣深度
        avg_discount = np.mean([segment.discount for segment in schedule])

        evaluation = {
            'total_expected_sales': total_expected_sales,
            'total_revenue': round(total_revenue, 2),
            'total_profit': round(total_profit, 2),
            'remaining_stock': remaining_stock,
            'sell_out_probability': round(sell_out_probability, 3),
            'profit_margin': round(profit_margin, 3),
            'average_discount': round(avg_discount, 3),
            'stock_clearance_rate': round(total_expected_sales / initial_stock, 3) if initial_stock > 0 else 0,
            'recommendation': self._generate_recommendation(sell_out_probability, profit_margin)
        }

        return evaluation

    def _generate_recommendation(self, sell_out_probability: float,
                                 profit_margin: float) -> str:
        """生成推荐建议"""

        if sell_out_probability < 0.7:
            if profit_margin < 0.1:
                return "风险较高：售罄概率低且利润薄，建议加大折扣力度或考虑捆绑销售"
            else:
                return "售罄风险：建议适当增加折扣以提升售罄概率"
        elif sell_out_probability < 0.9:
            if profit_margin > 0.2:
                return "良好平衡：售罄概率和利润均表现良好"
            else:
                return "可接受方案：售罄概率尚可，但利润较薄"
        else:
            if profit_margin > 0.15:
                return "优秀方案：高售罄概率且利润可观"
            else:
                return "保守方案：售罄概率高但让利较多"
    def optimize_single_price(self, initial_stock: int,
                              promotion_hours: Tuple[int, int],
                              features: Dict) -> float:
        """优化单一价格（当不允许阶梯定价时）"""

        # 二分查找最优折扣
        low = self.min_discount
        high = 1.0
        best_discount = 1.0
        best_profit = -np.inf

        for _ in range(20):  # 二分查找20次
            mid = (low + high) / 2

            # 预测两个折扣的利润
            profit_mid = self._evaluate_discount_profit(mid, initial_stock, promotion_hours, features)
            profit_high = self._evaluate_discount_profit(mid + 0.01, initial_stock, promotion_hours, features)

            if profit_mid > best_profit:
                best_profit = profit_mid
                best_discount = mid

            # 决定搜索方向
            if profit_high > profit_mid:
                low = mid
            else:
                high = mid

        return best_discount