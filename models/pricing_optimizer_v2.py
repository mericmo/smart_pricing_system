# models/pricing_optimizer.py (完全重写)
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

@dataclass
class ClearanceSegment:
    """出清时段定价"""
    start_time: str
    end_time: str
    discount: float  # 折扣率，0.4表示4折
    price: float
    expected_sales: int
    urgency_level: float  # 紧迫程度 0-1
    clearance_priority: float  # 出清优先级 0-1
    sales_pressure: float  # 销售压力 0-1

class ClearanceFirstOptimizer:
    """出清优先定价优化器"""
    
    def __init__(self, demand_predictor, config_manager):
        """
        初始化优化器
        
        Args:
            demand_predictor: 需求预测器
            config_manager: 配置管理器
        """
        self.demand_predictor = demand_predictor
        self.config = config_manager
        self.clearance_config = config_manager.clearance_config
    
    def generate_clearance_strategy(self, 
                                  product_info: Dict,
                                  initial_stock: int,
                                  promotion_start: str,
                                  promotion_end: str,
                                  min_discount: float,
                                  max_discount: float,
                                  features: Dict) -> Dict:
        """
        生成出清优先定价策略
        
        Args:
            product_info: 商品信息
            initial_stock: 初始库存
            promotion_start: 促销开始时间
            promotion_end: 促销结束时间
            min_discount: 最低折扣
            max_discount: 最高折扣
            features: 特征字典
            
        Returns:
            Dict: 定价策略和评估结果
        """
        
        # 1. 评估出清可行性
        feasibility = self.assess_clearance_feasibility(
            product_info, initial_stock, promotion_start, 
            promotion_end, min_discount, features
        )
        
        # 2. 根据可行性选择策略类型
        if feasibility['clearance_probability'] >= self.clearance_config.min_clearance_probability:
            # 可出清：使用利润优化策略
            strategy = self._generate_profit_optimized_strategy(
                product_info, initial_stock, promotion_start, promotion_end,
                min_discount, max_discount, features
            )
            strategy_type = "profit_optimized"
        else:
            # 难以出清：使用紧急清仓策略
            strategy = self._generate_emergency_clearance_strategy(
                product_info, initial_stock, promotion_start, promotion_end,
                min_discount, max_discount, features, feasibility
            )
            strategy_type = "emergency_clearance"
        
        # 3. 评估策略
        evaluation = self.evaluate_strategy(
            strategy, initial_stock, product_info
        )
        
        # 4. 添加可行性分析和建议
        evaluation.update({
            'feasibility_analysis': feasibility,
            'strategy_type': strategy_type,
            'recommendations': self._generate_recommendations(feasibility, evaluation)
        })
        
        return {
            'strategy': strategy,
            'evaluation': evaluation,
            'feasibility': feasibility
        }
    
    def assess_clearance_feasibility(self,
                                   product_info: Dict,
                                   initial_stock: int,
                                   promotion_start: str,
                                   promotion_end: str,
                                   min_discount: float,
                                   features: Dict) -> Dict:
        """评估出清可行性"""
        
        # 解析时间
        start_hour, start_minute = map(int, promotion_start.split(':'))
        end_hour, end_minute = map(int, promotion_end.split(':'))
        
        # 计算促销时长
        start_minutes = start_hour * 60 + start_minute
        end_minutes = end_hour * 60 + end_minute
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60
        total_minutes = end_minutes - start_minutes
        
        # 获取历史销售数据
        hist_avg_sales = features.get('hist_avg_sales', 5)
        price_elasticity = features.get('price_elasticity', 1.2)
        promotion_sensitivity = features.get('promotion_sensitivity', 1.0)
        
        # 估算最大可能销量（使用最低折扣）
        # 计算时间衰减因子
        time_factor = self._calculate_time_pressure_factor(total_minutes)
        
        # 计算价格效应
        price_factor = (1.0 / min_discount) ** price_elasticity
        
        # 计算促销敏感效应
        promotion_factor = 1.0 + (1.0 - min_discount) * promotion_sensitivity
        
        # 计算库存压力效应
        stock_pressure = self._calculate_stock_pressure(initial_stock)
        
        # 估算总销量
        base_demand = hist_avg_sales * (total_minutes / 60)  # 转换为小时需求
        max_possible_sales = base_demand * price_factor * promotion_factor * time_factor * stock_pressure
        
        # 计算售罄概率
        clearance_probability = min(max_possible_sales / initial_stock, 1.0) if initial_stock > 0 else 1.0
        
        # 评估可行性等级
        if clearance_probability >= 0.9:
            feasibility_level = "high"
        elif clearance_probability >= 0.7:
            feasibility_level = "medium"
        elif clearance_probability >= 0.5:
            feasibility_level = "low"
        else:
            feasibility_level = "very_low"
        
        # 计算需要的销售速率
        required_rate = initial_stock / (total_minutes / 60)  # 每小时需要销售的数量
        
        # 计算实际可能的销售速率（使用历史平均值和折扣效应）
        possible_rate = hist_avg_sales * price_factor * promotion_factor
        
        return {
            'initial_stock': initial_stock,
            'promotion_duration_hours': total_minutes / 60,
            'max_possible_sales': int(max_possible_sales),
            'clearance_probability': round(clearance_probability, 3),
            'feasibility_level': feasibility_level,
            'required_sales_rate': round(required_rate, 1),
            'possible_sales_rate': round(possible_rate, 1),
            'stock_pressure': round(stock_pressure, 2),
            'time_pressure': round(time_factor, 2),
            'price_effect': round(price_factor, 2)
        }
    
    def _calculate_time_pressure_factor(self, total_minutes: float) -> float:
        """计算时间压力因子"""
        # 时间越短，压力越大
        if total_minutes <= 60:  # 1小时以内
            return 2.0
        elif total_minutes <= 120:  # 2小时以内
            return 1.5
        elif total_minutes <= 180:  # 3小时以内
            return 1.2
        else:
            return 1.0
    
    def _calculate_stock_pressure(self, stock: int) -> float:
        """计算库存压力因子"""
        # 库存越多，压力越大
        if stock >= self.clearance_config.high_stock_threshold:
            return 1.8
        elif stock >= self.clearance_config.medium_stock_threshold:
            return 1.4
        elif stock >= 20:
            return 1.2
        else:
            return 1.0
    
    def _generate_profit_optimized_strategy(self,
                                          product_info: Dict,
                                          initial_stock: int,
                                          promotion_start: str,
                                          promotion_end: str,
                                          min_discount: float,
                                          max_discount: float,
                                          features: Dict) -> List[ClearanceSegment]:
        """生成利润优化策略（在保证出清的前提下）"""
        
        # 使用动态规划求解
        strategy = self._dynamic_programming_optimization(
            product_info=product_info,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount,
            features=features,
            objective="clearance_first_profit"  # 出清优先的利润优化
        )
        
        return strategy
    
    def _generate_emergency_clearance_strategy(self,
                                             product_info: Dict,
                                             initial_stock: int,
                                             promotion_start: str,
                                             promotion_end: str,
                                             min_discount: float,
                                             max_discount: float,
                                             features: Dict,
                                             feasibility: Dict) -> List[ClearanceSegment]:
        """生成紧急清仓策略（当难以出清时）"""
        
        # 解析时间
        start_hour, start_minute = map(int, promotion_start.split(':'))
        end_hour, end_minute = map(int, promotion_end.split(':'))
        
        # 计算总时长和时段
        start_minutes = start_hour * 60 + start_minute
        end_minutes = end_hour * 60 + end_minute
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60
        total_minutes = end_minutes - start_minutes
        
        # 根据可行性调整折扣
        clearance_prob = feasibility['clearance_probability']
        if clearance_prob < 0.3:
            # 非常难以出清，使用更大折扣
            effective_min_discount = max(
                min_discount - self.clearance_config.emergency_discount_increment * 2,
                self.clearance_config.max_emergency_discount
            )
        elif clearance_prob < 0.5:
            # 难以出清，适当增加折扣
            effective_min_discount = max(
                min_discount - self.clearance_config.emergency_discount_increment,
                self.clearance_config.max_emergency_discount
            )
        else:
            effective_min_discount = min_discount
        
        # 生成紧急策略：更激进的折扣，更少的时间段
        strategy = []
        remaining_stock = initial_stock
        current_time_minutes = start_minutes
        
        # 划分时间段（更少的时间段，更激进的折扣变化）
        num_segments = min(3, max(2, int(total_minutes / 30)))  # 每30分钟至少一个时段
        
        for segment_idx in range(num_segments):
            # 计算时间
            segment_duration = total_minutes / num_segments
            segment_start_minutes = current_time_minutes
            segment_end_minutes = current_time_minutes + segment_duration
            
            segment_start_hour = int(segment_start_minutes // 60) % 24
            segment_start_minute = int(segment_start_minutes % 60)
            segment_end_hour = int(segment_end_minutes // 60) % 24
            segment_end_minute = int(segment_end_minutes % 60)
            
            # 计算折扣（随时间越来越低）
            time_progress = segment_idx / num_segments
            segment_discount = max_discount - (max_discount - effective_min_discount) * time_progress
            
            # 调整折扣以确保出清
            if segment_idx == num_segments - 1 and remaining_stock > 0:
                # 最后一个时段，如果还有库存，使用最低折扣
                segment_discount = effective_min_discount
            
            # 预测销量
            time_remaining = 1 - time_progress
            predicted_sales = self.demand_predictor.predict_demand(
                features=features,
                discount_rate=segment_discount,
                time_to_close=time_remaining,
                current_stock=remaining_stock,
                base_demand=features.get('hist_avg_sales', 5)
            )
            
            # 确保预测销量不超过库存
            actual_sales = min(int(predicted_sales), remaining_stock)
            
            # 计算紧迫程度和优先级
            urgency = 1 - time_progress
            clearance_priority = min(remaining_stock / initial_stock, 1.0) if initial_stock > 0 else 0
            sales_pressure = (initial_stock - remaining_stock) / initial_stock if initial_stock > 0 else 0
            
            # 创建时段
            segment = ClearanceSegment(
                start_time=f"{segment_start_hour:02d}:{segment_start_minute:02d}",
                end_time=f"{segment_end_hour:02d}:{segment_end_minute:02d}",
                discount=segment_discount,
                price=product_info['original_price'] * segment_discount,
                expected_sales=actual_sales,
                urgency_level=urgency,
                clearance_priority=clearance_priority,
                sales_pressure=sales_pressure
            )
            
            strategy.append(segment)
            remaining_stock -= actual_sales
            current_time_minutes = segment_end_minutes
            
            if remaining_stock <= 0:
                break
        
        return strategy
    
    def _dynamic_programming_optimization(self,
                                        product_info: Dict,
                                        initial_stock: int,
                                        promotion_start: str,
                                        promotion_end: str,
                                        min_discount: float,
                                        max_discount: float,
                                        features: Dict,
                                        objective: str = "clearance_first_profit") -> List[ClearanceSegment]:
        """动态规划优化（出清优先）"""
        
        # 解析时间
        start_hour, start_minute = map(int, promotion_start.split(':'))
        end_hour, end_minute = map(int, promotion_end.split(':'))
        
        # 计算总时长和时段
        start_minutes = start_hour * 60 + start_minute
        end_minutes = end_hour * 60 + end_minute
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60
        total_minutes = end_minutes - start_minutes
        
        # 设置时段数量（基于最小调价间隔）
        min_interval = self.clearance_config.min_time_between_changes
        max_segments = min(self.clearance_config.max_discount_changes, 
                          int(total_minutes / min_interval))
        num_segments = max(2, min(4, max_segments))
        
        # 离散化折扣空间
        discount_levels = np.linspace(min_discount, max_discount, num=10)
        
        # 初始化DP表
        dp = np.full((num_segments + 1, initial_stock + 1), -np.inf)
        dp[0, initial_stock] = 0  # 初始状态
        
        # 决策记录表
        decisions = np.full((num_segments, initial_stock + 1), -1, dtype=int)
        
        # 成本价和原价
        cost_price = product_info['cost_price']
        original_price = product_info['original_price']
        
        # 动态规划
        for t in range(num_segments):
            for s in range(initial_stock + 1):  # 剩余库存
                if dp[t, s] == -np.inf:
                    continue
                
                # 当前时段剩余时间比例
                time_remaining = 1.0 - (t * total_minutes / num_segments) / total_minutes
                
                for i, discount in enumerate(discount_levels):
                    # 预测销量
                    predicted_sales = self.demand_predictor.predict_demand(
                        features=features,
                        discount_rate=discount,
                        time_to_close=time_remaining,
                        current_stock=s,
                        base_demand=features.get('hist_avg_sales', 5)
                    )
                    
                    # 实际销售量
                    actual_sales = min(int(predicted_sales), s)
                    
                    # 计算利润
                    price = original_price * discount
                    profit = (price - cost_price) * actual_sales
                    
                    # 根据目标函数计算值
                    if objective == "clearance_first_profit":
                        # 出清优先的利润：利润 + 出清奖励 - 库存惩罚
                        clearance_bonus = 0
                        stock_penalty = 0
                        
                        new_stock = s - actual_sales
                        
                        if new_stock == 0:
                            # 完全出清奖励
                            clearance_bonus = profit * 0.5  # 额外50%利润作为奖励
                        elif t == num_segments - 1 and new_stock > 0:
                            # 最后时段还有库存，惩罚
                            stock_penalty = new_stock * cost_price * 0.7  # 损失成本的70%
                        
                        value = profit + clearance_bonus - stock_penalty
                    else:
                        value = profit
                    
                    # 更新状态
                    new_stock = s - actual_sales
                    new_value = dp[t, s] + value
                    
                    if new_value > dp[t + 1, new_stock]:
                        dp[t + 1, new_stock] = new_value
                        decisions[t, s] = i
        
        # 回溯找到最优解
        strategy = self._backtrack_strategy(
            decisions, discount_levels, dp, product_info,
            initial_stock, start_minutes, total_minutes, num_segments, features
        )
        
        return strategy
    
    def _backtrack_strategy(self, decisions, discount_levels, dp, product_info,
                          initial_stock, start_minutes, total_minutes, 
                          num_segments, features) -> List[ClearanceSegment]:
        """回溯构建策略"""
        
        # 找到最终状态（优先选择库存为0的状态）
        final_segment = num_segments
        
        # 优先选择库存为0的状态
        final_stock = 0
        if dp[final_segment, 0] > -np.inf:
            final_stock = 0
        else:
            # 如果没有完全出清的状态，选择库存最少的状态
            for s in range(initial_stock + 1):
                if dp[final_segment, s] > -np.inf:
                    final_stock = s
                    break
        
        strategy = []
        current_stock = initial_stock
        
        for t in range(num_segments):
            if decisions[t, current_stock] == -1:
                break
            
            discount_idx = decisions[t, current_stock]
            discount = discount_levels[discount_idx]
            
            # 计算时间
            segment_start_minutes = start_minutes + t * (total_minutes / num_segments)
            segment_end_minutes = segment_start_minutes + (total_minutes / num_segments)
            
            segment_start_hour = int(segment_start_minutes // 60) % 24
            segment_start_minute = int(segment_start_minutes % 60)
            segment_end_hour = int(segment_end_minutes // 60) % 24
            segment_end_minute = int(segment_end_minutes % 60)
            
            # 预测销量
            time_remaining = 1.0 - ((t + 1) * total_minutes / num_segments) / total_minutes
            predicted_sales = self.demand_predictor.predict_demand(
                features=features,
                discount_rate=discount,
                time_to_close=time_remaining,
                current_stock=current_stock,
                base_demand=features.get('hist_avg_sales', 5)
            )
            
            actual_sales = min(int(predicted_sales), current_stock)
            
            # 计算紧迫程度
            time_progress = t / num_segments
            urgency = 1 - time_progress
            clearance_priority = min(current_stock / initial_stock, 1.0) if initial_stock > 0 else 0
            sales_pressure = (initial_stock - current_stock) / initial_stock if initial_stock > 0 else 0
            
            # 创建时段
            segment = ClearanceSegment(
                start_time=f"{segment_start_hour:02d}:{segment_start_minute:02d}",
                end_time=f"{segment_end_hour:02d}:{segment_end_minute:02d}",
                discount=discount,
                price=product_info['original_price'] * discount,
                expected_sales=actual_sales,
                urgency_level=urgency,
                clearance_priority=clearance_priority,
                sales_pressure=sales_pressure
            )
            
            strategy.append(segment)
            current_stock -= actual_sales
            
            if current_stock <= 0:
                break
        
        return strategy
    
    def evaluate_strategy(self, strategy: List[ClearanceSegment],
                         initial_stock: int,
                         product_info: Dict) -> Dict:
        """评估策略"""
        
        if not strategy:
            return {
                'success': False,
                'message': '未生成有效策略'
            }
        
        # 计算总指标
        total_expected_sales = sum(segment.expected_sales for segment in strategy)
        total_revenue = sum(segment.price * segment.expected_sales for segment in strategy)
        total_profit = sum((segment.price - product_info['cost_price']) * segment.expected_sales 
                          for segment in strategy)
        remaining_stock = max(0, initial_stock - total_expected_sales)
        
        # 计算售罄率
        clearance_rate = total_expected_sales / initial_stock if initial_stock > 0 else 1.0
        
        # 计算利润率
        profit_margin = total_profit / total_revenue if total_revenue > 0 else 0
        
        # 计算平均折扣
        avg_discount = np.mean([segment.discount for segment in strategy])
        
        # 评估成功与否
        success = clearance_rate >= self.clearance_config.clearance_threshold
        
        # 计算紧迫程度变化
        urgency_start = strategy[0].urgency_level if strategy else 0
        urgency_end = strategy[-1].urgency_level if strategy else 0
        
        # 计算折扣变化
        discount_start = strategy[0].discount if strategy else 1.0
        discount_end = strategy[-1].discount if strategy else 1.0
        
        return {
            'success': success,
            'clearance_rate': round(clearance_rate, 3),
            'total_expected_sales': int(total_expected_sales),
            'total_revenue': round(total_revenue, 2),
            'total_profit': round(total_profit, 2),
            'remaining_stock': remaining_stock,
            'profit_margin': round(profit_margin, 3),
            'average_discount': round(avg_discount, 3),
            'urgency_change': round(urgency_start - urgency_end, 3),
            'discount_change': round(discount_start - discount_end, 3),
            'num_segments': len(strategy),
            'expected_clearance_time': self._estimate_clearance_time(strategy)
        }
    
    def _estimate_clearance_time(self, strategy: List[ClearanceSegment]) -> str:
        """估计售罄时间"""
        if not strategy:
            return "未知"
        
        # 假设每个时段销售预期销量的一半时售罄
        cumulative_sales = 0
        for i, segment in enumerate(strategy):
            cumulative_sales += segment.expected_sales
            
            # 如果这是最后一个时段或累计销售超过预期
            if i == len(strategy) - 1 or cumulative_sales >= sum(s.expected_sales for s in strategy) * 0.5:
                # 返回该时段中间时间
                start_time = datetime.strptime(segment.start_time, "%H:%M")
                end_time = datetime.strptime(segment.end_time, "%H:%M")
                
                # 计算中间时间
                mid_time = start_time + (end_time - start_time) / 2
                return mid_time.strftime("%H:%M")
        
        return strategy[-1].end_time
    
    def _generate_recommendations(self, feasibility: Dict, 
                                evaluation: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        clearance_prob = feasibility['clearance_probability']
        clearance_rate = evaluation['clearance_rate']
        
        if clearance_prob < 0.5:
            recommendations.append("⚠️ 库存过高或时间窗口过短，难以完全出清")
            recommendations.append("建议：1) 提前开始促销 2) 考虑捆绑销售 3) 联系内部员工购买")
        
        if clearance_prob >= 0.5 and clearance_prob < 0.8:
            recommendations.append("📊 出清概率中等，需要谨慎定价")
            recommendations.append("建议：1) 使用更激进的阶梯折扣 2) 加强促销宣传 3) 考虑搭配销售")
        
        if clearance_rate < 0.7:
            recommendations.append("🎯 当前策略售罄率偏低")
            recommendations.append("建议：1) 加大折扣力度 2) 延长促销时间 3) 增加销售渠道")
        
        if evaluation['profit_margin'] < 0.1:
            recommendations.append("💰 利润率偏低，考虑成本控制")
            recommendations.append("建议：1) 优化采购成本 2) 减少浪费 3) 提高运营效率")
        
        if not recommendations:
            recommendations.append("✅ 策略合理，按计划执行")
        
        return recommendations