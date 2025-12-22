# core/emergency_controller.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

class EmergencyClearanceController:
    """紧急清仓控制器"""
    
    def __init__(self, config_manager):
        """
        初始化紧急控制器
        
        Args:
            config_manager: 配置管理器
        """
        self.config = config_manager
        self.clearance_config = config_manager.clearance_config
        self.logger = logging.getLogger(__name__)
        
        # 监控状态
        self.monitoring_status = {}
    
    def monitor_sales_progress(self, strategy_id: str,
                              current_time: datetime,
                              actual_sales: int,
                              remaining_stock: int,
                              promotion_end: str) -> Dict:
        """
        监控销售进度
        
        Returns:
            Dict: 监控结果和建议
        """
        
        # 获取策略信息
        if strategy_id not in self.monitoring_status:
            self.monitoring_status[strategy_id] = {
                'start_time': current_time,
                'total_sales': 0,
                'checkpoints': [],
                'adjustments_made': 0
            }
        
        status = self.monitoring_status[strategy_id]
        status['total_sales'] += actual_sales
        
        # 记录检查点
        checkpoint = {
            'time': current_time,
            'sales': actual_sales,
            'remaining_stock': remaining_stock,
            'time_to_close': self._calculate_time_to_close(current_time, promotion_end)
        }
        status['checkpoints'].append(checkpoint)
        
        # 分析进度
        progress_analysis = self._analyze_progress(status, current_time, promotion_end)
        
        # 检查是否需要紧急调整
        adjustment_needed, adjustment_type = self._check_emergency_adjustment(
            progress_analysis, remaining_stock, status
        )
        
        # 生成建议
        recommendations = self._generate_emergency_recommendations(
            progress_analysis, adjustment_needed, adjustment_type
        )
        
        return {
            'strategy_id': strategy_id,
            'current_time': current_time.isoformat(),
            'total_sales': status['total_sales'],
            'remaining_stock': remaining_stock,
            'time_to_close': checkpoint['time_to_close'],
            'progress_analysis': progress_analysis,
            'adjustment_needed': adjustment_needed,
            'adjustment_type': adjustment_type,
            'recommendations': recommendations,
            'checkpoint_count': len(status['checkpoints'])
        }
    
    def _calculate_time_to_close(self, current_time: datetime, 
                                promotion_end: str) -> float:
        """计算距离结束时间（小时）"""
        end_hour, end_minute = map(int, promotion_end.split(':'))
        end_time = current_time.replace(hour=end_hour, minute=end_minute, second=0)
        
        if end_time < current_time:
            end_time += timedelta(days=1)
        
        return (end_time - current_time).total_seconds() / 3600
    
    def _analyze_progress(self, status: Dict,
                         current_time: datetime,
                         promotion_end: str) -> Dict:
        """分析销售进度"""
        
        if not status['checkpoints']:
            return {'status': 'no_data', 'progress_rate': 0}
        
        # 计算已用时间
        elapsed_hours = (current_time - status['start_time']).total_seconds() / 3600
        total_sales = status['total_sales']
        
        # 计算剩余时间
        time_to_close = self._calculate_time_to_close(current_time, promotion_end)
        
        # 计算销售速率
        if elapsed_hours > 0:
            sales_rate = total_sales / elapsed_hours
        else:
            sales_rate = 0
        
        # 计算所需速率
        # 这里需要知道初始库存，从检查点中获取
        if status['checkpoints']:
            initial_stock = status['checkpoints'][0]['remaining_stock'] + total_sales
        else:
            initial_stock = 0
        
        if time_to_close > 0 and initial_stock > total_sales:
            required_rate = (initial_stock - total_sales) / time_to_close
        else:
            required_rate = 0
        
        # 计算进度比率
        if required_rate > 0:
            progress_ratio = sales_rate / required_rate
        else:
            progress_ratio = 1.0
        
        # 评估进度状态
        if progress_ratio >= 1.2:
            status_level = "ahead"
        elif progress_ratio >= 0.8:
            status_level = "on_track"
        elif progress_ratio >= 0.5:
            status_level = "behind"
        else:
            status_level = "critical"
        
        # 计算时间压力
        time_pressure = min(1.0, max(0, 1 - (time_to_close / elapsed_hours))) if elapsed_hours > 0 else 0
        
        return {
            'status': status_level,
            'progress_rate': round(progress_ratio, 2),
            'sales_rate': round(sales_rate, 1),
            'required_rate': round(required_rate, 1),
            'elapsed_hours': round(elapsed_hours, 2),
            'time_to_close': round(time_to_close, 2),
            'time_pressure': round(time_pressure, 2),
            'clearance_progress': total_sales / initial_stock if initial_stock > 0 else 0
        }
    
    def _check_emergency_adjustment(self, progress: Dict,
                                   remaining_stock: int,
                                   status: Dict) -> Tuple[bool, str]:
        """检查是否需要紧急调整"""
        
        # 检查调整次数限制
        max_adjustments = self.clearance_config.max_adjustments_per_session
        if status['adjustments_made'] >= max_adjustments:
            return False, "max_adjustments_reached"
        
        # 检查进度状态
        if progress['status'] == "critical":
            # 进度严重落后
            if progress['time_to_close'] < 1.0:  # 剩余时间少于1小时
                return True, "emergency_discount"
            else:
                return True, "increase_discount"
        
        elif progress['status'] == "behind":
            # 进度落后
            if progress['time_pressure'] > 0.7:  # 时间压力大
                return True, "moderate_discount_increase"
            else:
                return False, "monitoring"
        
        elif progress['status'] == "on_track":
            # 进度正常
            return False, "no_adjustment_needed"
        
        else:  # ahead
            # 进度超前
            return False, "consider_reducing_discount"
    
    def _generate_emergency_recommendations(self, progress: Dict,
                                          adjustment_needed: bool,
                                          adjustment_type: str) -> List[str]:
        """生成紧急建议"""
        
        recommendations = []
        
        if not adjustment_needed:
            if adjustment_type == "consider_reducing_discount":
                recommendations.append("销售进度超前，可以考虑减少折扣力度以提高利润")
            else:
                recommendations.append("销售进度正常，继续监控")
            return recommendations
        
        # 需要调整的情况
        if adjustment_type == "emergency_discount":
            recommendations.append("紧急情况：销售严重落后，剩余时间不足")
            recommendations.append(f"建议：立即执行紧急折扣（增加{self.clearance_config.emergency_discount_increment*100}%折扣力度）")
            recommendations.append("行动：1) 大幅降低价格 2) 加强现场宣传 3) 考虑捆绑销售")
        
        elif adjustment_type == "increase_discount":
            recommendations.append("警告：销售进度落后")
            recommendations.append(f"建议：增加折扣力度（增加{self.clearance_config.emergency_discount_increment*50}%折扣力度）")
            recommendations.append("行动：1) 调整价格 2) 增加促销标识 3) 员工重点推荐")
        
        elif adjustment_type == "moderate_discount_increase":
            recommendations.append("注意：销售进度稍慢")
            recommendations.append("建议：适度增加折扣力度")
            recommendations.append("行动：1) 小幅降价 2) 检查陈列位置 3) 增加试吃/试用")
        
        # 添加通用建议
        if progress['time_to_close'] < 0.5:
            recommendations.append("剩余时间不足30分钟，考虑最后清仓策略")
        
        if progress['clearance_progress'] < 0.3 and progress['time_to_close'] < 1.0:
            recommendations.append("库存清理困难，考虑员工内部购买或捐赠处理")
        
        return recommendations
    
    def calculate_emergency_discount(self, current_discount: float,
                                   adjustment_type: str) -> float:
        """计算紧急折扣"""
        
        if adjustment_type == "emergency_discount":
            increment = self.clearance_config.emergency_discount_increment * 2
        elif adjustment_type == "increase_discount":
            increment = self.clearance_config.emergency_discount_increment
        elif adjustment_type == "moderate_discount_increase":
            increment = self.clearance_config.emergency_discount_increment * 0.5
        else:
            increment = 0
        
        new_discount = max(
            self.clearance_config.max_emergency_discount,
            current_discount - increment
        )
        
        return new_discount
    
    def record_adjustment(self, strategy_id: str,
                         adjustment_type: str,
                         old_discount: float,
                         new_discount: float):
        """记录调整"""
        
        if strategy_id in self.monitoring_status:
            self.monitoring_status[strategy_id]['adjustments_made'] += 1
            
            # 记录调整历史
            if 'adjustment_history' not in self.monitoring_status[strategy_id]:
                self.monitoring_status[strategy_id]['adjustment_history'] = []
            
            adjustment_record = {
                'time': datetime.now(),
                'type': adjustment_type,
                'old_discount': old_discount,
                'new_discount': new_discount,
                'discount_change': old_discount - new_discount
            }
            
            self.monitoring_status[strategy_id]['adjustment_history'].append(adjustment_record)
            
            self.logger.info(f"策略 {strategy_id} 调整记录: {adjustment_type}, "
                           f"折扣 {old_discount:.2f} -> {new_discount:.2f}")
    
    def get_monitoring_summary(self, strategy_id: str) -> Dict:
        """获取监控摘要"""
        
        if strategy_id not in self.monitoring_status:
            return {'status': 'not_monitored'}
        
        status = self.monitoring_status[strategy_id]
        
        return {
            'status': 'monitored',
            'start_time': status['start_time'].isoformat() if 'start_time' in status else None,
            'total_sales': status.get('total_sales', 0),
            'checkpoints': len(status.get('checkpoints', [])),
            'adjustments_made': status.get('adjustments_made', 0),
            'adjustment_history': [
                {
                    'time': adj['time'].isoformat(),
                    'type': adj['type'],
                    'discount_change': adj['discount_change']
                }
                for adj in status.get('adjustment_history', [])
            ],
            'latest_checkpoint': status['checkpoints'][-1] if status.get('checkpoints') else None
        }