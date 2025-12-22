# discount_optimizer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import math
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

class DiscountOptimizer:
    """
    智能打折促销方案推荐算法
    使用动态规划生成阶梯式折扣方案，确保日清品在指定时间内售罄
    """
    
    def __init__(self, model_predictor, feature_store, data_preprocessor):
        """
        初始化折扣优化器
        
        参数:
        - model_predictor: 销量预测模型
        - feature_store: 特征仓库
        - data_preprocessor: 数据预处理器
        """
        self.model_predictor = model_predictor
        self.feature_store = feature_store
        self.data_preprocessor = data_preprocessor
        
        # 默认参数
        self.default_params = {
            'min_gross_margin': 0.1,  # 最低毛利率10%
            'allow_staggered_pricing': True,  # 允许阶梯定价
            'time_slots': 4,  # 默认分4个时段
            'customer_perception_weight': 0.3,  # 顾客感知权重
            'max_discount_change': 0.2,  # 最大折扣变化幅度
            'safety_stock_factor': 0.1,  # 安全库存系数
        }
    
    def generate_discount_plan(self, input_params: Dict) -> Dict:
        """
        生成折扣方案
        
        参数:
        - input_params: 输入参数字典，包含:
            - product_code: 商品编码
            - current_inventory: 当前库存
            - promotion_window: 促销时间窗口 (如: ['20:00', '22:00'])
            - min_gross_margin: 最低毛利率
            - allow_staggered_pricing: 是否允许阶梯定价
            - historical_sales_data: 历史销售数据 (可选)
            - current_date: 当前日期
            - base_price: 基准价格 (可选)
            - cost_price: 成本价 (可选)
        
        返回:
        - 折扣方案字典
        """
        try:
            # 验证输入参数
            self._validate_input_params(input_params)
            
            # 获取必要参数
            product_code = input_params['product_code']
            current_inventory = input_params['current_inventory']
            promotion_window = input_params.get('promotion_window', ['20:00', '22:00'])
            min_gross_margin = input_params.get('min_gross_margin', self.default_params['min_gross_margin'])
            allow_staggered = input_params.get('allow_staggered_pricing', self.default_params['allow_staggered_pricing'])
            
            # 获取当前日期
            current_date = input_params.get('current_date', datetime.now())
            
            # 获取商品信息
            product_info = self._get_product_info(product_code)
            if not product_info:
                raise ValueError(f"无法获取商品 {product_code} 的信息")
            
            # 获取历史销售数据
            historical_data = input_params.get('historical_sales_data')
            if historical_data is None:
                historical_data = self._get_historical_sales(product_code)
            
            # 获取基准价格和成本价
            base_price = input_params.get('base_price', product_info.get('price', 0))
            cost_price = input_params.get('cost_price', base_price * 0.7)  # 默认成本为售价的70%
            
            # 计算历史销售速率
            sales_rate = self._calculate_sales_rate(historical_data, promotion_window)
            
            # 生成预测特征
            prediction_features = self._generate_prediction_features(
                product_code, current_date, historical_data
            )
            
            # 生成折扣方案
            if allow_staggered:
                discount_plan = self._generate_staggered_discount_plan(
                    current_inventory=current_inventory,
                    sales_rate=sales_rate,
                    promotion_window=promotion_window,
                    base_price=base_price,
                    cost_price=cost_price,
                    min_gross_margin=min_gross_margin,
                    prediction_features=prediction_features
                )
            else:
                discount_plan = self._generate_single_discount_plan(
                    current_inventory=current_inventory,
                    sales_rate=sales_rate,
                    promotion_window=promotion_window,
                    base_price=base_price,
                    cost_price=cost_price,
                    min_gross_margin=min_gross_margin,
                    prediction_features=prediction_features
                )
            
            # 计算方案指标
            plan_metrics = self._calculate_plan_metrics(
                discount_plan, current_inventory, base_price, cost_price
            )
            
            # 方案可行性分析
            feasibility = self._analyze_feasibility(
                discount_plan, plan_metrics, min_gross_margin
            )
            
            # 返回完整方案
            return {
                'product_code': product_code,
                'current_inventory': current_inventory,
                'promotion_window': promotion_window,
                'min_gross_margin': min_gross_margin,
                'base_price': base_price,
                'cost_price': cost_price,
                'discount_plan': discount_plan,
                'plan_metrics': plan_metrics,
                'feasibility_analysis': feasibility,
                'recommendation': self._generate_recommendation(feasibility)
            }
            
        except Exception as e:
            raise Exception(f"生成折扣方案失败: {str(e)}")
    
    def _validate_input_params(self, params: Dict):
        """验证输入参数"""
        required_params = ['product_code', 'current_inventory']
        for param in required_params:
            if param not in params:
                raise ValueError(f"缺少必要参数: {param}")
        
        if params['current_inventory'] <= 0:
            raise ValueError("当前库存必须大于0")
    
    def _get_product_info(self, product_code: str) -> Optional[Dict]:
        """获取商品信息"""
        # 在实际应用中，这里应该从数据库或数据文件中获取商品信息
        # 这里返回模拟数据
        return {
            'product_code': product_code,
            'price': 10.0,  # 默认价格
            'category': '日清品',
            'shelf_life': 1  # 保质期天数
        }
    
    def _get_historical_sales(self, product_code: str) -> pd.DataFrame:
        """获取历史销售数据"""
        # 在实际应用中，这里应该从数据库或数据文件中获取历史数据
        # 这里返回空DataFrame，需要外部提供
        return pd.DataFrame()
    
    def _calculate_sales_rate(self, historical_data: pd.DataFrame, 
                            promotion_window: List[str]) -> Dict:
        """计算历史销售速率"""
        if historical_data.empty:
            # 如果没有历史数据，使用默认速率
            return {
                'avg_rate': 5,  # 每小时默认销售5件
                'peak_rate': 10,
                'low_rate': 2,
                'confidence': 0.5  # 置信度
            }
        
        try:
            # 提取促销时间段的数据
            start_hour = int(promotion_window[0].split(':')[0])
            end_hour = int(promotion_window[1].split(':')[0])
            
            # 假设historical_data中有'time_hour'字段
            if 'time_hour' in historical_data.columns:
                promo_data = historical_data[
                    (historical_data['time_hour'] >= start_hour) & 
                    (historical_data['time_hour'] < end_hour)
                ]
            else:
                promo_data = historical_data
            
            # 计算销售速率
            sales_rate = {
                'avg_rate': promo_data['销售数量'].mean() if len(promo_data) > 0 else 5,
                'peak_rate': promo_data['销售数量'].max() if len(promo_data) > 0 else 10,
                'low_rate': promo_data['销售数量'].min() if len(promo_data) > 0 else 2,
                'confidence': min(0.9, len(promo_data) / 100)  # 基于数据量的置信度
            }
            
            return sales_rate
            
        except Exception:
            return {
                'avg_rate': 5,
                'peak_rate': 10,
                'low_rate': 2,
                'confidence': 0.5
            }

    def _generate_prediction_features(self, product_code: str,
                                      current_date,
                                      historical_data: pd.DataFrame) -> pd.DataFrame:
        """生成预测特征"""
        # 确保日期是datetime类型
        if isinstance(current_date, str):
            try:
                current_date = pd.to_datetime(current_date)
            except:
                current_date = datetime.now()
        elif isinstance(current_date, (date, datetime)):
            current_date = pd.to_datetime(current_date)
        else:
            current_date = pd.to_datetime(datetime.now())

        # 创建基础特征DataFrame
        features = pd.DataFrame({
            '商品编码': [product_code],
            '日期': [current_date],
            '售价': [10.0],  # 基准价格
            '促销天数': [1],
            '是否周末': [1 if current_date.weekday() >= 5 else 0],
            '是否月末': [1 if current_date.day >= 25 else 0],
            '月份': [current_date.month],
            '星期': [current_date.weekday()],
            '季节': [self._get_season(current_date.month)]
        })

        # 添加滞后特征（需要历史数据）
        if not historical_data.empty:
            features = self._add_lag_features(features, historical_data)

        return features
    
    def _get_season(self, month: int) -> str:
        """获取季节"""
        if month in [12, 1, 2]:
            return '冬季'
        elif month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        else:
            return '秋季'
    
    def _add_lag_features(self, features: pd.DataFrame, 
                         historical_data: pd.DataFrame) -> pd.DataFrame:
        """添加滞后特征"""
        # 简化的滞后特征计算
        # 在实际应用中，应该使用feature_store中的方法
        if len(historical_data) >= 7:
            features['销量_滞后7天'] = historical_data['销售数量'].tail(7).mean()
        if len(historical_data) >= 30:
            features['销量_滞后30天'] = historical_data['销售数量'].tail(30).mean()
        
        return features

    def _generate_staggered_discount_plan(self, **kwargs) -> List[Dict]:
        """
        生成阶梯式折扣方案
        使用动态规划算法
        """
        current_inventory = kwargs['current_inventory']
        sales_rate = kwargs['sales_rate']
        promotion_window = kwargs['promotion_window']
        base_price = kwargs['base_price']
        cost_price = kwargs['cost_price']
        min_gross_margin = kwargs['min_gross_margin']

        # 解析时间窗口
        start_time = datetime.strptime(promotion_window[0], '%H:%M')
        end_time = datetime.strptime(promotion_window[1], '%H:%M')
        total_hours = (end_time - start_time).seconds / 3600

        # 划分时间段
        time_slots = self.default_params['time_slots']
        slot_duration = total_hours / time_slots * 60  # 分钟

        # 定义折扣选项
        discount_options = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # 10折到5折

        # 动态规划表
        dp = np.zeros((time_slots + 1, current_inventory + 1))
        decision = np.zeros((time_slots + 1, current_inventory + 1), dtype=int)

        # 初始化
        for i in range(current_inventory + 1):
            dp[time_slots][i] = 0  # 结束时的利润

        # 动态规划
        for t in range(time_slots - 1, -1, -1):
            for inv in range(current_inventory + 1):
                max_profit = -float('inf')
                best_discount = 0

                for d_idx, discount in enumerate(discount_options):
                    # 计算折扣后的价格
                    final_price = base_price * discount

                    # 检查毛利率
                    gross_margin = (final_price - cost_price) / final_price
                    if gross_margin < min_gross_margin:
                        continue  # 跳过不符合毛利率要求的折扣

                    # 预测销量
                    expected_sales = self._predict_sales_with_discount(
                        discount, sales_rate, slot_duration, inv
                    )

                    # 不能超过剩余库存
                    actual_sales = min(expected_sales, inv)

                    # 计算利润
                    profit = actual_sales * (final_price - cost_price)

                    # 顾客感知惩罚（频繁调价）
                    if t > 0:
                        prev_discount = discount_options[decision[t+1][inv-actual_sales]] if inv-actual_sales >= 0 else 1.0
                        price_change_penalty = abs(discount - prev_discount) * self.default_params['customer_perception_weight']
                        profit -= price_change_penalty * actual_sales * base_price

                    # 更新最优值
                    if inv - actual_sales >= 0:
                        total_profit = profit + dp[t+1][inv-actual_sales]
                        if total_profit > max_profit:
                            max_profit = total_profit
                            best_discount = d_idx

                dp[t][inv] = max_profit if max_profit > -float('inf') else 0
                decision[t][inv] = best_discount

        # 回溯得到最优方案
        discount_plan = []
        remaining_inventory = current_inventory

        for t in range(time_slots):
            best_discount_idx = decision[t][remaining_inventory]
            discount = discount_options[best_discount_idx]

            # 计算该时段的销售预测
            final_price = base_price * discount
            expected_sales = self._predict_sales_with_discount(
                discount, sales_rate, slot_duration, remaining_inventory
            )
            actual_sales = min(expected_sales, remaining_inventory)

            # 计算时间段
            start_minutes = start_time.hour * 60 + start_time.minute + t * slot_duration
            end_minutes = start_minutes + slot_duration

            start_str = f"{start_minutes//60:02d}:{start_minutes%60:02d}"
            end_str = f"{end_minutes//60:02d}:{end_minutes%60:02d}"

            discount_plan.append({
                'time_slot': f"{start_str}-{end_str}",
                'discount_percentage': round((1 - discount) * 100, 1),
                'final_price': round(final_price, 2),
                'expected_sales': int(actual_sales),
                'expected_revenue': round(actual_sales * final_price, 2),
                'expected_profit': round(actual_sales * (final_price - cost_price), 2)
            })

            remaining_inventory -= actual_sales

        return discount_plan

    def _generate_single_discount_plan(self, **kwargs) -> List[Dict]:
        """
        生成单一折扣方案
        """
        current_inventory = kwargs['current_inventory']
        sales_rate = kwargs['sales_rate']
        promotion_window = kwargs['promotion_window']
        base_price = kwargs['base_price']
        cost_price = kwargs['cost_price']
        min_gross_margin = kwargs['min_gross_margin']

        # 解析时间窗口
        try:
            start_time = datetime.strptime(promotion_window[0], '%H:%M')
            end_time = datetime.strptime(promotion_window[1], '%H:%M')
        except ValueError:
            start_time = datetime.strptime('20:00', '%H:%M')
            end_time = datetime.strptime('22:00', '%H:%M')

        total_minutes = (end_time - start_time).seconds / 60

        # 寻找最优单一折扣
        discount_options = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        best_discount = 1.0
        max_profit = 0

        for discount in discount_options:
            final_price = base_price * discount

            # 检查毛利率
            if final_price <= 0:
                continue

            gross_margin = (final_price - cost_price) / final_price
            if gross_margin < min_gross_margin:
                continue

            # 预测总销量
            expected_sales = self._predict_sales_with_discount(
                discount, sales_rate, total_minutes, current_inventory
            )

            # 转换为整数
            actual_sales_int = int(np.floor(expected_sales))
            actual_sales_int = min(actual_sales_int, current_inventory)

            # 计算利润
            profit = actual_sales_int * (final_price - cost_price)

            if profit > max_profit:
                max_profit = profit
                best_discount = discount

        # 创建单一折扣方案
        final_price = base_price * best_discount
        expected_sales = self._predict_sales_with_discount(
            best_discount, sales_rate, total_minutes, current_inventory
        )

        # 转换为整数
        actual_sales_int = int(np.floor(expected_sales))
        actual_sales_int = min(actual_sales_int, current_inventory)

        return [{
            'time_slot': f"{promotion_window[0]}-{promotion_window[1]}",
            'discount_percentage': round((1 - best_discount) * 100, 1),
            'final_price': round(final_price, 2),
            'expected_sales': actual_sales_int,  # 使用整数
            'expected_revenue': round(actual_sales_int * final_price, 2),
            'expected_profit': round(actual_sales_int * (final_price - cost_price), 2)
        }]

    def _predict_sales_with_discount(self, discount: float,
                                     sales_rate: Dict,
                                     duration_minutes: float,
                                     available_inventory: int) -> float:
        """
        预测给定折扣下的销量

        使用指数需求函数: Q = Q0 * (P/P0)^(-E)
        其中 E 是价格弹性系数
        """
        # 基准销量（无折扣）
        base_sales_rate = sales_rate['avg_rate']  # 每小时销量

        # 转换为分钟销量
        base_minute_rate = base_sales_rate / 60

        # 价格弹性系数（经验值，不同品类不同）
        # 日清品通常有较高弹性（-1.5到-2.5）
        price_elasticity = -2.0

        # 计算折扣对销量的影响
        if discount == 1.0:
            sales_multiplier = 1.0
        else:
            sales_multiplier = discount ** price_elasticity

        # 预测销量
        predicted_sales = base_minute_rate * duration_minutes * sales_multiplier

        # 考虑库存限制
        predicted_sales = min(predicted_sales, available_inventory)

        # 添加随机波动
        confidence = sales_rate.get('confidence', 0.5)
        noise = np.random.normal(0, 0.1 * (1 - confidence))
        predicted_sales *= (1 + noise)

        # 确保返回非负值
        return max(0, predicted_sales)

    def _generate_staggered_discount_plan(self, **kwargs) -> List[Dict]:
        """
        生成阶梯式折扣方案
        使用动态规划算法
        """
        current_inventory = kwargs['current_inventory']
        sales_rate = kwargs['sales_rate']
        promotion_window = kwargs['promotion_window']
        base_price = kwargs['base_price']
        cost_price = kwargs['cost_price']
        min_gross_margin = kwargs['min_gross_margin']

        # 解析时间窗口
        try:
            start_time = datetime.strptime(promotion_window[0], '%H:%M')
            end_time = datetime.strptime(promotion_window[1], '%H:%M')
        except ValueError:
            # 如果时间格式有问题，使用默认值
            start_time = datetime.strptime('20:00', '%H:%M')
            end_time = datetime.strptime('22:00', '%H:%M')

        total_hours = (end_time - start_time).seconds / 3600

        # 划分时间段
        time_slots = self.default_params['time_slots']
        slot_duration = total_hours / time_slots * 60  # 分钟

        # 定义折扣选项
        discount_options = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # 10折到5折

        # 动态规划表 - 使用float类型
        dp = np.zeros((time_slots + 1, current_inventory + 1), dtype=float)
        decision = np.zeros((time_slots + 1, current_inventory + 1), dtype=int)

        # 初始化
        for i in range(current_inventory + 1):
            dp[time_slots][i] = 0.0  # 结束时的利润

        # 动态规划
        for t in range(time_slots - 1, -1, -1):
            for inv in range(current_inventory + 1):
                max_profit = -float('inf')
                best_discount = 0

                for d_idx, discount in enumerate(discount_options):
                    # 计算折扣后的价格
                    final_price = base_price * discount

                    # 检查毛利率
                    if final_price <= 0:
                        continue

                    gross_margin = (final_price - cost_price) / final_price
                    if gross_margin < min_gross_margin:
                        continue  # 跳过不符合毛利率要求的折扣

                    # 预测销量（返回浮点数）
                    expected_sales = self._predict_sales_with_discount(
                        discount, sales_rate, slot_duration, inv
                    )

                    # 将预测销量转换为整数（实际销售必须是整数）
                    actual_sales_int = int(np.floor(expected_sales))

                    # 不能超过剩余库存
                    actual_sales_int = min(actual_sales_int, inv)

                    # 计算利润
                    profit = actual_sales_int * (final_price - cost_price)

                    # 顾客感知惩罚（频繁调价）
                    if t > 0 and inv - actual_sales_int >= 0:
                        prev_discount = discount_options[decision[t + 1][inv - actual_sales_int]]
                        price_change_penalty = abs(discount - prev_discount) * self.default_params[
                            'customer_perception_weight']
                        profit -= price_change_penalty * actual_sales_int * base_price

                    # 更新最优值
                    if inv - actual_sales_int >= 0:
                        total_profit = profit + dp[t + 1][inv - actual_sales_int]
                        if total_profit > max_profit:
                            max_profit = total_profit
                            best_discount = d_idx

                dp[t][inv] = max_profit if max_profit > -float('inf') else 0.0
                decision[t][inv] = best_discount

        # 回溯得到最优方案
        discount_plan = []
        remaining_inventory = current_inventory

        for t in range(time_slots):
            best_discount_idx = decision[t][remaining_inventory]
            discount = discount_options[best_discount_idx]

            # 计算该时段的销售预测
            final_price = base_price * discount
            expected_sales = self._predict_sales_with_discount(
                discount, sales_rate, slot_duration, remaining_inventory
            )

            # 将预测销量转换为整数
            actual_sales_int = int(np.floor(expected_sales))
            actual_sales_int = min(actual_sales_int, remaining_inventory)

            # 计算时间段
            start_minutes = start_time.hour * 60 + start_time.minute + t * slot_duration
            end_minutes = start_minutes + slot_duration

            start_str = f"{int(start_minutes // 60):02d}:{int(start_minutes % 60):02d}"
            end_str = f"{int(end_minutes // 60):02d}:{int(end_minutes % 60):02d}"

            discount_plan.append({
                'time_slot': f"{start_str}-{end_str}",
                'discount_percentage': round((1 - discount) * 100, 1),
                'final_price': round(final_price, 2),
                'expected_sales': actual_sales_int,  # 使用整数
                'expected_revenue': round(actual_sales_int * final_price, 2),
                'expected_profit': round(actual_sales_int * (final_price - cost_price), 2)
            })

            remaining_inventory -= actual_sales_int

        return discount_plan
    
    def _calculate_plan_metrics(self, discount_plan: List[Dict],
                              current_inventory: int,
                              base_price: float,
                              cost_price: float) -> Dict:
        """计算方案指标"""
        total_expected_sales = sum(item['expected_sales'] for item in discount_plan)
        total_expected_revenue = sum(item['expected_revenue'] for item in discount_plan)
        total_expected_profit = sum(item['expected_profit'] for item in discount_plan)
        
        clearance_rate = total_expected_sales / current_inventory if current_inventory > 0 else 0
        
        # 计算平均折扣
        weighted_discount = sum(
            item['expected_sales'] * (1 - item['discount_percentage']/100) 
            for item in discount_plan
        ) / total_expected_sales if total_expected_sales > 0 else 0
        
        avg_gross_margin = total_expected_profit / total_expected_revenue if total_expected_revenue > 0 else 0
        
        return {
            'total_expected_sales': total_expected_sales,
            'total_expected_revenue': total_expected_revenue,
            'total_expected_profit': total_expected_profit,
            'clearance_rate': clearance_rate,
            'avg_discount': round((1 - weighted_discount) * 100, 2),
            'avg_gross_margin': avg_gross_margin,
            'remaining_inventory': max(0, current_inventory - total_expected_sales)
        }
    
    def _analyze_feasibility(self, discount_plan: List[Dict],
                           plan_metrics: Dict,
                           min_gross_margin: float) -> Dict:
        """分析方案可行性"""
        feasibility = {
            'can_clear_inventory': plan_metrics['clearance_rate'] >= 0.95,
            'meets_margin_requirement': plan_metrics['avg_gross_margin'] >= min_gross_margin,
            'reasonable_pricing_strategy': True,
            'customer_acceptance_likely': True,
            'operational_feasibility': True
        }
        
        # 检查折扣变化是否合理
        if len(discount_plan) > 1:
            discounts = [item['discount_percentage'] for item in discount_plan]
            max_change = max(abs(discounts[i] - discounts[i-1]) 
                           for i in range(1, len(discounts)))
            feasibility['reasonable_pricing_strategy'] = max_change <= self.default_params['max_discount_change'] * 100
        
        # 检查顾客接受度
        avg_discount = plan_metrics['avg_discount']
        feasibility['customer_acceptance_likely'] = avg_discount <= 50  # 平均折扣不超过5折
        
        return feasibility
    
    def _generate_recommendation(self, feasibility: Dict) -> str:
        """生成推荐建议"""
        if not feasibility['can_clear_inventory']:
            return "建议增加折扣力度或延长促销时间以确保库存清空"
        
        if not feasibility['meets_margin_requirement']:
            return "当前方案无法满足最低毛利率要求，建议调整折扣策略"
        
        if not feasibility['reasonable_pricing_strategy']:
            return "折扣变化幅度较大，可能会引起顾客不满，建议平滑折扣变化"
        
        return "方案可行，建议按计划执行"