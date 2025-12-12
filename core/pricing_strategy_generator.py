# core/pricing_strategy_generator.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict

from data.data_processor import TransactionDataProcessor
from data.feature_engineer import PricingFeatureEngineer
from models.demand_predictor import DemandPredictor
from models.pricing_optimizer import PricingOptimizer, PricingSegment

@dataclass
class PricingStrategy:
    """定价策略"""
    product_code: str
    product_name: str
    original_price: float
    cost_price: float
    initial_stock: int
    promotion_start: str
    promotion_end: str
    min_discount: float
    max_discount: float
    time_segments: int
    pricing_schedule: List[Dict]
    evaluation: Dict
    generated_time: str
    strategy_id: Optional[str] = None

class PricingStrategyGenerator:
    """定价策略生成器"""
    
    def __init__(self, transaction_data: pd.DataFrame):
        """
        初始化策略生成器
        
        Args:
            transaction_data: 交易数据DataFrame
        """
        self.transaction_data = transaction_data
        self.data_processor = TransactionDataProcessor(transaction_data)
        self.feature_engineer = PricingFeatureEngineer()
        self.demand_predictor = DemandPredictor(model_type='xgboost')
        
        # 缓存
        self._product_cache = {}
        self._strategy_cache = {}
    
    def generate_pricing_strategy(self,
                                 product_code: str,
                                 initial_stock: int,
                                 promotion_start: str = "20:00",
                                 promotion_end: str = "22:00",
                                 min_discount: float = 0.4,
                                 max_discount: float = 0.9,
                                 time_segments: int = 4,
                                 store_code: Optional[str] = None,
                                 current_time: Optional[datetime] = None) -> PricingStrategy:
        """
        生成定价策略
        
        Args:
            product_code: 商品编码
            initial_stock: 初始库存
            promotion_start: 促销开始时间，格式 "HH:MM"
            promotion_end: 促销结束时间，格式 "HH:MM"
            min_discount: 最低折扣（0.4表示4折）
            max_discount: 最高折扣（0.9表示9折）
            time_segments: 时间段数量
            store_code: 门店编码（可选）
            current_time: 当前时间（可选）
            
        Returns:
            PricingStrategy: 定价策略
        """
        
        # 设置当前时间
        if current_time is None:
            current_time = datetime.now()
        
        # 解析促销时间
        start_hour, start_minute = map(int, promotion_start.split(':'))
        end_hour, end_minute = map(int, promotion_end.split(':'))
        
        # 获取商品信息
        product_info = self._get_product_info(product_code, store_code)
        
        # 准备特征
        features = self._prepare_features(
            product_code=product_code,
            promotion_hours=(start_hour, end_hour),
            current_time=current_time,
            store_code=store_code
        )
        
        # 训练需求预测模型（如果还未训练）
        self._train_demand_predictor(product_code, store_code)
        
        # 初始化定价优化器
        pricing_optimizer = PricingOptimizer(
            demand_predictor=self.demand_predictor,
            cost_price=product_info['cost_price'],
            original_price=product_info['original_price']
        )
        
        # 生成阶梯定价方案
        pricing_schedule = pricing_optimizer.optimize_staged_pricing(
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount,
            time_segments=time_segments,
            features=features
        )
        
        # 转换为字典格式
        schedule_dict = []
        for segment in pricing_schedule:
            segment_dict = {
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'discount': round(segment.discount, 3),
                'discount_percentage': f"{round((1-segment.discount)*100, 1)}%",
                'price': segment.price,
                'expected_sales': segment.expected_sales,
                'expected_revenue': segment.revenue,
                'expected_profit': segment.profit
            }
            schedule_dict.append(segment_dict)
        
        # 评估方案
        evaluation = pricing_optimizer.evaluate_pricing_schedule(
            schedule=pricing_schedule,
            initial_stock=initial_stock
        )
        
        # 生成策略ID
        strategy_id = self._generate_strategy_id(
            product_code, promotion_start, promotion_end, current_time
        )
        
        # 创建定价策略
        strategy = PricingStrategy(
            product_code=product_code,
            product_name=product_info.get('product_name', '未知商品'),
            original_price=product_info['original_price'],
            cost_price=product_info['cost_price'],
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount,
            time_segments=time_segments,
            pricing_schedule=schedule_dict,
            evaluation=evaluation,
            generated_time=current_time.isoformat(),
            strategy_id=strategy_id
        )
        
        # 缓存策略
        self._strategy_cache[strategy_id] = strategy
        
        return strategy
    
    def _get_product_info(self, product_code: str, store_code: Optional[str] = None) -> Dict:
        """获取商品信息"""
        
        cache_key = f"{product_code}_{store_code}"
        if cache_key in self._product_cache:
            return self._product_cache[cache_key].copy()
        
        # 筛选商品数据
        product_data = self.data_processor.filter_by_product(product_code, store_code)
        
        if product_data.empty:
            # 如果没有数据，使用默认值
            product_info = {
                'product_code': product_code,
                'product_name': '未知商品',
                'original_price': 100.0,  # 默认原价
                'cost_price': 60.0,  # 默认成本价
                'category': 'unknown',
                'price_elasticity': 1.2,
                'promotion_sensitivity': 1.0
            }
        else:
            # 从数据中提取信息
            product_info = {
                'product_code': product_code,
                'product_name': product_data['商品名称'].iloc[0] if '商品名称' in product_data.columns else '未知商品',
                'original_price': product_data['售价'].mean() if '售价' in product_data.columns else 100.0,
                'cost_price': self._estimate_cost_price(product_data),
                'category': product_data['小类编码'].iloc[0] if '小类编码' in product_data.columns else 'unknown',
                'price_elasticity': self.data_processor.get_product_summary(product_code)['价格弹性'],
                'promotion_sensitivity': self.data_processor.get_product_summary(product_code)['促销敏感度']
            }
        
        # 缓存结果
        self._product_cache[cache_key] = product_info.copy()
        
        return product_info
    
    def _estimate_cost_price(self, product_data: pd.DataFrame) -> float:
        """估算成本价"""
        # 简单估算：假设毛利率为30%
        if '售价' in product_data.columns and not product_data['售价'].isna().all():
            avg_price = product_data['售价'].mean()
            return avg_price * 0.7  # 30%毛利率
        else:
            return 60.0  # 默认成本价
    
    def _prepare_features(self, product_code: str,
                         promotion_hours: Tuple[int, int],
                         current_time: datetime,
                         store_code: Optional[str] = None) -> Dict:
        """准备特征"""
        
        # 使用特征工程模块
        features = self.feature_engineer.create_features(
            transaction_data=self.transaction_data,
            product_code=product_code,
            promotion_hours=promotion_hours,
            current_time=current_time
        )
        
        # 添加商品特定特征
        product_summary = self.data_processor.get_product_summary(product_code)
        customer_insights = self.data_processor.get_customer_price_sensitivity(product_code)
        
        features.update(product_summary)
        features.update(customer_insights)
        
        # 添加门店特定特征（如果有）
        if store_code:
            store_features = self._extract_store_features(store_code, product_code)
            features.update(store_features)
        
        return features
    
    def _extract_store_features(self, store_code: str, product_code: str) -> Dict:
        """提取门店特征"""
        
        store_data = self.transaction_data[self.transaction_data['门店编码'] == store_code]
        store_product_data = store_data[store_data['商品编码'] == product_code]
        
        if store_product_data.empty:
            return {
                'store_sales_rank': 0.5,  # 默认中等排名
                'store_traffic_index': 1.0,
                'store_conversion_rate': 0.3
            }
        
        # 计算门店销售排名
        store_sales = store_product_data['销售数量'].sum()
        all_stores_sales = self.transaction_data[
            self.transaction_data['商品编码'] == product_code
        ].groupby('门店编码')['销售数量'].sum()
        
        if len(all_stores_sales) > 0:
            rank = (all_stores_sales > store_sales).sum() / len(all_stores_sales)
        else:
            rank = 0.5
        
        # 计算转化率
        transactions = len(store_product_data)
        # 这里简化计算，实际可能需要更多数据
        
        return {
            'store_sales_rank': rank,
            'store_traffic_index': 1.0,  # 可扩展：基于门店历史数据
            'store_conversion_rate': min(transactions / 100, 1.0) if transactions > 0 else 0.3
        }
    
    def _train_demand_predictor(self, product_code: str, store_code: Optional[str] = None):
        """训练需求预测器"""
        
        # 检查是否已训练
        cache_key = f"predictor_{product_code}_{store_code}"
        if cache_key in self._product_cache:
            return
        
        try:
            # 准备训练数据
            X, y = self.demand_predictor.prepare_training_data(
                transaction_data=self.transaction_data,
                product_code=product_code,
                promotion_hours=(20, 22)  # 默认促销时段
            )
            
            if len(X) >= 10:  # 至少有10个数据点
                # 训练模型
                self.demand_predictor.train(X, y)
                
                # 缓存训练状态
                self._product_cache[cache_key] = True
            else:
                # 数据不足，使用启发式模型
                self._product_cache[cache_key] = False
        except Exception as e:
            print(f"训练需求预测器失败: {e}")
            self._product_cache[cache_key] = False
    
    def _generate_strategy_id(self, product_code: str,
                            promotion_start: str,
                            promotion_end: str,
                            current_time: datetime) -> str:
        """生成策略ID"""
        
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        return f"STRAT_{product_code}_{promotion_start}_{promotion_end}_{timestamp}"
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[PricingStrategy]:
        """根据ID获取策略"""
        return self._strategy_cache.get(strategy_id)
    
    def save_strategy(self, strategy: PricingStrategy, filepath: str):
        """保存策略到文件"""
        
        strategy_dict = asdict(strategy)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(strategy_dict, f, ensure_ascii=False, indent=2)
    
    def load_strategy(self, filepath: str) -> PricingStrategy:
        """从文件加载策略"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            strategy_dict = json.load(f)
        
        # 重建PricingStrategy对象
        strategy = PricingStrategy(**strategy_dict)
        
        # 重新缓存
        if strategy.strategy_id:
            self._strategy_cache[strategy.strategy_id] = strategy
        
        return strategy
    
    def generate_alternative_strategies(self, product_code: str,
                                      initial_stock: int,
                                      promotion_start: str = "20:00",
                                      promotion_end: str = "22:00",
                                      min_discount: float = 0.4,
                                      max_discount: float = 0.9) -> Dict[str, Any]:
        """生成备选策略"""
        
        alternatives = {}
        
        # 1. 激进策略（更多时段，更快降价）
        aggressive_strategy = self.generate_pricing_strategy(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=0.95,  # 允许更高折扣
            time_segments=6,  # 更多时段
            current_time=datetime.now()
        )
        
        alternatives['aggressive'] = {
            'strategy': aggressive_strategy,
            'description': '激进策略：更多时段，更快降价，追求高售罄率',
            'suitable_for': '库存压力大，保质期临近'
        }
        
        # 2. 保守策略（较少时段，缓慢降价）
        conservative_strategy = self.generate_pricing_strategy(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=0.6,  # 限制最低折扣
            max_discount=0.95,
            time_segments=3,  # 较少时段
            current_time=datetime.now()
        )
        
        alternatives['conservative'] = {
            'strategy': conservative_strategy,
            'description': '保守策略：较少时段，缓慢降价，追求高利润率',
            'suitable_for': '库存压力小，商品价值高'
        }
        
        # 3. 单一价格策略（非阶梯定价）
        product_info = self._get_product_info(product_code)
        pricing_optimizer = PricingOptimizer(
            demand_predictor=self.demand_predictor,
            cost_price=product_info['cost_price'],
            original_price=product_info['original_price']
        )
        
        start_hour, start_minute = map(int, promotion_start.split(':'))
        end_hour, end_minute = map(int, promotion_end.split(':'))
        
        single_discount = pricing_optimizer.optimize_single_price(
            initial_stock=initial_stock,
            promotion_hours=(start_hour, end_hour),
            features=self._prepare_features(
                product_code=product_code,
                promotion_hours=(start_hour, end_hour),
                current_time=datetime.now()
            )
        )
        
        alternatives['single_price'] = {
            'discount': single_discount,
            'price': product_info['original_price'] * single_discount,
            'description': '单一价格策略：整个促销时段统一价格',
            'suitable_for': '运营简单，避免频繁调价'
        }
        
        return alternatives
    
    def validate_strategy_feasibility(self, product_code: str,
                                    initial_stock: int,
                                    promotion_start: str,
                                    promotion_end: str) -> Dict:
        """验证策略可行性"""
        
        # 获取商品信息
        product_info = self.data_processor.get_product_summary(product_code)
        
        # 计算促销时段长度（小时）
        start_hour = int(promotion_start.split(':')[0])
        end_hour = int(promotion_end.split(':')[0])
        if end_hour <= start_hour:
            end_hour += 24
        promotion_hours = end_hour - start_hour
        
        # 计算历史平均销售速率
        hist_avg_sales = product_info.get('hist_avg_sales', 5)
        hist_sales_std = product_info.get('hist_sales_std', 2)
        
        # 估算最大可能销量（在最低折扣下）
        max_sales_potential = hist_avg_sales * promotion_hours * 2.0  # 假设促销能提升100%
        
        # 评估可行性
        feasibility_score = min(max_sales_potential / initial_stock, 2.0)
        
        if feasibility_score < 0.5:
            feasibility = '高风险'
            recommendation = '库存过高，时间窗口不足。建议：1) 提前开始促销 2) 加大折扣力度 3) 考虑捆绑销售'
        elif feasibility_score < 0.8:
            feasibility = '中等风险'
            recommendation = '可能需要加大促销力度。建议：1) 使用更激进的阶梯定价 2) 配合营销活动'
        elif feasibility_score < 1.2:
            feasibility = '可行'
            recommendation = '在合理促销下可以售罄。建议使用标准阶梯定价策略'
        else:
            feasibility = '容易达成'
            recommendation = '售罄目标容易达成。可以考虑更保守的策略以提高利润'
        
        return {
            'feasibility': feasibility,
            'feasibility_score': round(feasibility_score, 2),
            'initial_stock': initial_stock,
            'max_sales_potential': round(max_sales_potential),
            'recommendation': recommendation,
            'historical_avg_sales': hist_avg_sales,
            'promotion_hours': promotion_hours
        }