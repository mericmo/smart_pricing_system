# core/pricing_strategy_generator.py (更新版)
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict
# 使用LRU缓存
from functools import lru_cache
from data.data_processor import TransactionDataProcessor
from data.feature_engineer import PricingFeatureEngineer
from models.demand_predictor import EnhancedDemandPredictor, ProductInfo
from models.pricing_optimizer import PricingOptimizer, PricingSegment

@dataclass
class EnhancedPricingStrategy:
    """增强版定价策略"""
    strategy_id: str
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
    features_used: List[str]
    generated_time: str
    weather_consideration: bool
    calendar_consideration: bool
    confidence_score: float
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

class EnhancedPricingStrategyGenerator:
    """增强版定价策略生成器"""
    
    def __init__(self, transaction_data: pd.DataFrame,
                 weather_data: pd.DataFrame = None,
                 calendar_data: pd.DataFrame = None,
                 config=None):
        """
        初始化策略生成器
        
        Args:
            transaction_data: 交易数据DataFrame
            weather_data: 天气数据DataFrame
            calendar_data: 日历数据DataFrame
            config: 配置管理器
        """
        self.transaction_data = transaction_data
        self.weather_data = weather_data
        self.calendar_data = calendar_data
        self.config = config
        
        # 初始化组件
        self.data_processor = TransactionDataProcessor(
            transaction_data, weather_data, calendar_data
        )
        self.feature_engineer = PricingFeatureEngineer(config)
        self.demand_predictor = EnhancedDemandPredictor(
            model_type='ensemble', 
            config=config
        )
        
        # 缓存
        self._product_cache = {}
        self._product_info_cache_max_size = 100  # 可配置
        self._strategy_cache = {}
        self._model_cache = {}
    
    def generate_pricing_strategy(self,
                                 product_code: str,
                                 initial_stock: int,
                                 promotion_start: str = "20:00",
                                 promotion_end: str = "22:00",
                                 min_discount: float = 0.4,
                                 max_discount: float = 0.9,
                                 time_segments: int = 4,
                                 store_code: Optional[str] = None,
                                 current_time: Optional[datetime] = None,
                                 use_weather: bool = True,
                                 use_calendar: bool = True) -> EnhancedPricingStrategy:
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
            use_weather: 是否使用天气数据
            use_calendar: 是否使用日历数据
            
        Returns:
            EnhancedPricingStrategy: 定价策略
        """
        
        # 设置当前时间
        if current_time is None:
            current_time = pd.Timestamp.now()
        elif not isinstance(current_time, pd.Timestamp):
            current_time = pd.Timestamp(current_time)

        print(f"开始为商品 {product_code} 生成定价策略...")
        print(f"库存: {initial_stock}, 促销时段: {promotion_start}-{promotion_end}")
        
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
            store_code=store_code,
            use_weather=use_weather,
            use_calendar=use_calendar
        )
        
        # 训练需求预测模型
        self._train_demand_predictor(product_code, start_hour, end_hour, store_code)
        
        # 创建ProductInfo对象
        # product_info_obj = ProductInfo(
        #     product_code=product_code,
        #     product_name=product_info['product_name'],
        #     category=product_info.get('category', 'unknown'),
        #     price=product_info['original_price'],
        #     cost=product_info['cost_price'],
        #     weight=product_info.get('weight', 0),
        #     is_fresh=product_info.get('is_fresh', False),
        #     shelf_life_hours=product_info.get('shelf_life_hours', 24)
        # )
        
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
        total_sales = 0
        total_revenue = 0
        total_profit = 0
        
        for segment in pricing_schedule:
            segment_dict = {
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'discount': round(segment.discount, 3),
                'discount_percentage': f"{round((1-segment.discount)*100, 1)}%",
                'price': round(segment.price, 2),
                'expected_sales': segment.expected_sales,
                'expected_revenue': round(segment.revenue, 2),
                'expected_profit': round(segment.profit, 2)
            }
            schedule_dict.append(segment_dict)
            
            total_sales += segment.expected_sales
            total_revenue += segment.revenue
            total_profit += segment.profit
        
        # 评估方案
        evaluation = pricing_optimizer.evaluate_pricing_schedule(
            schedule=pricing_schedule,
            initial_stock=initial_stock
        )
        
        # 计算置信度
        confidence_score = self._calculate_confidence_score(
            product_code, store_code, features, evaluation
        )
        
        # 获取使用的特征
        features_used = list(features.keys())
        
        # 生成策略ID
        strategy_id = self._generate_strategy_id(
            product_code, promotion_start, promotion_end, current_time
        )
        
        # 创建定价策略
        strategy = EnhancedPricingStrategy(
            strategy_id=strategy_id,
            product_code=product_code,
            product_name=product_info['product_name'],
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
            features_used=features_used,
            generated_time=current_time.isoformat(),
            weather_consideration=use_weather,
            calendar_consideration=use_calendar,
            confidence_score=confidence_score
        )
        
        # 缓存策略
        self._strategy_cache[strategy_id] = strategy
        
        print(f"定价策略生成完成，策略ID: {strategy_id}")
        print(f"预期总销量: {total_sales}, 预期总利润: {total_profit:.2f}")
        print(f"售罄概率: {evaluation.get('sell_out_probability', 0):.1%}")
        
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
                'original_price': 100.0,
                'cost_price': 60.0,
                'category': 'unknown',
                'weight': 0,
                'is_fresh': False,
                'shelf_life_hours': 24,
                'price_elasticity': 1.2,
                'promotion_sensitivity': 1.2
            }
        else:
            # 从数据中提取信息
            first_row = product_data.iloc[0]
            
            product_info = {
                'product_code': product_code,
                'product_name': first_row.get('商品名称', '未知商品'),
                'original_price': float(first_row.get('售价', 100.0)),
                # 成本价带跟进
                'cost_price': self._estimate_cost_price(product_data),
                'category': first_row.get('小类编码', 'unknown') if '小类编码' in first_row else 'unknown',
                # 获取商品重量信息
                'weight': self._extract_product_weight(first_row.get('商品名称', '')),
                # 判断是否新鲜商品
                'is_fresh': self._is_fresh_product(first_row),
                # 估算保质期（小时）
                'shelf_life_hours': self._estimate_shelf_life(first_row),
                'price_elasticity': self.data_processor.get_product_summary(product_code).get('价格弹性', 1.2),
                'promotion_sensitivity': self.data_processor.get_product_summary(product_code).get('促销敏感度', 1.2)
            }
        
        # 缓存结果
        self._product_cache[cache_key] = product_info.copy()
        
        return product_info
    
    def _estimate_cost_price(self, product_data: pd.DataFrame) -> float:
        """估算成本价"""
        # # 基于售价和平均折扣估算
        # if '售价' in product_data.columns and '实际折扣率' in product_data.columns:
        #     avg_price = product_data['售价'].mean()
        #     avg_discount = product_data['实际折扣率'].mean()
        #
        #     # 假设正常利润率为30%，计算成本价
        #     normal_price = avg_price / avg_discount if avg_discount > 0 else avg_price
        #     # 假设成本价是10%
        #     cost_price = normal_price * 0.1  # 30%毛利率
        # else:
        #     # cost_price = 60.0  # 默认成本价
        #     cost_price = 0  # 默认成本价
        #
        # return float(cost_price)
        """改进的成本价估算"""
        # 方法1: 如果有折扣数据，通过折扣率反推
        if '售价' in product_data.columns and '实际折扣率' in product_data.columns:
            # 计算平均折扣后的价格
            avg_discounted_price = (product_data['售价'] * product_data['实际折扣率']).mean()

            # 假设目标毛利率（如70%）
            target_margin = 0.7
            estimated_cost = avg_discounted_price * (1 - target_margin)

            # 防止异常值
            return float(np.clip(estimated_cost,
                                 avg_discounted_price * 0.1,  # 最低成本
                                 avg_discounted_price * 0.8))  # 最高成本

        # 方法2: 基于类别估算
        if '小类编码' in product_data.columns:
            category = product_data['小类编码'].iloc[0]
            # 可配置不同类别的默认成本率
            category_cost_rates = {
                '20': 0.4,  # 生鲜类
                '30': 0.3,  # 食品类
                '40': 0.2,  # 日用品类
            }
            prefix = category[:2] if isinstance(category, str) else '00'
            default_rate = category_cost_rates.get(prefix, 0.25)

            avg_price = product_data['售价'].mean() if '售价' in product_data.columns else 100.0
            return avg_price * default_rate

        # 默认情况
        return 0.0  # 或抛出异常
    
    def _extract_product_weight(self, product_name: str) -> float:
        """从商品名称中提取重量"""
        import re
        
        if not isinstance(product_name, str):
            return 0.0
        
        patterns = [
            r'(\d+(\.\d+)?)\s*[kK]?[gG]',  # 重量，如380g
            r'(\d+(\.\d+)?)\s*[mM]?[lL]',  # 容量，如500ml
        ]
        
        for pattern in patterns:
            match = re.search(pattern, product_name)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        
        return 0.0
    
    def _is_fresh_product(self, product_row: pd.Series) -> bool:
        """判断是否为生鲜商品"""
        product_name = str(product_row.get('商品名称', ''))
        category = str(product_row.get('小类编码', ''))
        
        # 根据商品名称判断
        fresh_keywords = ['鲜', '奶', '豆腐', '面包', '糕点', '蔬菜', '水果', '肉', '鱼', '蛋']
        for keyword in fresh_keywords:
            if keyword in product_name:
                return True
        
        # 根据小类编码判断（假设20开头为生鲜）
        if category.startswith('20'):
            return True
        
        return False
    
    def _estimate_shelf_life(self, product_row: pd.Series) -> int:
        """估算保质期（小时）"""
        if self._is_fresh_product(product_row):
            product_name = str(product_row.get('商品名称', ''))
            
            # 根据商品类型判断保质期
            if any(keyword in product_name for keyword in ['鲜奶', '酸奶', '豆浆']):
                return 48  # 乳制品2天
            elif any(keyword in product_name for keyword in ['面包', '糕点', '蛋糕']):
                return 24  # 烘焙品1天
            elif any(keyword in product_name for keyword in ['豆腐', '豆花']):
                return 24  # 豆制品1天
            else:
                return 24  # 其他生鲜默认1天
        else:
            return 168  # 非生鲜商品7天
    
    def _prepare_features(self, product_code: str,
                         promotion_hours: Tuple[int, int],
                         current_time: datetime,
                         store_code: Optional[str],
                         use_weather: bool,
                         use_calendar: bool) -> Dict:
        """准备特征"""
        
        # 使用特征工程模块
        features = self.feature_engineer.create_features(
            transaction_data=self.transaction_data,
            product_code=product_code,
            promotion_hours=promotion_hours,
            current_time=current_time
        )
        
        # 添加商品特定特征
        product_summary = self.data_processor.get_product_summary(product_code, store_code)
        features.update(product_summary)
        
        # 添加门店特定特征（如果有）
        if store_code:
            store_features = self._extract_store_features(store_code, product_code)
            features.update(store_features)
        
        # 过滤不需要的特征
        if not use_weather:
            weather_keys = [k for k in features.keys() if 'weather' in k or 'temp' in k]
            for key in weather_keys:
                features.pop(key, None)
        
        if not use_calendar:
            calendar_keys = [k for k in features.keys() if 'holiday' in k or 'calendar' in k]
            for key in calendar_keys:
                features.pop(key, None)
        
        return features
    
    def _extract_store_features(self, store_code: str, product_code: str) -> Dict:
        """提取门店特征"""
        
        store_data = self.transaction_data[self.transaction_data['门店编码'] == store_code]
        store_product_data = store_data[store_data['商品编码'] == product_code]
        
        if store_product_data.empty:
            return {
                'store_sales_rank': 0.5,
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
        
        # 计算门店特征
        total_transactions = len(store_product_data)
        avg_transaction_value = store_product_data['销售金额'].mean() if '销售金额' in store_product_data.columns else 0
        
        return {
            'store_sales_rank': float(rank),
            'store_traffic_index': min(total_transactions / 100, 2.0),  # 基于交易次数估算人流
            'store_conversion_rate': min(total_transactions / 500, 1.0),  # 估算转化率
            'store_avg_transaction': float(avg_transaction_value)
        }
    
    def _train_demand_predictor(self, product_code: str, start_hour: int = 18, end_hour: int = 22,  store_code: Optional[str] = None):
        """训练需求预测器"""
        
        cache_key = f"predictor_{product_code}_{store_code}"
        if cache_key in self._model_cache:
            return
        
        try:
            # 准备训练数据
            X, y = self.demand_predictor.prepare_training_data_from_transactions(
                transaction_data=self.transaction_data,
                product_code=product_code,
                promotion_hours=(start_hour, end_hour),
                store_code=store_code
            )
            
            if len(X) >= 20:  # 至少有20个数据点
                # 获取商品信息
                product_info = self._get_product_info(product_code, store_code)
                product_info_obj = ProductInfo(
                    product_code=product_code,
                    product_name=product_info['product_name'],
                    category=product_info.get('category', 'unknown'),
                    price=product_info['original_price'],
                    cost=product_info['cost_price'],
                    weight=product_info.get('weight', 0),
                    is_fresh=product_info.get('is_fresh', False),
                    shelf_life_hours=product_info.get('shelf_life_hours', 24)
                )
                
                # 训练模型
                self.demand_predictor.train(X, y, product_info_obj)
                
                # 缓存训练状态
                self._model_cache[cache_key] = True
                
                print(f"商品 {product_code} 的需求预测模型训练完成，使用 {len(X)} 个样本")
            else:
                # 数据不足，使用启发式模型
                self._model_cache[cache_key] = False
                print(f"商品 {product_code} 的历史数据不足，使用启发式模型")
        except Exception as e:
            print(f"训练需求预测器失败: {e}")
            self._model_cache[cache_key] = False
    
    def _calculate_confidence_score(self, product_code: str,
                                  store_code: Optional[str],
                                  features: Dict,
                                  evaluation: Dict) -> float:
        """计算置信度分数"""
        
        # # 数据充足度（0-1）
        # data_sufficiency = min(features.get('total_transactions', 0) / 100, 1.0)
        #
        # # 模型准确性（基于历史预测误差，这里简化）
        # model_accuracy = 0.8  # 假设80%准确率
        #
        # # 特征完整性（0-1）
        # required_features = ['hist_avg_sales', 'price_elasticity', 'hist_promo_sales_ratio']
        # available_features = sum(1 for feat in required_features if feat in features)
        # feature_completeness = available_features / len(required_features)
        #
        # # 销售趋势稳定性（0-1）
        # sales_trend_abs = abs(features.get('sales_trend', 0))
        # trend_stability = 1.0 / (1.0 + sales_trend_abs * 10)
        #
        # # 综合置信度
        # confidence = (
        #     data_sufficiency * 0.3 +
        #     model_accuracy * 0.3 +
        #     feature_completeness * 0.2 +
        #     trend_stability * 0.2
        # )
        #
        # return min(max(confidence, 0.1), 0.95)  # 限制在0.1-0.95之间
        """改进的置信度计算"""

        scores = {}

        # 1. 数据质量分数
        total_transactions = features.get('total_transactions', 0)
        recent_days = features.get('recent_transaction_days', 0)

        data_quality = min(
            total_transactions / 200,  # 至少200条交易记录
            recent_days / 30,  # 至少30天有数据
            1.0
        )
        scores['data_quality'] = data_quality

        # 2. 特征一致性分数
        key_features = ['hist_avg_sales', 'hist_std_sales', 'price_elasticity']
        feature_values = [features.get(feat) for feat in key_features]
        feature_completeness = sum(1 for v in feature_values if v is not None) / len(key_features)
        scores['feature_completeness'] = feature_completeness

        # 3. 模型性能分数（如果有）
        cache_key = f"predictor_{product_code}_{store_code}"
        if cache_key in self._model_cache and self._model_cache[cache_key]:
            # 使用交叉验证分数（如果可用）
            model_score = features.get('cv_score', 0.7)
        else:
            # 启发式模型分数较低
            model_score = 0.5
        scores['model_performance'] = model_score

        # 4. 业务合理性分数
        sell_out_prob = evaluation.get('sell_out_probability', 0)
        # 售罄概率在0.7-0.9之间为最佳
        business_rationality = 1.0 - abs(sell_out_prob - 0.8) * 2  # 0.8为目标值
        scores['business_rationality'] = max(business_rationality, 0)

        # 加权平均（权重可配置）
        weights = {
            'data_quality': 0.25,
            'feature_completeness': 0.25,
            'model_performance': 0.30,
            'business_rationality': 0.20
        }

        confidence = sum(scores[k] * weights[k] for k in scores)
        return np.clip(confidence, 0.1, 0.95)
    
    def _generate_strategy_id(self, product_code: str,
                            promotion_start: str,
                            promotion_end: str,
                            current_time: datetime) -> str:
        """生成策略ID"""
        
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        return f"STRAT_{product_code}_{promotion_start}_{promotion_end}_{timestamp}"
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[EnhancedPricingStrategy]:
        """根据ID获取策略"""
        return self._strategy_cache.get(strategy_id)
    
    def save_strategy(self, strategy: EnhancedPricingStrategy, filepath: str):
        """保存策略到文件"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(strategy.to_json())
        
        print(f"策略已保存到: {filepath}")
    
    def load_strategy(self, filepath: str) -> EnhancedPricingStrategy:
        """从文件加载策略"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            strategy_dict = json.load(f)
        
        # 重建EnhancedPricingStrategy对象
        strategy = EnhancedPricingStrategy(**strategy_dict)
        
        # 重新缓存
        self._strategy_cache[strategy.strategy_id] = strategy
        
        return strategy

    @lru_cache(maxsize=100)
    def _get_product_info_cached(self, product_code: str, store_code: Optional[str] = None) -> Dict:
        """带缓存的商品信息获取"""
        return self._get_product_info(product_code, store_code)

    def clear_cache(self, cache_type: Optional[str] = None):
        """清理缓存"""
        if cache_type == 'product' or cache_type is None:
            self._get_product_info_cached.cache_clear()
            self._product_cache.clear()
        if cache_type == 'strategy' or cache_type is None:
            self._strategy_cache.clear()
        if cache_type == 'model' or cache_type is None:
            self._model_cache.clear()