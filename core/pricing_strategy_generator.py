# core/pricing_strategy_generator.py (更新版，使用PIL进行可视化)
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from functools import lru_cache

# 导入PIL用于图像处理

import matplotlib

import matplotlib.pyplot as plt

from .model_evaluator import ModelEvaluationResult, SimplifiedModelVisualizer
from data.data_processor import TransactionDataProcessor
from data.feature_engineer import PricingFeatureEngineer
from models.demand_predictor import EnhancedDemandPredictor, ProductInfo
from models.pricing_optimizer import PricingOptimizer, PricingSegment
# matplotlib.use('Agg')  # 使用非交互式后端
# matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    mae: float  # 平均绝对误差
    mse: float  # 均方误差
    rmse: float  # 均方根误差
    r2: float  # 决定系数
    mape: float  # 平均绝对百分比误差
    smape: float  # 对称平均绝对百分比误差

    def to_dict(self) -> Dict:
        """转换为字典 - 修复：返回数值而不是字符串"""
        return {
            'mae': round(self.mae, 3),
            'mse': round(self.mse, 3),
            'rmse': round(self.rmse, 3),
            'r2': round(self.r2, 3),
            'mape': round(self.mape, 2),  # 返回数值，不是字符串
            'smape': round(self.smape, 2)  # 返回数值，不是字符串
        }

@dataclass 
class TrainingHistory:
    """训练历史记录"""
    product_code: str
    store_code: Optional[str]
    training_time: str
    sample_count: int
    feature_count: int
    performance_metrics: ModelPerformanceMetrics
    plot_paths: Dict[str, str]  # 各种图表路径
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'product_code': self.product_code,
            'store_code': self.store_code,
            'training_time': self.training_time,
            'sample_count': self.sample_count,
            'feature_count': self.feature_count,
            'performance_metrics': self.performance_metrics.to_dict(),
            'plot_paths': self.plot_paths,
            'feature_importance': self.feature_importance
        }

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
    model_performance: Optional[Dict] = None  # 新增：模型性能指标
    visualization_paths: Optional[Dict[str, str]] = None  # 新增：可视化图表路径
    
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
        self._training_history = {}  # 新增：训练历史记录
        

        # 初始化可视化器
        self.visualizer = SimplifiedModelVisualizer()
        # 设置matplotlib样式
        plt.style.use('default')
    
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
                                 use_calendar: bool = True,
                                 generate_visualizations: bool = True) -> EnhancedPricingStrategy:
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
            generate_visualizations: 是否生成可视化图表
            
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
        
        # 训练需求预测模型，并获取训练历史
        training_history = self._train_demand_predictor(
            product_code, start_hour, end_hour, store_code, generate_visualizations
        )
        
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
        
        # 生成可视化图表（如果启用）
        visualization_paths = None
        if generate_visualizations:
            print(f"[DEBUG] 开始生成可视化图表")
            try:
                # 确保product_info包含初始库存
                product_info['initial_stock'] = initial_stock

                visualization_paths = self.visualizer._generate_strategy_visualizations_with_pil(
                    strategy_id=strategy_id,
                    pricing_schedule=pricing_schedule,
                    product_info=product_info,
                    features=features,
                    evaluation=evaluation,
                    training_history=training_history,
                    total_sales=total_sales,
                    total_profit=total_profit,
                    confidence_score=confidence_score
                )
                print(f"[DEBUG] 可视化图表生成结果: {visualization_paths}")
            except Exception as e:
                print(f"[DEBUG] 生成可视化图表时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG] generate_visualizations参数为False，跳过可视化生成")
        
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
            confidence_score=confidence_score,
            model_performance=training_history.to_dict() if training_history else None,
            visualization_paths=visualization_paths
        )
        
        # 缓存策略
        self._strategy_cache[strategy_id] = strategy
        
        print(f"定价策略生成完成，策略ID: {strategy_id}")
        print(f"预期总销量: {total_sales}, 预期总利润: {total_profit:.2f}")
        print(f"售罄概率: {evaluation.get('sell_out_probability', 0):.1%}")
        
        return strategy

    def _train_demand_predictor(self, product_code: str, start_hour: int = 18, end_hour: int = 22,
                               store_code: Optional[str] = None, generate_visualizations: bool = True) -> Optional[TrainingHistory]:
        """训练需求预测器，返回训练历史"""

        cache_key = f"predictor_{product_code}_{store_code}"
        if cache_key in self._model_cache and cache_key in self._training_history:
            return self._training_history.get(cache_key)

        # 准备训练数据
        X, y = self.demand_predictor.prepare_training_data_from_transactions(
            transaction_data=self.transaction_data,
            product_code=product_code,
            promotion_hours=(start_hour, end_hour),
            store_code=store_code
        )

        print(f"[DEBUG] 训练数据准备完成: X.shape={X.shape if hasattr(X, 'shape') else 'N/A'}, y.shape={y.shape if hasattr(y, 'shape') else 'N/A'}")

        # 数据质量检查
        data_warning = None
        if hasattr(y, '__len__') and len(y) < 10:
            data_warning = f"数据样本过少（{len(y)}个），建议收集更多数据"
            print(f"[DEBUG] 数据警告: {data_warning}")
        elif hasattr(y, '__len__') and len(y) == 0:
            data_warning = "无可用数据"
            print(f"[DEBUG] 数据警告: {data_warning}")

        # 创建一个基础的TrainingHistory对象
        # 即使数据不足或训练失败，也返回一个基础对象
        base_metrics = ModelPerformanceMetrics(
            mae=0.0, mse=0.0, rmse=0.0, r2=0.0, mape=0.0, smape=0.0
        )

        base_history = TrainingHistory(
            product_code=product_code,
            store_code=store_code,
            training_time=datetime.now().isoformat(),
            sample_count=len(X) if hasattr(X, '__len__') else 0,
            feature_count=X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 0,
            performance_metrics=base_metrics,
            plot_paths={},
            feature_importance=None
        )

        # 如果数据太少，直接返回基础历史
        if len(X) < 10:
            print(f"[DEBUG] 数据不足，使用基础训练历史")
            self._model_cache[cache_key] = False
            self._training_history[cache_key] = base_history
            return base_history

        try:
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

            # 计算性能指标
            y_pred = self.demand_predictor.predict(X) if hasattr(self.demand_predictor, 'predict') else np.zeros_like(y)
            metrics = self._calculate_performance_metrics(y, y_pred)

            # 更新训练历史
            base_history.performance_metrics = metrics
            base_history.sample_count = len(X)
            # 可视化模型信息：
            if generate_visualizations:
                try:
                    # 创建评估结果对象
                    evaluation_result = ModelEvaluationResult(
                        y_true=y,
                        y_pred=y_pred,
                        metrics=metrics.to_dict(),
                        feature_names=list(X.columns) if hasattr(X, 'columns') else [],
                        feature_importance=None,
                        data_quality_warning=data_warning
                    )

                    # 生成综合报告
                    plot_paths = self.visualizer.create_comprehensive_report(
                        evaluation_result=evaluation_result,
                        product_code=product_code,
                        store_code=store_code
                    )

                    base_history.plot_paths = plot_paths
                except Exception as e:
                    print(f"[DEBUG] 生成训练可视化失败: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"[DEBUG] 训练过程异常: {e}")
            self._model_cache[cache_key] = False

        # 缓存并返回训练历史
        self._training_history[cache_key] = base_history
        return base_history
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelPerformanceMetrics:
        """计算模型性能指标"""
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # 计算MAPE（平均绝对百分比误差）
        # 避免除以0
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        # 计算SMAPE（对称平均绝对百分比误差）
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        if np.any(mask):
            smape = 2.0 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
        else:
            smape = 0.0
        
        return ModelPerformanceMetrics(
            mae=mae, mse=mse, rmse=rmse, r2=r2, mape=mape, smape=smape
        )


    # 以下方法保持不变
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
    
    def _calculate_confidence_score(self, product_code: str,
                                  store_code: Optional[str],
                                  features: Dict,
                                  evaluation: Dict) -> float:
        """计算置信度分数"""
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
            self._training_history.clear()
    
    def get_training_history(self, product_code: str, store_code: Optional[str] = None) -> Optional[TrainingHistory]:
        """获取训练历史"""
        cache_key = f"predictor_{product_code}_{store_code}"
        return self._training_history.get(cache_key)