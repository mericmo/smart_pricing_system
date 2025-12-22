# core/pricing_strategy_generator.py (更新版，使用PIL进行可视化)
"""
定价策略生成器核心模块

本模块实现了智能定价策略的生成功能，包括：
- 需求预测模型的训练和评估
- 阶梯定价方案的优化
- 策略评估和置信度计算
- 可视化图表的生成

主要类：
- ModelPerformanceMetrics: 模型性能指标数据类
- TrainingHistory: 训练历史记录数据类
- EnhancedPricingStrategy: 增强版定价策略数据类
- EnhancedPricingStrategyGenerator: 定价策略生成器主类
"""

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

# 导入自定义模块
from .model_evaluator import ModelEvaluationResult, SimplifiedModelVisualizer
from data.data_processor import TransactionDataProcessor
from data.feature_engineer import PricingFeatureEngineer
from models.demand_predictor import EnhancedDemandPredictor, ProductInfo
from models.pricing_optimizer import PricingOptimizer, PricingSegment


# matplotlib.use('Agg')  # 使用非交互式后端（已注释，可根据需要启用）
# matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体设置（已注释）
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题（已注释）
@dataclass
class ModelPerformanceMetrics:
    """
    模型性能指标数据类
    
    用于存储和展示需求预测模型的各项性能指标，包括：
    - MAE: 平均绝对误差，衡量预测值与真实值的平均偏差
    - MSE: 均方误差，对大误差更敏感
    - RMSE: 均方根误差，与目标变量同单位
    - R2: 决定系数，衡量模型解释方差的比例（0-1，越接近1越好）
    - MAPE: 平均绝对百分比误差，相对误差指标
    - SMAPE: 对称平均绝对百分比误差，对零值更友好
    """
    mae: float  # 平均绝对误差 (Mean Absolute Error)
    mse: float  # 均方误差 (Mean Squared Error)
    rmse: float  # 均方根误差 (Root Mean Squared Error)
    r2: float  # 决定系数 (R-squared)
    mape: float  # 平均绝对百分比误差 (Mean Absolute Percentage Error)
    smape: float  # 对称平均绝对百分比误差 (Symmetric Mean Absolute Percentage Error)

    def to_dict(self) -> Dict:
        """
        转换为字典格式
        
        Returns:
            Dict: 包含所有性能指标的字典，数值已四舍五入
        """
        return {
            'mae': round(self.mae, 3),
            'mse': round(self.mse, 3),
            'rmse': round(self.rmse, 3),
            'r2': round(self.r2, 3),
            'mape': round(self.mape, 2),
            'smape': round(self.smape, 2)
        }


@dataclass
class TrainingHistory:
    """
    训练历史记录数据类
    
    记录需求预测模型的训练过程信息，包括训练参数、性能指标和可视化结果。
    用于追踪模型训练历史，支持模型性能分析和对比。
    """
    product_code: str  # 商品编码
    store_code: Optional[str]  # 门店编码（可选，None表示全门店）
    training_time: str  # 训练时间（ISO格式字符串）
    sample_count: int  # 训练样本数量
    feature_count: int  # 特征数量
    performance_metrics: ModelPerformanceMetrics  # 模型性能指标
    plot_paths: Dict[str, str]  # 各种可视化图表的文件路径字典
    feature_importance: Optional[Dict[str, float]] = None  # 特征重要性字典（可选）

    def to_dict(self) -> Dict:
        """
        转换为字典格式
        
        Returns:
            Dict: 包含所有训练历史信息的字典
        """
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
    """
    增强版定价策略数据类
    
    包含完整的定价策略信息，包括：
    - 商品基本信息（编码、名称、价格等）
    - 促销设置（时间段、折扣范围等）
    - 阶梯定价方案（各时间段的定价策略）
    - 策略评估结果（预期销量、利润、售罄概率等）
    - 模型性能指标和可视化结果
    """
    strategy_id: str  # 策略唯一标识符
    product_code: str  # 商品编码
    product_name: str  # 商品名称
    original_price: float  # 商品原价
    cost_price: float  # 商品成本价
    initial_stock: int  # 初始库存数量
    promotion_start: str  # 促销开始时间（格式："HH:MM"）
    promotion_end: str  # 促销结束时间（格式："HH:MM"）
    min_discount: float  # 最低折扣（0.4表示4折）
    max_discount: float  # 最高折扣（0.9表示9折）
    time_segments: int  # 时间段数量
    pricing_schedule: List[Dict]  # 阶梯定价方案列表，每个元素包含时间段、折扣、价格、预期销量等
    evaluation: Dict  # 策略评估结果字典（包含总销量、总利润、售罄概率等）
    features_used: List[str]  # 使用的特征列表
    generated_time: str  # 策略生成时间（ISO格式字符串）
    weather_consideration: bool  # 是否考虑天气因素
    calendar_consideration: bool  # 是否考虑日历因素（节假日等）
    confidence_score: float  # 策略置信度分数（0-1之间）
    model_performance: Optional[Dict] = None  # 模型性能指标字典（可选）
    visualization_paths: Optional[Dict[str, str]] = None  # 可视化图表路径字典（可选）

    def to_dict(self) -> Dict:
        """
        转换为字典格式
        
        Returns:
            Dict: 包含所有策略信息的字典
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        转换为JSON字符串
        
        Returns:
            str: JSON格式的策略字符串，使用UTF-8编码，支持中文
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class EnhancedPricingStrategyGenerator:
    """
    增强版定价策略生成器
    
    核心功能：
    1. 基于历史交易数据训练需求预测模型
    2. 根据库存、促销时段等约束条件优化阶梯定价方案
    3. 评估策略效果并计算置信度
    4. 生成可视化报告和图表
    
    主要方法：
    - generate_pricing_strategy: 生成完整的定价策略
    - _train_demand_predictor: 训练需求预测模型
    - _prepare_features: 准备特征数据
    - _calculate_confidence_score: 计算策略置信度
    """

    def __init__(self, transaction_data: pd.DataFrame,
                 weather_data: pd.DataFrame = None,
                 calendar_data: pd.DataFrame = None,
                 config=None):
        """
        初始化策略生成器
        
        Args:
            transaction_data: 交易数据DataFrame，必须包含商品编码、日期、销售数量等字段
            weather_data: 天气数据DataFrame（可选），用于考虑天气对需求的影响
            calendar_data: 日历数据DataFrame（可选），用于考虑节假日等因素
            config: 配置管理器对象（可选），包含系统配置参数
        """
        # 注释掉的数据预处理代码
        # transaction_data = transaction_data[transaction_data['销售数量'] > 0]
        # if "销售净额" in transaction_data.columns and "销售金额" in transaction_data.columns:
        #     transaction_data['平均售价'] = transaction_data['销售净额'] / transaction_data['销售净额'] * transaction_data['销售数量']
        #     # 确保销售数量为数值（若有非数字或空值会变为 NaN）
        #     transaction_data["销售数量"] = pd.to_numeric(transaction_data["销售数量"], errors="coerce")
        #     # 确保金额列为数值
        #     transaction_data['销售净额'] = pd.to_numeric(transaction_data['销售净额'], errors="coerce")
        #     transaction_data["平均售价"] = np.where(
        #         transaction_data["销售数量"] > 0,
        #         transaction_data['销售净额'] / transaction_data["销售数量"],
        #         np.nan
        #     )

        # 存储输入数据
        # self._transaction_data = transaction_data # 原始数据
        # self.weather_data = weather_data
        # self.calendar_data = calendar_data
        self.config = config

        # 初始化核心组件
        # 数据处理器：负责数据清洗、筛选和汇总
        self.data_processor = TransactionDataProcessor(
            transaction_data, weather_data, calendar_data
        )
        self.transaction_data = self.data_processor.transaction_data
        self.calendar_data = self.data_processor.calendar_data
        self.weather_data = self.data_processor.weather_data
        # 特征工程器：负责特征创建和转换
        self.feature_engineer = PricingFeatureEngineer(config)
        # 需求预测器：使用XGBoost模型进行需求预测
        self.demand_predictor = EnhancedDemandPredictor(
            # model_type='ensemble',  # 可选：集成模型
            model_type='xgboost',  # 使用XGBoost模型
            config=self.config
        )

        # 初始化缓存系统（提升性能，避免重复计算）
        self._product_cache = {}  # 商品信息缓存
        self._product_info_cache_max_size = 100  # 商品信息缓存最大容量（可配置）
        self._strategy_cache = {}  # 策略缓存
        self._model_cache = {}  # 模型训练状态缓存
        self._training_history = {}  # 训练历史记录缓存

        # 初始化可视化器：用于生成模型评估和策略报告图表
        self.visualizer = SimplifiedModelVisualizer()

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

        # 设置当前时间（用于特征生成和策略ID生成）
        if current_time is None:
            # current_time = pd.Timestamp.now()
            raise ValueError("日期时间不能为空")
        elif not isinstance(current_time, pd.Timestamp):
            current_time = pd.Timestamp(current_time)

        # 注释掉的代码：可用于计算当日已售库存（可根据需要启用）
        # self.current_date = pd.to_datetime('2025-10-31')
        # if product_code and store_code:
        #     self.config['today_saled_stock'] = self.transaction_data[
        #         (self.transaction_data['门店编码'] == store_code) &
        #         (self.transaction_data['日期'] == current_time.date()) & (
        #                     self.transaction_data['商品编码'] == product_code)]['销售数量'].sum()

        print(f"开始为商品 {product_code} 生成定价策略...")
        print(f"库存: {initial_stock}, 促销时段: {promotion_start}-{promotion_end}")

        # 解析促销时间字符串，提取小时和分钟
        start_hour, start_minute = map(int, promotion_start.split(':'))
        end_hour, end_minute = map(int, promotion_end.split(':'))

        # 获取商品基本信息（名称、价格、成本、类别等）
        product_info = self._get_product_info(product_code, store_code)

        # 准备特征数据（历史销量、价格弹性、天气、节假日等）
        features = self._prepare_features(
            product_code=product_code,
            promotion_hours=(start_hour, end_hour),
            current_time=current_time,
            store_code=store_code,
            use_weather=use_weather,
            use_calendar=use_calendar
        )

        # 训练需求预测模型，并获取训练历史（包含性能指标和可视化结果）
        training_history = self._train_demand_predictor(
            product_code, start_hour, end_hour, store_code, generate_visualizations
        )

        # 初始化定价优化器（用于生成最优的阶梯定价方案）
        pricing_optimizer = PricingOptimizer(
            demand_predictor=self.demand_predictor,
            cost_price=product_info['cost_price'],
            original_price=product_info['original_price']
        )

        # 生成阶梯定价方案（优化各时间段的折扣和价格）
        pricing_schedule = pricing_optimizer.optimize_staged_pricing(
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount,
            time_segments=time_segments,
            features=features
        )

        # 将定价方案转换为字典格式（便于序列化和展示）
        schedule_dict = []
        total_sales = 0  # 累计预期销量
        total_revenue = 0  # 累计预期收入
        total_profit = 0  # 累计预期利润

        for segment in pricing_schedule:
            # 构建单个时间段的定价信息字典
            segment_dict = {
                'start_time': segment.start_time,  # 开始时间
                'end_time': segment.end_time,  # 结束时间
                'discount': round(segment.discount, 3),  # 折扣系数（0-1）
                'discount_percentage': f"{round((1 - segment.discount) * 100, 1)}%",  # 折扣百分比（如"20%"表示8折）
                'price': round(segment.price, 2),  # 执行价格
                'expected_sales': segment.expected_sales,  # 预期销量
                'expected_revenue': round(segment.revenue, 2),  # 预期收入
                'expected_profit': round(segment.profit, 2)  # 预期利润
            }
            schedule_dict.append(segment_dict)

            # 累计统计
            total_sales += segment.expected_sales
            total_revenue += segment.revenue
            total_profit += segment.profit

        # 评估定价方案（计算售罄概率、利润率等指标）
        evaluation = pricing_optimizer.evaluate_pricing_schedule(
            schedule=pricing_schedule,
            initial_stock=initial_stock
        )

        # 计算策略置信度（基于数据质量、模型性能、业务合理性等）
        confidence_score = self._calculate_confidence_score(
            product_code, store_code, features, evaluation
        )

        # 获取使用的特征列表（用于策略报告）
        features_used = list(features.keys())

        # 生成唯一策略ID（包含商品编码、促销时段和时间戳）
        strategy_id = self._generate_strategy_id(
            product_code, promotion_start, promotion_end, current_time
        )

        # 生成可视化图表（如果启用）
        visualization_paths = None
        if generate_visualizations:
            print(f"[DEBUG] 开始生成可视化图表")
            try:
                # 确保product_info包含初始库存（用于可视化）
                product_info['initial_stock'] = initial_stock

                # 生成策略可视化图表（使用PIL进行图像处理）
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

        # 创建定价策略对象
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

        # 缓存策略（便于后续查询和重用）
        self._strategy_cache[strategy_id] = strategy

        print(f"定价策略生成完成，策略ID: {strategy_id}")
        print(f"预期总销量: {total_sales}, 预期总利润: {total_profit:.2f}")
        print(f"售罄概率: {evaluation.get('sell_out_probability', 0):.1%}")

        return strategy

    def _train_demand_predictor(self, product_code: str, start_hour: int = 18, end_hour: int = 22,
                                store_code: Optional[str] = None, generate_visualizations: bool = True) -> Optional[
        TrainingHistory]:
        """
        训练需求预测模型，返回训练历史
        
        该方法负责：
        1. 准备训练数据（从交易数据中提取特征和标签）
        2. 训练XGBoost需求预测模型
        3. 计算模型性能指标（MAE、RMSE、R2等）
        4. 生成训练可视化图表（如果启用）
        5. 缓存训练结果，避免重复训练
        
        Args:
            product_code: 商品编码
            start_hour: 促销开始小时（默认18点）
            end_hour: 促销结束小时（默认22点）
            store_code: 门店编码（可选，None表示全门店）
            generate_visualizations: 是否生成可视化图表
            
        Returns:
            Optional[TrainingHistory]: 训练历史对象，包含性能指标和可视化路径
        """
        # 检查缓存，如果已训练过则直接返回历史记录
        cache_key = f"predictor_{product_code}_{store_code}"
        if cache_key in self._model_cache and cache_key in self._training_history:
            return self._training_history.get(cache_key)

        # 准备训练数据（特征矩阵X和标签向量y）
        X, y = self.demand_predictor.prepare_training_data_from_transactions(
            transaction_data=self.transaction_data,
            product_code=product_code,
            promotion_hours=(start_hour, end_hour),  # 只使用促销时段的数据
            store_code=store_code
        )

        print(
            f"[DEBUG] 训练数据准备完成: X.shape={X.shape if hasattr(X, 'shape') else 'N/A'}, y.shape={y.shape if hasattr(y, 'shape') else 'N/A'}")

        # 数据质量检查（样本数量是否足够）
        data_warning = None
        if hasattr(y, '__len__') and len(y) < 10:
            data_warning = f"数据样本过少（{len(y)}个），建议收集更多数据"
            print(f"[DEBUG] 数据警告: {data_warning}")
        elif hasattr(y, '__len__') and len(y) == 0:
            data_warning = "无可用数据"
            print(f"[DEBUG] 数据警告: {data_warning}")

        # 创建一个基础的TrainingHistory对象
        # 即使数据不足或训练失败，也返回一个基础对象（保证接口一致性）
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

        # 如果数据太少（少于10个样本），直接返回基础历史（不进行训练）
        if len(X) < 10:
            print(f"[DEBUG] 数据不足，使用基础训练历史")
            self._model_cache[cache_key] = False  # 标记为未训练
            self._training_history[cache_key] = base_history
            return base_history

        try:
            # 获取商品信息（用于模型训练）
            product_info = self._get_product_info(product_code, store_code)
            # 构建ProductInfo对象（模型需要的商品信息格式）
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

            # 训练需求预测模型（XGBoost）
            self.demand_predictor.train(X, y, product_info_obj)

            # 缓存训练状态（标记为已训练）
            self._model_cache[cache_key] = True

            # 计算模型性能指标（在训练集上预测并评估）
            y_pred = self.demand_predictor.predict_train_set(X)  # 获取训练集预测结果
            metrics = self._calculate_performance_metrics(y, y_pred)  # 计算各项指标

            # 更新训练历史（用实际性能指标替换基础指标）
            base_history.performance_metrics = metrics
            base_history.sample_count = len(X)

            # 生成模型可视化图表（如果启用）
            if generate_visualizations:
                try:
                    # 创建评估结果对象（用于可视化）
                    evaluation_result = ModelEvaluationResult(
                        y_true=y,  # 真实值
                        y_pred=y_pred,  # 预测值
                        metrics=metrics.to_dict(),  # 性能指标
                        feature_names=list(X.columns) if hasattr(X, 'columns') else [],  # 特征名称
                        feature_importance=None,  # 特征重要性（可选）
                        data_quality_warning=data_warning  # 数据质量警告
                    )

                    # 生成综合报告（包含预测散点图、残差图、误差分布等）
                    plot_paths = self.visualizer.create_comprehensive_report(
                        evaluation_result=evaluation_result,
                        product_code=product_code,
                        store_code=store_code
                    )

                    base_history.plot_paths = plot_paths  # 保存图表路径
                except Exception as e:
                    print(f"[DEBUG] 生成训练可视化失败: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            # 训练失败时记录错误并标记为未训练
            print(f"[DEBUG] 训练过程异常: {e}")
            self._model_cache[cache_key] = False

        # 缓存并返回训练历史（无论成功或失败都返回）
        self._training_history[cache_key] = base_history
        return base_history

    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelPerformanceMetrics:
        """
        计算模型性能指标
        
        计算多种评估指标，全面评估模型性能：
        - MAE: 平均绝对误差，单位与目标变量相同
        - MSE: 均方误差，对大误差更敏感
        - RMSE: 均方根误差，常用指标
        - R2: 决定系数，衡量模型解释方差的比例
        - MAPE: 平均绝对百分比误差，相对误差指标
        - SMAPE: 对称平均绝对百分比误差，对零值更友好
        
        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            
        Returns:
            ModelPerformanceMetrics: 包含所有性能指标的对象
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # 基础指标计算
        mae = mean_absolute_error(y_true, y_pred)  # 平均绝对误差
        mse = mean_squared_error(y_true, y_pred)  # 均方误差
        rmse = np.sqrt(mse)  # 均方根误差
        r2 = r2_score(y_true, y_pred)  # 决定系数（R²）

        # 计算MAPE（平均绝对百分比误差）
        # 注意：需要避免除以0的情况
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0

        # 计算SMAPE（对称平均绝对百分比误差）
        # SMAPE对零值更友好，分母是真实值和预测值的绝对值之和
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        if np.any(mask):
            smape = 2.0 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
        else:
            smape = 0.0

        return ModelPerformanceMetrics(
            mae=mae, mse=mse, rmse=rmse, r2=r2, mape=mape, smape=smape
        )

    def _get_product_info(self, product_code: str, store_code: Optional[str] = None) -> Dict:
        """
        获取商品信息
        
        从交易数据中提取商品的基本信息，包括：
        - 商品名称、编码、类别
        - 原价、成本价（估算）
        - 是否生鲜商品、保质期
        - 价格弹性、促销敏感度
        
        如果商品数据不存在，返回默认值。
        结果会被缓存以提升性能。
        
        Args:
            product_code: 商品编码
            store_code: 门店编码（可选）
            
        Returns:
            Dict: 包含商品信息的字典
        """
        # 检查缓存
        cache_key = f"{product_code}_{store_code}"
        if cache_key in self._product_cache:
            return self._product_cache[cache_key].copy()

        # 筛选商品数据（根据商品编码和门店编码）
        product_data = self.data_processor.filter_by_product(product_code, store_code)

        if product_data.empty:
            # 如果没有数据，使用默认值（保证系统正常运行）
            product_info = {
                'product_code': product_code,
                'product_name': '未知商品',
                'original_price': 100.0,  # 默认原价
                'cost_price': 60.0,  # 默认成本价
                'category': 'unknown',
                # 'weight': 0,  # 重量信息（已注释）
                'is_fresh': False,  # 默认非生鲜
                'shelf_life_hours': 24,  # 默认保质期24小时
                'price_elasticity': 1.2,  # 默认价格弹性
                'promotion_sensitivity': 1.2  # 默认促销敏感度
            }
        else:
            # 从数据中提取信息
            # first_row = product_data.iloc[0]
            first_row = product_data.iloc[-1]
            summary_info = self.data_processor.get_product_summary(product_code, store_code)
            product_info = {
                'product_code': product_code,
                'product_name': first_row.get('商品名称', '未知商品'),
                'original_price': float(first_row.get('售价', 0)),  # 从售价字段获取原价
                # 成本价通过估算方法获得（基于折扣数据或类别）
                'cost_price': self._estimate_cost_price(product_data),
                'category': first_row.get('小类编码', 'unknown') if '小类编码' in first_row else 'unknown',
                # 获取商品重量信息（已注释，可根据需要启用）
                # 'weight': self._extract_product_weight(first_row.get('商品名称', '')),
                # 判断是否新鲜商品（基于商品名称和类别）
                'is_fresh': self._is_fresh_product(first_row),
                # 估算保质期（小时，基于商品类型）
                'shelf_life_hours': self._estimate_shelf_life(first_row),
                # 从数据汇总中获取价格弹性和促销敏感度
                'price_elasticity': summary_info.get('价格弹性',
                                                                                                          1.2),
                'promotion_sensitivity': summary_info.get(
                    '促销敏感度', 1.2)
            }

        # 缓存结果（避免重复计算）
        self._product_cache[cache_key] = product_info.copy()

        return product_info

    def _estimate_cost_price(self, product_data: pd.DataFrame) -> float:
        """
        改进的成本价估算方法
        
        使用多种方法估算商品成本价：
        1. 如果有折扣数据，通过折扣后的价格和目标毛利率反推成本
        2. 基于商品类别使用默认成本率估算
        3. 如果都不可用，返回0.0
        
        Args:
            product_data: 商品交易数据DataFrame
            
        Returns:
            float: 估算的成本价
        """
        # 方法1: 如果有折扣数据，通过折扣率反推成本
        if '售价' in product_data.columns and '实际折扣率' in product_data.columns:
            # 计算平均折扣后的价格（实际销售价格）
            avg_discounted_price = (product_data['售价'] * product_data['实际折扣率']).mean()

            # 假设目标毛利率（如70%，即成本占30%）
            target_margin = 0.7
            estimated_cost = avg_discounted_price * (1 - target_margin)

            # 防止异常值（成本应在合理范围内）
            return float(np.clip(estimated_cost,
                                 avg_discounted_price * 0.1,  # 最低成本（不低于售价的10%）
                                 avg_discounted_price * 0.9))  # 最高成本（不高于售价的90%）

        # 方法2: 基于类别估算（不同类别有不同的成本率）
        if '小类编码' in product_data.columns:
            category = product_data['小类编码'].iloc[0]
            # 可配置不同类别的默认成本率（成本占售价的比例）
            category_cost_rates = {
                '20': 0.4,  # 生鲜类（成本率40%）
                '30': 0.3,  # 食品类（成本率30%）
                '40': 0.2,  # 日用品类（成本率20%）
            }
            prefix = category[:2] if isinstance(category, str) else '00'  # 取类别编码前两位
            default_rate = category_cost_rates.get(prefix, 0.25)  # 默认成本率25%

            avg_price = product_data['售价'].mean() if '售价' in product_data.columns else 0.0
            return avg_price * default_rate

        # 默认情况（无法估算时返回0.0）
        return 0.0  # 或抛出异常

    def _extract_product_weight(self, product_name: str) -> float:
        """
        从商品名称中提取重量或容量
        
        使用正则表达式从商品名称中提取重量（g/kg）或容量（ml/L）信息。
        例如："福荫川式豆花380g" -> 380.0
        
        Args:
            product_name: 商品名称字符串
            
        Returns:
            float: 提取的重量或容量值，如果未找到则返回0.0
        """
        import re

        if not isinstance(product_name, str):
            return 0.0

        # 定义匹配模式（支持重量和容量）
        patterns = [
            r'(\d+(\.\d+)?)\s*[kK]?[gG]',  # 重量，如380g、1.5kg
            r'(\d+(\.\d+)?)\s*[mM]?[lL]',  # 容量，如500ml、1.2L
        ]

        # 尝试匹配每个模式
        for pattern in patterns:
            match = re.search(pattern, product_name)
            if match:
                try:
                    return float(match.group(1))  # 提取数字部分
                except:
                    continue

        return 0.0  # 未找到匹配则返回0.0

    def _is_fresh_product(self, product_row: pd.Series) -> bool:
        """
        判断是否为生鲜商品
        
        通过商品名称关键词和小类编码判断商品是否为生鲜商品。
        生鲜商品通常需要更频繁的促销和更短的保质期。
        
        Args:
            product_row: 商品数据行（Series）
            
        Returns:
            bool: True表示是生鲜商品，False表示不是
        """
        product_name = str(product_row.get('商品名称', ''))
        category = str(product_row.get('小类编码', ''))

        # 方法1: 根据商品名称关键词判断
        fresh_keywords = ['鲜', '奶', '豆腐', '面包', '糕点', '蔬菜', '水果', '肉', '鱼', '蛋']
        for keyword in fresh_keywords:
            if keyword in product_name:
                return True

        # 方法2: 根据小类编码判断（假设20开头为生鲜类别）
        if category.startswith('20'):
            return True

        return False

    def _estimate_shelf_life(self, product_row: pd.Series) -> int:
        """
        估算商品保质期（单位：小时）
        
        根据商品类型估算保质期，用于判断商品的紧迫性。
        保质期越短，促销策略越需要激进。
        
        Args:
            product_row: 商品数据行（Series）
            
        Returns:
            int: 保质期（小时）
        """
        if self._is_fresh_product(product_row):
            # 生鲜商品的保质期较短
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
            # 非生鲜商品的保质期较长
            return 168  # 非生鲜商品7天

    def _prepare_features(self, product_code: str,
                          promotion_hours: Tuple[int, int],
                          current_time: datetime,
                          store_code: Optional[str],
                          use_weather: bool,
                          use_calendar: bool) -> Dict:
        """
        准备特征数据
        
        整合多种特征来源，构建用于需求预测的特征字典：
        1. 基础特征（历史销量、价格等）- 通过特征工程模块生成
        2. 商品特定特征（价格弹性、促销敏感度等）
        3. 门店特定特征（门店排名、流量指数等）
        4. 外部特征（天气、日历）- 根据参数决定是否包含
        
        Args:
            product_code: 商品编码
            promotion_hours: 促销时段（开始小时，结束小时）
            current_time: 当前时间
            store_code: 门店编码（可选）
            use_weather: 是否使用天气特征
            use_calendar: 是否使用日历特征（节假日等）
            
        Returns:
            Dict: 特征字典，键为特征名，值为特征值
        """
        calendar_data = self.calendar_data if use_calendar else None
        weather_data = self.weather_data if use_weather else None
        # 使用特征工程模块创建基础特征（历史销量、价格趋势等）
        features = self.feature_engineer.create_features(
            transaction_data=self.transaction_data,
            calendar_data=calendar_data,
            weather_data=weather_data,
            product_code=product_code,
            store_code=store_code,
            promotion_hours=promotion_hours,
            current_time=current_time
        )

        # 添加商品特定特征（从数据汇总中获取）
        product_summary = self.data_processor.get_product_summary(product_code, store_code)
        features.update(product_summary)

        # 添加门店特定特征
        if store_code:
            store_features = self._extract_store_features(store_code, product_code)
            features.update(store_features)

        # 根据参数过滤不需要的特征
        if not use_weather:
            # 移除天气相关特征
            weather_keys = [k for k in features.keys() if 'weather' in k or 'temp' in k]
            for key in weather_keys:
                features.pop(key, None)

        if not use_calendar:
            # 移除日历相关特征（节假日等）
            calendar_keys = [k for k in features.keys() if 'holiday' in k or 'calendar' in k]
            for key in calendar_keys:
                features.pop(key, None)

        return features

    def _extract_store_features(self, store_code: str, product_code: str) -> Dict:
        """
        提取门店特定特征
        
        计算门店在该商品上的表现特征，用于个性化定价策略：
        - 门店销售排名：该门店在所有门店中的销售排名
        - 门店流量指数：基于交易次数估算的人流量
        - 门店转化率：估算的购买转化率
        - 门店平均交易额：平均每笔交易的金额
        
        Args:
            store_code: 门店编码
            product_code: 商品编码
            
        Returns:
            Dict: 门店特征字典
        """
        # 筛选门店数据和该门店的商品数据
        store_data = self.transaction_data[self.transaction_data['门店编码'] == store_code]
        store_product_data = store_data[store_data['商品编码'] == product_code]

        # 如果门店没有该商品的销售数据，返回默认值
        if store_product_data.empty:
            return {
                'store_sales_rank': 0.5,  # 默认中等排名
                'store_traffic_index': 1.0,  # 默认正常流量
                'store_conversion_rate': 0.3  # 默认转化率30%
            }

        # 计算门店销售排名（相对于所有门店）
        store_sales = store_product_data['销售数量'].sum()  # 该门店的总销量
        all_stores_sales = self.transaction_data[
            self.transaction_data['商品编码'] == product_code
            ].groupby('门店编码')['销售数量'].sum()  # 所有门店的销量

        if len(all_stores_sales) > 0:
            # 排名 = 销量超过该门店的门店数 / 总门店数（0-1之间，越小排名越靠前）
            rank = (all_stores_sales > store_sales).sum() / len(all_stores_sales)
        else:
            rank = 0.5  # 默认中等排名

        # 计算门店特征
        total_transactions = len(store_product_data)  # 交易次数
        avg_transaction_value = store_product_data['销售金额'].mean() if '销售金额' in store_product_data.columns else 0

        return {
            'store_sales_rank': float(rank),  # 门店销售排名（0-1）
            'store_traffic_index': min(total_transactions / 100, 2.0),  # 基于交易次数估算人流（上限2.0）
            'store_conversion_rate': min(total_transactions / 500, 1.0),  # 估算转化率（上限1.0）
            'store_avg_transaction': float(avg_transaction_value)  # 平均交易额
        }

    def _calculate_confidence_score(self, product_code: str,
                                    store_code: Optional[str],
                                    features: Dict,
                                    evaluation: Dict) -> float:
        """
        计算策略置信度分数
        
        综合考虑多个维度计算策略的置信度：
        1. 数据质量：交易记录数量和覆盖天数
        2. 特征完整性：关键特征是否齐全
        3. 模型性能：预测模型的性能表现
        4. 业务合理性：售罄概率是否在合理范围内
        
        置信度分数用于评估策略的可靠性，帮助决策者判断是否采用该策略。
        
        Args:
            product_code: 商品编码
            store_code: 门店编码（可选）
            features: 特征字典
            evaluation: 策略评估结果字典
            
        Returns:
            float: 置信度分数（0.1-0.95之间）
        """
        scores = {}

        # 1. 数据质量分数（基于交易记录数量和覆盖天数）
        total_transactions = features.get('total_transactions', 0)  # 总交易记录数
        recent_days = features.get('recent_transaction_days', 0)  # 有交易记录的天数

        data_quality = min(
            total_transactions / 200,  # 至少200条交易记录为满分
            recent_days / 30,  # 至少30天有数据为满分
            1.0  # 上限为1.0
        )
        scores['data_quality'] = data_quality

        # 2. 特征完整性分数（关键特征是否齐全）
        key_features = ['hist_avg_sales', 'hist_std_sales', 'price_elasticity']
        feature_values = [features.get(feat) for feat in key_features]
        feature_completeness = sum(1 for v in feature_values if v is not None) / len(key_features)
        scores['feature_completeness'] = feature_completeness

        # 3. 模型性能分数（如果模型已训练）
        cache_key = f"predictor_{product_code}_{store_code}"
        if cache_key in self._model_cache and self._model_cache[cache_key]:
            # 使用交叉验证分数（如果可用），否则使用默认值
            model_score = features.get('cv_score', 0.7)
        else:
            # 如果模型未训练，使用启发式方法，分数较低
            model_score = 0.5
        scores['model_performance'] = model_score

        # 4. 业务合理性分数（售罄概率是否在合理范围内）
        sell_out_prob = evaluation.get('sell_out_probability', 0)
        # 售罄概率在0.7-0.9之间为最佳，0.8为目标值
        business_rationality = 1.0 - abs(sell_out_prob - 0.8) * 2  # 距离0.8越远分数越低
        scores['business_rationality'] = max(business_rationality, 0)  # 确保非负

        # 加权平均计算最终置信度（权重可配置）
        weights = {
            'data_quality': 0.25,  # 数据质量权重25%
            'feature_completeness': 0.25,  # 特征完整性权重25%
            'model_performance': 0.30,  # 模型性能权重30%（最重要）
            'business_rationality': 0.20  # 业务合理性权重20%
        }

        confidence = sum(scores[k] * weights[k] for k in scores)
        # 限制置信度在0.1-0.95之间（避免极端值）
        return np.clip(confidence, 0.1, 0.95)

    def _generate_strategy_id(self, product_code: str,
                              promotion_start: str,
                              promotion_end: str,
                              current_time: datetime) -> str:
        """
        生成策略唯一标识符
        
        格式：STRAT_{商品编码}_{开始时间}_{结束时间}_{时间戳}
        例如：STRAT_8006144_20:00_22:00_20251031_102125
        
        Args:
            product_code: 商品编码
            promotion_start: 促销开始时间（格式："HH:MM"）
            promotion_end: 促销结束时间（格式："HH:MM"）
            current_time: 当前时间
            
        Returns:
            str: 策略ID字符串
        """
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        return f"STRAT_{product_code}_{promotion_start}_{promotion_end}_{timestamp}"

    def get_strategy_by_id(self, strategy_id: str) -> Optional[EnhancedPricingStrategy]:
        """
        根据策略ID获取已缓存的策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[EnhancedPricingStrategy]: 策略对象，如果不存在则返回None
        """
        return self._strategy_cache.get(strategy_id)

    def save_strategy(self, strategy: EnhancedPricingStrategy, filepath: str):
        """
        保存策略到JSON文件
        
        Args:
            strategy: 定价策略对象
            filepath: 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(strategy.to_json())

        print(f"策略已保存到: {filepath}")

    def load_strategy(self, filepath: str) -> EnhancedPricingStrategy:
        """
        从JSON文件加载策略
        
        Args:
            filepath: 策略文件路径
            
        Returns:
            EnhancedPricingStrategy: 加载的策略对象
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            strategy_dict = json.load(f)

        # 重建EnhancedPricingStrategy对象
        strategy = EnhancedPricingStrategy(**strategy_dict)

        # 重新缓存（便于后续查询）
        self._strategy_cache[strategy.strategy_id] = strategy

        return strategy

    @lru_cache(maxsize=100)
    def _get_product_info_cached(self, product_code: str, store_code: Optional[str] = None) -> Dict:
        """
        带缓存的商品信息获取（使用LRU缓存）
        
        这是_get_product_info的缓存版本，使用functools.lru_cache装饰器。
        最多缓存100个商品的信息。
        
        Args:
            product_code: 商品编码
            store_code: 门店编码（可选）
            
        Returns:
            Dict: 商品信息字典
        """
        return self._get_product_info(product_code, store_code)

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        清理缓存
        
        可以清理指定类型的缓存，或清理所有缓存。
        
        Args:
            cache_type: 缓存类型，可选值：
                - 'product': 只清理商品信息缓存
                - 'strategy': 只清理策略缓存
                - 'model': 只清理模型和训练历史缓存
                - None: 清理所有缓存
        """
        if cache_type == 'product' or cache_type is None:
            self._get_product_info_cached.cache_clear()  # 清理LRU缓存
            self._product_cache.clear()  # 清理字典缓存
        if cache_type == 'strategy' or cache_type is None:
            self._strategy_cache.clear()
        if cache_type == 'model' or cache_type is None:
            self._model_cache.clear()
            self._training_history.clear()

    def get_training_history(self, product_code: str, store_code: Optional[str] = None) -> Optional[TrainingHistory]:
        """
        获取指定商品的训练历史
        
        Args:
            product_code: 商品编码
            store_code: 门店编码（可选）
            
        Returns:
            Optional[TrainingHistory]: 训练历史对象，如果不存在则返回None
        """
        cache_key = f"predictor_{product_code}_{store_code}"
        return self._training_history.get(cache_key)
