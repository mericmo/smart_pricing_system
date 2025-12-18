# core/config.py
import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import time
import logging

@dataclass
class DataConfig:
    """数据配置"""
    transaction_data_path: str = "data/historical_transactions.csv"
    db_path: str = "data/sales_data.db"
    max_history_days: int = 365
    min_transactions_for_training: int = 50
    
@dataclass
class FeatureConfig:
    """特征工程配置"""
    time_features: List[str] = field(default_factory=lambda: [
        'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'quarter'
    ])
    historical_features: List[str] = field(default_factory=lambda: [
        'hist_avg_sales', 'hist_sales_std', 'hist_promo_sales_ratio',
        'sales_trend', 'last_week_sales'
    ])
    price_features: List[str] = field(default_factory=lambda: [
        'avg_price', 'price_std', 'min_price', 'max_price',
        'avg_discount', 'price_elasticity'
    ])
    customer_features: List[str] = field(default_factory=lambda: [
        'price_sensitive_customers', 'loyal_customers',
        'average_basket_size', 'repeat_purchase_rate'
    ])
    normalize_features: bool = True
    feature_scaling_method: str = "minmax"  # "minmax", "standard", "robust"

@dataclass
class PredictionConfig:
    """预测模型配置"""
    model_type: str = "xgboost"  # xgboost, lightgbm, catboost, ensemble
    ensemble_models: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "random_forest"
    ])
    train_test_split_ratio: float = 0.8
    cross_validation_folds: int = 5
    hyperparameter_tuning: bool = True
    prediction_horizon_minutes: int = 30
    confidence_level: float = 0.95
    
@dataclass
class OptimizationConfig:
    """优化配置"""
    time_segments: int = 4
    min_segment_duration_minutes: int = 15
    max_segment_duration_minutes: int = 60
    discount_precision: float = 0.01  # 折扣精度
    optimization_method: str = "dynamic_programming"  # "dp", "genetic", "rl"
    objective_function: str = "max_profit"  # "max_profit", "max_sales", "balanced"
    constraints: Dict[str, Any] = field(default_factory=lambda: {
        "min_profit_margin": 0.1,
        "max_discount_change_per_step": 0.15,
        "min_time_between_changes": 900,  # 15分钟
        "max_price_changes": 10
    })

@dataclass
class RLConfig:
    """强化学习配置"""
    enabled: bool = False
    state_dim: int = 10
    action_dim: int = 7  # 7种折扣调整动作
    hidden_dim: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    target_update_frequency: int = 100
    training_episodes: int = 10000
    evaluation_interval: int = 100

@dataclass
class RealTimeConfig:
    """实时调整配置"""
    enabled: bool = True
    monitoring_interval_seconds: int = 300  # 5分钟
    adjustment_threshold: float = 0.3  # 30%偏差触发调整
    max_adjustments_per_session: int = 3
    adjustment_delay_minutes: int = 5
    performance_metrics_window: int = 30  # 使用最近30分钟数据

@dataclass
class ProductCategoryConfig:
    """商品品类配置"""
    name: str
    price_elasticity: float
    shelf_life_hours: int
    base_demand_rate: float
    promotion_sensitivity: float
    clearance_discount_pattern: str  # "aggressive", "moderate", "conservative"

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        # self._setup_logging()
        
        # 初始化各配置类
        # self.data_config = DataConfig(**self.config.get('data', {}))
        # self.feature_config = FeatureConfig(**self.config.get('features', {}))
        # self.prediction_config = PredictionConfig(**self.config.get('prediction', {}))
        # self.optimization_config = OptimizationConfig(**self.config.get('optimization', {}))
        # self.rl_config = RLConfig(**self.config.get('rl', {}))
        # self.real_time_config = RealTimeConfig(**self.config.get('real_time', {}))
        
        # 加载商品品类配置
        # self.product_categories = self._load_product_categories()
        
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        # 首先尝试当前目录
        local_config = "config.yaml"
        if os.path.exists(local_config):
            return local_config
        
        # 尝试项目根目录
        project_root = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        if os.path.exists(project_root):
            return project_root
        
        # 创建默认配置
        return self._create_default_config()
    
    def _create_default_config(self) -> str:
        """创建默认配置文件"""
        default_config = {
            'data': {
                'transaction_data_path': 'data/historical_transactions.csv',
                'db_path': 'data/sales_data.db',
                'max_history_days': 365,
                'min_transactions_for_training': 50
            },
            'features': {
                'time_features': ['hour_of_day', 'day_of_week', 'is_weekend', 'month', 'quarter'],
                'historical_features': ['hist_avg_sales', 'hist_sales_std', 'hist_promo_sales_ratio', 'sales_trend', 'last_week_sales'],
                'price_features': ['avg_price', 'price_std', 'min_price', 'max_price', 'avg_discount', 'price_elasticity'],
                'customer_features': ['price_sensitive_customers', 'loyal_customers', 'average_basket_size', 'repeat_purchase_rate'],
                'normalize_features': True,
                'feature_scaling_method': 'minmax'
            },
            'prediction': {
                'model_type': 'xgboost',
                'ensemble_models': ['xgboost', 'lightgbm', 'random_forest'],
                'train_test_split_ratio': 0.8,
                'cross_validation_folds': 5,
                'hyperparameter_tuning': True,
                'prediction_horizon_minutes': 30,
                'confidence_level': 0.95
            },
            'optimization': {
                'time_segments': 4,
                'min_segment_duration_minutes': 15,
                'max_segment_duration_minutes': 60,
                'discount_precision': 0.01,
                'optimization_method': 'dynamic_programming',
                'objective_function': 'max_profit',
                'constraints': {
                    'min_profit_margin': 0.1,
                    'max_discount_change_per_step': 0.15,
                    'min_time_between_changes': 900,
                    'max_price_changes': 10
                }
            },
            'rl': {
                'enabled': False,
                'state_dim': 10,
                'action_dim': 7,
                'hidden_dim': 64,
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'memory_size': 10000,
                'batch_size': 64,
                'target_update_frequency': 100,
                'training_episodes': 10000,
                'evaluation_interval': 100
            },
            'real_time': {
                'enabled': True,
                'monitoring_interval_seconds': 300,
                'adjustment_threshold': 0.3,
                'max_adjustments_per_session': 3,
                'adjustment_delay_minutes': 5,
                'performance_metrics_window': 30
            },
            'product_categories': {
                'fresh_food': {
                    'price_elasticity': 1.5,
                    'shelf_life_hours': 24,
                    'base_demand_rate': 10.0,
                    'promotion_sensitivity': 1.8,
                    'clearance_discount_pattern': 'aggressive'
                },
                'bakery': {
                    'price_elasticity': 1.3,
                    'shelf_life_hours': 12,
                    'base_demand_rate': 8.0,
                    'promotion_sensitivity': 1.5,
                    'clearance_discount_pattern': 'moderate'
                },
                'dairy': {
                    'price_elasticity': 1.2,
                    'shelf_life_hours': 48,
                    'base_demand_rate': 12.0,
                    'promotion_sensitivity': 1.3,
                    'clearance_discount_pattern': 'conservative'
                },
                'default': {
                    'price_elasticity': 1.2,
                    'shelf_life_hours': 24,
                    'base_demand_rate': 5.0,
                    'promotion_sensitivity': 1.0,
                    'clearance_discount_pattern': 'moderate'
                }
            }
        }
        
        # 保存到文件
        config_dir = os.path.dirname(self._get_default_config_path())
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        config_path = os.path.join(config_dir, "config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return config_path
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            return {}
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'pricing_system.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _load_product_categories(self) -> Dict[str, ProductCategoryConfig]:
        """加载商品品类配置"""
        categories = {}
        categories_config = self.config.get('product_categories', {})
        
        for cat_name, cat_config in categories_config.items():
            categories[cat_name] = ProductCategoryConfig(
                name=cat_name,
                price_elasticity=cat_config.get('price_elasticity', 1.2),
                shelf_life_hours=cat_config.get('shelf_life_hours', 24),
                base_demand_rate=cat_config.get('base_demand_rate', 5.0),
                promotion_sensitivity=cat_config.get('promotion_sensitivity', 1.0),
                clearance_discount_pattern=cat_config.get('clearance_discount_pattern', 'moderate')
            )
        
        return categories
    
    def get_product_category(self, category_name: str) -> ProductCategoryConfig:
        """获取商品品类配置"""
        if category_name in self.product_categories:
            return self.product_categories[category_name]
        return self.product_categories.get('default', self.product_categories['default'])
    
    def save_config(self, path: str = None):
        """保存配置到文件"""
        save_path = path or self.config_path
        
        # 构建配置字典
        config_dict = {
            'data': self.data_config.__dict__,
            'features': self.feature_config.__dict__,
            'prediction': self.prediction_config.__dict__,
            'optimization': self.optimization_config.__dict__,
            'rl': self.rl_config.__dict__,
            'real_time': self.real_time_config.__dict__,
            'product_categories': {
                name: cat.__dict__
                for name, cat in self.product_categories.items()
            }
        }
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def update_config(self, section: str, key: str, value: Any):
        """更新配置"""
        if hasattr(self, f"{section}_config"):
            config_obj = getattr(self, f"{section}_config")
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                
                # 更新内存中的配置字典
                if section in self.config:
                    if isinstance(self.config[section], dict):
                        self.config[section][key] = value
                else:
                    self.config[section] = {key: value}