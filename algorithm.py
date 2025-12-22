# algorithm.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib
import os
warnings.filterwarnings('ignore')

class SalesPredictor:
    """
    销售预测算法类
    从FeatureStore获取特征进行模型训练和预测
    支持多种算法和全面的模型评估
    """
    
    def __init__(self, forecast_horizon=7, model_dir="models"):
        """
        初始化销售预测器
        
        参数:
        - forecast_horizon: 预测天数
        - model_dir: 模型保存目录
        """
        self.forecast_horizon = forecast_horizon
        self.model_dir = model_dir
        self.models = {}
        self.feature_importance = {}
        self.evaluation_results = {}
        self.feature_columns = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # 确保模型目录存在
        os.makedirs(model_dir, exist_ok=True)
        
        # 支持的算法列表
        self.supported_algorithms = [
            'lightgbm', 'xgboost', 'random_forest', 
            'gradient_boosting', 'linear_regression', 
            'ridge', 'lasso', 'svr'
        ]
    
    def prepare_features_for_training(self, features_df, target_col='销售数量', test_size=0.2, random_state=42):
        """
        准备训练特征
        
        参数:
        - features_df: 特征数据框
        - target_col: 目标列名
        - test_size: 测试集比例
        - random_state: 随机种子
        
        返回:
        - X_train, X_test, y_train, y_test, feature_columns
        """
        # 选择数值型特征和分类特征
        numeric_features = self._get_numeric_features(features_df)
        categorical_features = self._get_categorical_features(features_df)
        
        # 移除目标变量和标识列
        feature_columns = [col for col in numeric_features + categorical_features 
                          if col not in ['门店编码', '商品编码', '商品名称', '日期', target_col]]
        
        # 准备数据
        X = features_df[feature_columns].copy()
        y = features_df[target_col]
        
        # 处理分类特征 - 对每个分类特征进行编码
        for col in categorical_features:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # 处理可能的未知值
                    X[col] = X[col].astype(str)
                    self.label_encoders[col].fit(X[col])
                
                X[col] = self.label_encoders[col].transform(X[col])
        
        # 处理无穷大值和过大的值
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # 填充NaN值
        X = X.fillna(0)
        y = y.fillna(0)
        
        # 限制数值范围，避免过大值
        for col in X.select_dtypes(include=[np.number]).columns:
            max_val = X[col].quantile(0.99)
            min_val = X[col].quantile(0.01)
            X[col] = X[col].clip(min_val, max_val)
        
        # 标准化数值特征（对某些算法有帮助）
        if 'standard_scaler' not in self.scalers:
            self.scalers['standard_scaler'] = StandardScaler()
            X_scaled = self.scalers['standard_scaler'].fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        else:
            X_scaled = self.scalers['standard_scaler'].transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 关键修改：确保数据按时间顺序排序后再划分
        # 添加日期索引用于排序
        if '日期' in features_df.columns:
            # 按日期排序整个数据集
            sorted_indices = features_df['日期'].sort_values().index
            X_scaled = X_scaled.loc[sorted_indices]
            y = y.loc[sorted_indices]
        
        # 划分训练测试集 - 基于时间顺序的最后20%
        split_index = int(len(X_scaled) * (1 - test_size))
        X_train = X_scaled.iloc[:split_index]
        X_test = X_scaled.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        # 保存特征列信息
        self.feature_columns['all'] = feature_columns
        self.feature_columns['numeric'] = numeric_features
        self.feature_columns['categorical'] = categorical_features
        # X_train.to_csv
        return X_train, X_test, y_train, y_test, feature_columns

    
    def train_lightgbm(self, features_df, target_col='销售数量', params=None):
        """
        训练LightGBM模型
        """
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        
        # 准备特征
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features_for_training(
            features_df, target_col
        )
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # 保存模型和特征重要性
        self.models['lightgbm'] = model
        self.feature_importance['lightgbm'] = dict(zip(
            feature_columns, model.feature_importance()
        ))
        
        # 评估模型
        predictions = model.predict(X_test)
        self.evaluation_results['lightgbm'] = self._evaluate_model(y_test, predictions)
        
        return model
    
    def train_xgboost(self, features_df, target_col='销售数量', params=None):
        """
        训练XGBoost模型
        """
        if params is None:
            params = {
                'max_depth': 6,
                'eta': 0.1,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            }
        
        # 准备特征
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features_for_training(
            features_df, target_col
        )
        # X_train.to_csv("X.csv", encodings='utf-8')
        # 训练模型 - 移除不支持的参数
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 保存模型和特征重要性
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = dict(zip(
            feature_columns, model.feature_importances_
        ))
        
        # 评估模型
        predictions = model.predict(X_test)
        self.evaluation_results['xgboost'] = self._evaluate_model(y_test, predictions)
        
        return model
    
    def train_random_forest(self, features_df, target_col='销售数量', params=None):
        """
        训练随机森林模型
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # 准备特征
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features_for_training(
            features_df, target_col
        )
        
        # 训练模型 - 移除不支持的参数
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # 保存模型和特征重要性
        self.models['random_forest'] = model
        self.feature_importance['random_forest'] = dict(zip(
            feature_columns, model.feature_importances_
        ))
        
        # 评估模型
        predictions = model.predict(X_test)
        self.evaluation_results['random_forest'] = self._evaluate_model(y_test, predictions)
        
        return model
    
    def train_gradient_boosting(self, features_df, target_col='销售数量', params=None):
        """
        训练梯度提升模型
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        
        # 准备特征
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features_for_training(
            features_df, target_col
        )
        
        # 训练模型
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        # 保存模型和特征重要性
        self.models['gradient_boosting'] = model
        self.feature_importance['gradient_boosting'] = dict(zip(
            feature_columns, model.feature_importances_
        ))
        
        # 评估模型
        predictions = model.predict(X_test)
        self.evaluation_results['gradient_boosting'] = self._evaluate_model(y_test, predictions)
        
        return model
    
    def train_linear_regression(self, features_df, target_col='销售数量'):
        """
        训练线性回归模型
        """
        # 准备特征
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features_for_training(
            features_df, target_col
        )
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 保存模型和特征重要性
        self.models['linear_regression'] = model
        # 线性回归的特征重要性是系数的绝对值
        self.feature_importance['linear_regression'] = dict(zip(
            feature_columns, np.abs(model.coef_)
        ))
        
        # 评估模型
        predictions = model.predict(X_test)
        self.evaluation_results['linear_regression'] = self._evaluate_model(y_test, predictions)
        
        return model
    
    def train_ridge(self, features_df, target_col='销售数量', params=None):
        """
        训练岭回归模型
        """
        if params is None:
            params = {'alpha': 1.0, 'random_state': 42}
        
        # 准备特征
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features_for_training(
            features_df, target_col
        )
        
        # 训练模型
        model = Ridge(**params)
        model.fit(X_train, y_train)
        
        # 保存模型和特征重要性
        self.models['ridge'] = model
        # 岭回归的特征重要性是系数的绝对值
        self.feature_importance['ridge'] = dict(zip(
            feature_columns, np.abs(model.coef_)
        ))
        
        # 评估模型
        predictions = model.predict(X_test)
        self.evaluation_results['ridge'] = self._evaluate_model(y_test, predictions)
        
        return model
    
    def train_lasso(self, features_df, target_col='销售数量', params=None):
        """
        训练Lasso回归模型
        """
        if params is None:
            params = {'alpha': 1.0, 'random_state': 42}
        
        # 准备特征
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features_for_training(
            features_df, target_col
        )
        
        # 训练模型
        model = Lasso(**params)
        model.fit(X_train, y_train)
        
        # 保存模型和特征重要性
        self.models['lasso'] = model
        # Lasso回归的特征重要性是系数的绝对值
        self.feature_importance['lasso'] = dict(zip(
            feature_columns, np.abs(model.coef_)
        ))
        
        # 评估模型
        predictions = model.predict(X_test)
        self.evaluation_results['lasso'] = self._evaluate_model(y_test, predictions)
        
        return model
    
    def train_svr(self, features_df, target_col='销售数量', params=None):
        """
        训练支持向量回归模型
        """
        if params is None:
            params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            }
        
        # 准备特征
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_features_for_training(
            features_df, target_col
        )
        
        # 训练模型
        model = SVR(**params)
        model.fit(X_train, y_train)
        
        # 保存模型
        self.models['svr'] = model
        # SVR没有直接的特征重要性
        self.feature_importance['svr'] = {}
        
        # 评估模型
        predictions = model.predict(X_test)
        self.evaluation_results['svr'] = self._evaluate_model(y_test, predictions)
        
        return model
    
    def train_all_models(self, features_df, target_col='销售数量'):
        """
        训练所有支持的模型
        
        参数:
        - features_df: 特征数据框
        - target_col: 目标列名
        
        返回:
        - 所有训练好的模型字典
        """
        print("开始训练所有模型...")
        
        # 训练LightGBM
        print("训练LightGBM模型...")
        self.train_lightgbm(features_df, target_col)
        
        # 训练XGBoost
        print("训练XGBoost模型...")
        self.train_xgboost(features_df, target_col)
        
        # 训练随机森林
        print("训练随机森林模型...")
        self.train_random_forest(features_df, target_col)
        
        # 训练梯度提升
        print("训练梯度提升模型...")
        self.train_gradient_boosting(features_df, target_col)
        
        # 训练线性回归
        print("训练线性回归模型...")
        self.train_linear_regression(features_df, target_col)
        
        # 训练岭回归
        print("训练岭回归模型...")
        self.train_ridge(features_df, target_col)
        
        # 训练Lasso回归
        print("训练Lasso回归模型...")
        self.train_lasso(features_df, target_col)
        
        # 训练SVR
        print("训练支持向量回归模型...")
        self.train_svr(features_df, target_col)
        
        print("所有模型训练完成!")
        return self.models
    
    def predict(self, features_df, model_name='lightgbm'):
        """
        使用训练好的模型进行预测
        
        参数:
        - features_df: 特征数据框
        - model_name: 模型名称
        
        返回:
        - 包含预测结果的DataFrame
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 尚未训练")
        
        model = self.models[model_name]
        
        # 准备特征（与训练时相同的处理）
        numeric_features = self._get_numeric_features(features_df)
        categorical_features = self._get_categorical_features(features_df)
        
        feature_columns = [col for col in numeric_features + categorical_features 
                          if col not in ['商品编码', '商品名称', '日期', '销售数量']]
        
        X = features_df[feature_columns].copy()
        
        # 处理分类特征
        for col in categorical_features:
            if col in X.columns:
                # 处理可能的未知值
                X[col] = X[col].astype(str)
                # 如果遇到训练时未见过的类别，映射为未知类别
                unique_vals = set(self.label_encoders[col].classes_)
                X[col] = X[col].apply(lambda x: x if x in unique_vals else 'unknown')
                
                # 如果未知类别不在编码器中，添加它
                if 'unknown' not in self.label_encoders[col].classes_:
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 'unknown'
                    )
                
                X[col] = self.label_encoders[col].transform(X[col])
        
        # 处理无穷大值和过大的值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 限制数值范围
        for col in X.select_dtypes(include=[np.number]).columns:
            # 使用训练时的范围，这里暂时使用数据的分位数
            max_val = X[col].quantile(0.99)
            min_val = X[col].quantile(0.01)
            X[col] = X[col].clip(min_val, max_val)
        
        # 标准化
        X_scaled = self.scalers['standard_scaler'].transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 预测
        predictions = model.predict(X_scaled)
        
        # 创建结果DataFrame
        result_df = features_df[['商品编码', '日期']].copy() #'商品名称',
        result_df['预测销量'] = predictions
        if '销售数量' in features_df.columns:
            result_df['实际销量'] = features_df['销售数量']
        
        return result_df
    
    def _get_numeric_features(self, df):
        """获取数值型特征"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in ['商品编码']]
    
    def _get_categorical_features(self, df):
        """获取分类特征"""
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
        return [col for col in categorical_cols if col not in ['商品名称', '日期']]
    
    def _evaluate_model(self, y_true, y_pred):
        """评估模型性能"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # 计算MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def get_feature_importance(self, model_name='lightgbm', top_n=10):
        """
        获取特征重要性
        
        参数:
        - model_name: 模型名称
        - top_n: 返回前N个重要特征
        
        返回:
        - 特征重要性字典
        """
        if model_name not in self.feature_importance:
            return None
        
        importance_dict = self.feature_importance[model_name]
        if not importance_dict:
            return None
        
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance[:top_n])
    
    def get_evaluation_results(self, model_name='lightgbm'):
        """
        获取评估结果
        
        参数:
        - model_name: 模型名称
        
        返回:
        - 评估结果字典
        """
        return self.evaluation_results.get(model_name)
    
    def compare_models(self, metric='RMSE'):
        """
        比较所有训练过的模型
        
        参数:
        - metric: 比较的指标
        
        返回:
        - 模型比较结果DataFrame
        """
        comparison = {}
        for model_name in self.models.keys():
            if model_name in self.evaluation_results:
                comparison[model_name] = self.evaluation_results[model_name]
        
        # 转换为DataFrame
        comparison_df = pd.DataFrame.from_dict(comparison, orient='index')
        
        # 按指定指标排序
        if metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(by=metric)
        
        return comparison_df
    
    def plot_model_comparison(self, metric='RMSE', figsize=(10, 6)):
        """
        绘制模型比较图
        
        参数:
        - metric: 比较的指标
        - figsize: 图形大小
        """
        comparison_df = self.compare_models(metric)
        
        plt.figure(figsize=figsize)
        sns.barplot(x=comparison_df.index, y=comparison_df[metric])
        plt.title(f'模型比较 - {metric}')
        plt.ylabel(metric)
        plt.xlabel('模型')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model_name='lightgbm', top_n=10, figsize=(10, 6)):
        """
        绘制特征重要性图
        
        参数:
        - model_name: 模型名称
        - top_n: 显示前N个重要特征
        - figsize: 图形大小
        """
        importance = self.get_feature_importance(model_name, top_n)
        
        if importance is None:
            print(f"模型 {model_name} 没有特征重要性信息")
            return
        
        plt.figure(figsize=figsize)
        features = list(importance.keys())
        values = list(importance.values())
        
        sns.barplot(x=values, y=features)
        plt.title(f'特征重要性 - {model_name}')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name, filepath=None):
        """
        保存模型
        
        参数:
        - model_name: 模型名称
        - filepath: 保存路径，如果为None则使用默认路径
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 尚未训练")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        model_data = {
            'model': self.models[model_name],
            'feature_importance': self.feature_importance.get(model_name, {}),
            'evaluation_results': self.evaluation_results.get(model_name, {}),
            'feature_columns': self.feature_columns,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders
        }
        
        joblib.dump(model_data, filepath)
        print(f"模型 {model_name} 已保存到 {filepath}")
    
    def load_model(self, model_name, filepath=None):
        """
        加载模型
        
        参数:
        - model_name: 模型名称
        - filepath: 模型文件路径，如果为None则使用默认路径
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件 {filepath} 不存在")
        
        model_data = joblib.load(filepath)
        
        self.models[model_name] = model_data['model']
        self.feature_importance[model_name] = model_data.get('feature_importance', {})
        self.evaluation_results[model_name] = model_data.get('evaluation_results', {})
        self.feature_columns = model_data.get('feature_columns', {})
        self.scalers = model_data.get('scalers', {})
        self.label_encoders = model_data.get('label_encoders', {})
        
        print(f"模型 {model_name} 已从 {filepath} 加载")