# prediction_feature_builder.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings('ignore')


class PredictionFeatureBuilder:
    """
    预测特征构建器
    根据商品编码和预测日期，构建用于预测的特征数据
    """

    def __init__(self,
                 feature_store,
                 data_preprocessor,
                 sales_predictor,
                 historical_sales_data: pd.DataFrame,
                 calendar_data: pd.DataFrame,
                 weather_data: pd.DataFrame):
        """
        初始化预测特征构建器
        
        参数:
        - feature_store: 特征仓库实例
        - data_preprocessor: 数据预处理器实例
        - sales_predictor: 销售预测器实例
        - historical_sales_data: 历史销售数据
        - calendar_data: 日历数据
        - weather_data: 天气数据
        """
        self.feature_store = feature_store
        self.data_preprocessor = data_preprocessor
        self.sales_predictor = sales_predictor
        self.historical_sales_data = historical_sales_data
        self.calendar_data = calendar_data
        self.weather_data = weather_data
        self.feature_columns = {}
        self.label_encoders = {}
        self.scalers = {}
        # 缓存最后一天的数据用于滞后特征
        self.last_day_cache = {}

    def build_features_for_prediction(self,
                                      product_code: str,
                                      store_code: str,
                                      predict_date: Union[str, datetime],
                                      target_time_windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        为指定商品和日期构建预测特征
        
        参数:
        - product_code: 商品编码
        - store_code: 门店编码
        - predict_date: 预测日期 (格式: '2025-01-01' 或 datetime对象)
        - target_time_windows: 目标时间窗口列表 (1-48)，如果为None则预测所有48个窗口
        
        返回:
        - 包含预测特征的DataFrame
        """
        # 1. 将预测日期转换为datetime
        if isinstance(predict_date, str):
            predict_date = pd.to_datetime(predict_date)

        # 2. 确定需要预测的时间窗口
        if target_time_windows is None:
            target_time_windows = list(range(36, 44))  # 1-48

        # 3. 构建基础数据框架
        base_df = self._create_base_prediction_frame(
            product_code, store_code, predict_date, target_time_windows
        )

        # 4. 填充历史销售数据
        enriched_df = self._enrich_with_historical_data(base_df, product_code, store_code)

        # 5. 应用特征工程管道
        features_df = self._apply_feature_engineering_pipeline(enriched_df)

        return features_df

    def _create_base_prediction_frame(self,
                                      product_code: str,
                                      store_code: str,
                                      predict_date: datetime,
                                      time_windows: List[int]) -> pd.DataFrame:
        """
        创建基础预测数据框架
        """
        # 创建所有时间窗口的记录
        records = []
        for time_window in time_windows:
            records.append({
                '商品编码': product_code,
                '门店编码': store_code,
                '日期': predict_date,
                '时间窗口': time_window,
                # 预测时没有实际销售数据，用0或NaN填充
                '销售数量': np.nan,
                '平均售价': np.nan,
                '销售金额': np.nan,
                '售价': np.nan,
                '实际折扣率': 1.0,  # 默认不打折
                '是否促销': 0,  # 默认不促销
                '预测标志': 1  # 标记为预测数据
            })

        df = pd.DataFrame(records)

        # 添加日期特征
        df['星期'] = df['日期'].dt.dayofweek + 1  # 1=周一, 7=周日
        df['月份'] = df['日期'].dt.month
        df['年份'] = df['日期'].dt.year
        df['是否周末'] = (df['日期'].dt.dayofweek >= 5).astype(int) #df['星期'].isin([6, 7]).astype(int)
        df['是否月末'] = (df['日期'].dt.day >= 25).astype(int) #(df['日期'].dt.month != (df['日期'] + timedelta(days=1)).dt.month).astype(int)
        df['季度'] = df['日期'].dt.quarter#((df['日期'].dt.month - 1) // 3) + 1

        # 计算日期序号（从数据开始日期算起）
        if not self.historical_sales_data.empty:
            min_date = self.historical_sales_data['日期'].min()
            df['日期序号'] = (df['日期'] - min_date).dt.days + 1
        else:
            df['日期序号'] = 1

        # 季节特征 df['日期'].dt.month.apply(self._get_season)
        df['季节'] = df['日期'].dt.month.apply(self._get_season)

        return df

    def _get_season(self, month: int) -> int:
        """获取季节编码"""
        if month in [12, 1, 2]:
            return 1  # 冬季
        elif month in [3, 4, 5]:
            return 2  # 春季
        elif month in [6, 7, 8]:
            return 3  # 夏季
        else:
            return 4  # 秋季

    def _enrich_with_historical_data(self,
                                     base_df: pd.DataFrame,
                                     product_code: str,
                                     store_code: str) -> pd.DataFrame:
        """
        用历史数据丰富预测框架
        """
        df = base_df.copy()
        predict_date = df['日期'].iloc[0]

        # 1. 获取历史销售数据用于滞后特征
        historical_sales = self.historical_sales_data.copy()

        # 筛选指定商品和门店
        product_mask = historical_sales['商品编码'] == product_code
        store_mask = historical_sales['门店编码'] == store_code
        filtered_sales = historical_sales[product_mask & store_mask].copy()

        if not filtered_sales.empty:
            # 2. 计算滞后日期
            lag_dates = {
                1: predict_date - timedelta(days=1),
                3: predict_date - timedelta(days=3),
                7: predict_date - timedelta(days=7),
                14: predict_date - timedelta(days=14),
                30: predict_date - timedelta(days=30)
            }

            # 3. 为每个时间窗口计算滞后特征
            for time_window in df['时间窗口'].unique():
                window_mask = df['时间窗口'] == time_window

                # 计算各种滞后特征
                for lag_days, lag_date in lag_dates.items():
                    col_name = f'销量_滞后{lag_days}天'

                    # 查找滞后日期的销售数据
                    lag_sales = filtered_sales[
                        (filtered_sales['日期'] == lag_date) &
                        (filtered_sales['时间窗口'] == time_window)
                        ]

                    if not lag_sales.empty:
                        lag_value = lag_sales['销售数量'].mean()
                    else:
                        # 如果没有对应数据，尝试用相近数据填充
                        # 查找前几天的相同时间窗口
                        similar_sales = filtered_sales[
                            (filtered_sales['日期'] >= lag_date - timedelta(days=3)) &
                            (filtered_sales['日期'] <= lag_date) &
                            (filtered_sales['时间窗口'] == time_window)
                            ]

                        if not similar_sales.empty:
                            lag_value = similar_sales['销售数量'].mean()
                        else:
                            # 使用该时间窗口的历史平均值
                            window_history = filtered_sales[
                                filtered_sales['时间窗口'] == time_window
                                ]
                            lag_value = window_history['销售数量'].mean() if not window_history.empty else 0

                    df.loc[window_mask, col_name] = lag_value

            # 4. 计算滚动特征
            for time_window in df['时间窗口'].unique():
                window_mask = df['时间窗口'] == time_window

                # 获取该时间窗口的历史数据
                window_history = filtered_sales[filtered_sales['时间窗口'] == time_window]

                if len(window_history) >= 7:
                    last_7_days = window_history[
                        window_history['日期'] >= predict_date - timedelta(days=7)
                        ]

                    if len(last_7_days) > 0:
                        df.loc[window_mask, '销量_7天均值'] = last_7_days['销售数量'].mean()
                        df.loc[window_mask, '销量_7天标准差'] = last_7_days['销售数量'].std()

                        # 计算趋势（线性回归斜率）
                        if len(last_7_days) >= 2:
                            dates_numeric = (last_7_days['日期'] - last_7_days['日期'].min()).dt.days
                            slope = np.polyfit(dates_numeric, last_7_days['销售数量'], 1)[0]
                            df.loc[window_mask, '销量_7天趋势'] = slope

                if len(window_history) >= 30:
                    last_30_days = window_history[
                        window_history['日期'] >= predict_date - timedelta(days=30)
                        ]

                    if len(last_30_days) > 0:
                        df.loc[window_mask, '销量_30天均值'] = last_30_days['销售数量'].mean()
                        df.loc[window_mask, '销量_30天标准差'] = last_30_days['销售数量'].std()

                # 计算同比上周（同星期几）
                last_week_same_day = window_history[
                    (window_history['日期'] == predict_date - timedelta(days=7)) &
                    (window_history['星期'] == df.loc[window_mask, '星期'].iloc[0])
                    ]

                if not last_week_same_day.empty:
                    df.loc[window_mask, '销量_同比上周'] = last_week_same_day['销售数量'].values[0]

        # 5. 填充缺失的滞后和滚动特征
        lag_cols = [col for col in df.columns if '滞后' in col or '均值' in col or '标准差' in col or '趋势' in col]
        for col in lag_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def _apply_feature_engineering_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征工程管道（与训练时保持一致）
        """
        # 1. 时间特征已经在_base_prediction_frame中创建
        # 2. 天气特征
        weather_processed = self.data_preprocessor.preprocess_weather_data(self.weather_data)
        df = self.feature_store.create_weather_enhanced_features(df, weather_processed)

        # 3. 日历特征
        calendar_processed = self.data_preprocessor.preprocess_calendar_data(self.calendar_data)
        df = self.feature_store.create_calendar_enhanced_features(df, calendar_processed)

        # 4. 交互特征
        df = self.feature_store.create_interaction_features(df)

        # 5. 确保特征列与训练时一致
        df = self._ensure_feature_consistency(df)

        return df

    def _ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        确保特征列与训练时一致
        """
        # 获取训练时使用的特征列（从训练好的模型中）
        # 这里假设我们已经保存了训练时的特征列信息

        # 示例：从模型配置中获取特征列
        expected_features = [
            '时间窗口', '平均售价', '销售金额', '售价', '实际折扣率', '是否促销',
            '星期', '月份', '是否周末', '是否月末', '季度', '年份', '日期序号',
            '销量_滞后1天', '销量_滞后3天', '销量_滞后7天', '销量_滞后14天',
            '销量_滞后30天', '销量_7天均值', '销量_7天标准差', '销量_30天均值',
            '销量_30天标准差', '销量_7天趋势', '销量_同比上周', '温度', '最高温度',
            '最低温度', '降雨量', '天气严重程度', '温度差', '是否恶劣天气',
            '是否节假日', '节假日前1天', '节假日后1天', '节假日连续天数',
            '季节', '温度等级', '温度差等级', '节假日类型'
        ]

        # 添加缺失的特征列
        for feature in expected_features:
            if feature not in df.columns:
                if '滞后' in feature or '均值' in feature or '标准差' in feature or '趋势' in feature:
                    df[feature] = 0
                elif '温度' in feature:
                    df[feature] = 20  # 默认温度
                elif '等级' in feature:
                    df[feature] = '中等'
                elif '类型' in feature:
                    df[feature] = '普通日'
                else:
                    df[feature] = 0

        # 移除多余的特征列
        columns_to_keep = expected_features + ['商品编码', '日期', '预测标志']
        df = df[[col for col in columns_to_keep if col in df.columns]]

        return df

    def get_prediction_input(self,
                             product_code: str,
                             store_code: str,
                             predict_date: Union[str, datetime],
                             time_windows: Optional[List[int]] = None,
                             model_name: str = 'lightgbm') -> Dict[str, Any]:
        """
        获取完整的预测输入数据
        
        返回:
        - 包含特征数据和元数据的字典
        """
        # 构建特征
        features_df = self.build_features_for_prediction(
            product_code, store_code, predict_date, time_windows
        )

        # 准备模型输入格式
        # X_pred = self._prepare_for_model_prediction(features_df, model_name)
        X_pred = self.prepare_features_for_training(features_df)

        result = {
            'product_code': product_code,
            'store_code': store_code,
            'predict_date': predict_date,
            'features_df': features_df,
            'X_pred': X_pred,
            'time_windows': features_df['时间窗口'].tolist(),
            'feature_columns': X_pred.columns.tolist() if hasattr(X_pred, 'columns') else []
        }

        return result

    # def _prepare_for_model_prediction(self, features_df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     准备模型预测输入格式
    #     """
    #     # 复制数据框，避免修改原始数据
    #     df = features_df.copy()
    #
    #     # 移除不需要的列
    #     columns_to_drop = ['商品编码', '日期', '预测标志']
    #     if '销售数量' in df.columns:
    #         columns_to_drop.append('销售数量')
    #
    #     X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    #
    #     # 处理分类特征（使用与训练时相同的编码器）
    #     categorical_features = self._get_categorical_features(X)
    #
    #     for col in categorical_features:
    #         if col in X.columns:
    #             # 这里应该使用训练时保存的LabelEncoder
    #             # 暂时使用简单的编码方式
    #             if X[col].dtype == 'object' or X[col].dtype.name == 'category':
    #                 X[col] = X[col].astype('category').cat.codes
    #
    #     # 处理缺失值
    #     X = X.fillna(0)
    #
    #     # 限制数值范围
    #     for col in X.select_dtypes(include=[np.number]).columns:
    #         X[col] = X[col].clip(-1e6, 1e6)
    #
    #     return X
    def _get_numeric_features(self, df):
        """获取数值型特征"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in ['商品编码']]
    def _get_categorical_features(self, df: pd.DataFrame) -> List[str]:
        """获取分类特征列"""
        categorical_cols = df.select_dtypes(
            include=['category', 'object']
        ).columns.tolist()

        # 排除数值型的分类列
        categorical_cols = [col for col in categorical_cols
                            if col not in ['日期', '商品编码', '商品名称']]

        return categorical_cols

    def prepare_features_for_training(self, features_df):
        """
        准备训练特征

        参数:
        - features_df: 特征数据框
        返回:
        - X_train, X_test, y_train, y_test, feature_columns
        """
        # 复制数据框，避免修改原始数据
        # df = features_df.copy()

        # # 移除不需要的列
        # columns_to_drop = ['商品编码', '日期', '预测标志']
        # if '销售数量' in df.columns:
        #     columns_to_drop.append('销售数量')
        #
        # X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # 选择数值型特征和分类特征
        numeric_features = self._get_numeric_features(features_df)
        categorical_features = self._get_categorical_features(features_df)

        # 移除目标变量和标识列
        feature_columns = [col for col in numeric_features + categorical_features
                           if col not in ['门店编码', '商品编码', '商品名称', '日期', '预测标志']]

        # 准备数据
        X = features_df[feature_columns].copy()

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


        # 填充NaN值
        X = X.fillna(0)

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
        if '时间窗口' in features_df.columns:
            # 按日期排序整个数据集
            sorted_indices = features_df['时间窗口'].sort_values().index
            X_scaled = X_scaled.loc[sorted_indices]


        # 保存特征列信息
        self.feature_columns['all'] = feature_columns
        self.feature_columns['numeric'] = numeric_features
        self.feature_columns['categorical'] = categorical_features

        return X_scaled, feature_columns