import pandas as pd
from prediction_feature_builder import PredictionFeatureBuilder
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
class PredictionManager:
    """
    预测管理器 - 整合预测流程
    """

    def __init__(self, hrmain_instance):
        """
        初始化预测管理器

        参数:
        - hrmain_instance: HRMain类的实例
        """
        self.hrmain = hrmain_instance
        self.prediction_builder = None

    def initialize_prediction_builder(self):
        """初始化预测特征构建器"""
        # 使用HRMain中已加载的数据
        self.prediction_builder = PredictionFeatureBuilder(
            feature_store=self.hrmain.feature_store,
            data_preprocessor=self.hrmain.data_preprocessor,
            sales_predictor=self.hrmain.sales_predictor,
            historical_sales_data=self.hrmain.features_df,
            calendar_data=self.hrmain.calendar_df,
            weather_data=self.hrmain.weather_df
        )

    def predict_sales(self,
                      product_code: str,
                      store_code: str,
                      predict_date: Union[str, datetime],
                      time_windows: Optional[List[int]] = None,
                      model_name: str = None) -> pd.DataFrame:
        """
        预测指定商品在指定日期的销量

        参数:
        - product_code: 商品编码
        - store_code: 门店编码
        - predict_date: 预测日期
        - model_name: 模型名称，如果为None则使用最佳模型

        返回:
        - 包含预测结果的DataFrame
        """
        if self.prediction_builder is None:
            self.initialize_prediction_builder()

        # 1. 获取预测输入数据
        prediction_input = self.prediction_builder.get_prediction_input(
            product_code, store_code, predict_date, time_windows
        )

        # 2. 使用模型进行预测
        if model_name is None:
            # 查找已保存的最佳模型
            model_files = list(self.hrmain.params.model_dir_path.glob('best_model_*.pkl'))
            if model_files:
                model_path = model_files[0]
                model_name = model_path.stem.replace('best_model_', '')

        if model_name in self.hrmain.sales_predictor.models:
            # 使用已加载的模型
            model = self.hrmain.sales_predictor.models[model_name]
        else:
            # 尝试加载模型
            model_path = self.hrmain.params.model_dir_path / f'best_model_{model_name}.pkl'
            if model_path.exists():
                self.hrmain.sales_predictor.load_model(model_name, str(model_path))
                model = self.hrmain.sales_predictor.models.get(model_name)
            else:
                raise ValueError(f"模型 {model_name} 不存在")

        # 3. 准备特征数据
        X_pred, feature_columns = prediction_input['X_pred']

        # 确保特征列与训练时一致
        if hasattr(self.hrmain.sales_predictor, 'feature_columns'):
            expected_columns = self.hrmain.sales_predictor.feature_columns.get('all', [])
            missing_cols = set(expected_columns) - set(X_pred.columns)
            # extra_cols = set(X_pred.columns) - set(expected_columns)

            # 添加缺失列
            for col in missing_cols:
                X_pred[col] = 0

            # 移除多余列
            X_pred = X_pred[expected_columns]

        X_pred.to_csv("X_pred.csv", encoding='utf-8')
        # 4. 进行预测
        try:
            predictions = model.predict(X_pred)

            # 5. 创建结果DataFrame
            result_df = pd.DataFrame({
                '商品编码': product_code,
                '门店编码': store_code,
                '预测日期': predict_date,
                '时间窗口': prediction_input['time_windows'],
                '预测销量': predictions,
                '预测时间': datetime.now()
            })

            # 添加特征信息（可选）
            for i, feature in enumerate(prediction_input['feature_columns'][:5]):  # 只显示前5个特征
                if feature in X_pred.columns:
                    result_df[f'特征_{feature}'] = X_pred[feature].values

            return result_df

        except Exception as e:
            self.hrmain.log.error(f"预测失败: {str(e)}")
            raise

    def predict_multiple_days(self,
                              product_code: str,
                              store_code: str,
                              start_date: Union[str, datetime],
                              end_date: Union[str, datetime],
                              model_name: str = None) -> pd.DataFrame:
        """
        预测多天的销量

        参数:
        - product_code: 商品编码
        - store_code: 门店编码
        - start_date: 开始日期
        - end_date: 结束日期
        - model_name: 模型名称

        返回:
        - 包含多天预测结果的DataFrame
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        all_predictions = []

        current_date = start_date
        while current_date <= end_date:
            try:
                daily_predictions = self.predict_sales(
                    product_code, store_code, current_date, model_name
                )
                all_predictions.append(daily_predictions)
                self.hrmain.log.info(f"已完成 {current_date} 的预测")
            except Exception as e:
                self.hrmain.log.error(f"{current_date} 预测失败: {str(e)}")

            current_date += timedelta(days=1)

        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        else:
            return pd.DataFrame()
