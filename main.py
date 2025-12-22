# 导入必要的库
import shutil  # 文件操作
import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import datetime  # 日期时间处理

# 导入特征仓库
from feature_store import FeatureStore
# 数据统一处理
from data_preprocessor import DataPreprocessor
# 导入算法
from algorithm import SalesPredictor
# 导入可视化管理器
from visualization_manager import VisualizationManager

from discount_optimizer import DiscountOptimizer

from prediction_manager import PredictionManager

from typing import Dict, List, Optional, Union, Any


class HRMain(object):
    """
    负责数据处理、特征工程、模型训练和预测
    """

    def __init__(self, params):
        """
        初始化HRMain类
        :param params: 参数对象，包含所有配置参数
        """
        self.params = params
        self.product_code = params.product_code
        self.store_code = params.store_code
        if self.product_code is None or self.store_code is None:
            print(f"参数缺失：product_code:{self.product_code},store_code:{self.store_code}")
            raise ValueError("参数缺失")

        self.log = self.params.log
        self.data_preprocessor = DataPreprocessor(self.product_code, self.store_code)
        # 初始化特征仓库
        self.feature_store = FeatureStore()
        # 初始化算法
        self.sales_predictor = SalesPredictor(forecast_horizon=7)

        self.best_model_name = ''
        # 初始化可视化管理器
        self.visualization_manager = VisualizationManager(log=self.log, result_dir_path=self.params.result_dir_path)
        return

    def load_data(self):

        self.log.info('load_data')

        # 加载销售数据
        hsr_df = pd.read_csv(self.params.raw_sales_path, encoding='utf-8', parse_dates=['日期', '交易时间'],
                             dtype={"商品编码": str, "门店编码": str, "流水单号": str, "会员id": str})
        # 用Pandas快速检查数据
        print("HuarunStore数据基本信息：")
        print(f"数据行数：{len(hsr_df)}")
        print(f"数据列数：{len(hsr_df.columns)}")
        print(f"时间范围：{hsr_df['日期'].min()} 到 {hsr_df['日期'].max()}")
        print(f"商品种类数：{hsr_df['商品编码'].nunique()}")
        print(f"商品小类数：{hsr_df['小类编码'].nunique()}")
        print(f"统计会员数：{hsr_df['会员id'].nunique()}")
        print(f"统计非会员数(可能有重复)：{hsr_df['会员id'].isnull().sum()}")

        # 加载日历数据
        calendar_df = pd.read_csv(self.params.raw_calendar_path)

        # 加载天气
        weather_df = pd.read_csv(self.params.raw_weather_path)

        # 进行数据检测
        self.log.info("销售数据缺失值统计：")
        self.log.info(hsr_df.isnull().sum())
        self.log.info("日历数据缺失值统计：")
        self.log.info(calendar_df.isnull().sum())
        self.log.info("天气数据缺失值统计：")
        self.log.info(weather_df.isnull().sum())

        return hsr_df, calendar_df, weather_df

    def create_features_pipeline(self):
        """特征创建流水线"""
        self.log.info('开始特征工程...')

        # 1. 基础预处理
        sales_processed = self.data_preprocessor.preprocess_sales_data(self.hsr_df)
        weather_processed = self.data_preprocessor.preprocess_weather_data(self.weather_df)
        calendar_processed = self.data_preprocessor.preprocess_calendar_data(self.calendar_df)

        # 2. 时间特征
        sales_with_time = self.feature_store.create_time_features(sales_processed)

        # 3. 滞后特征（需要按商品单独计算）
        sales_with_lag = self.feature_store.create_lag_features(sales_with_time)

        # 4. 滚动特征（需要按商品单独计算）
        sales_with_rolling = self.feature_store.create_rolling_features(sales_with_lag)

        # 5. 趋势特征
        sales_with_trend = self.feature_store.create_trend_features(sales_with_rolling)

        # 6. 合并天气特征
        sales_with_weather = self.feature_store.create_weather_enhanced_features(
            sales_with_trend, weather_processed
        )

        # 7. 合并日历特征
        sales_with_calendar = self.feature_store.create_calendar_enhanced_features(
            sales_with_weather, calendar_processed
        )

        # 8. 交互特征
        final_features = self.feature_store.create_interaction_features(sales_with_calendar)

        self.log.info(f'特征工程完成，生成特征数量: {len(final_features.columns)}')
        self.log.info(f'特征列表: {list(final_features.columns)}')

        return final_features

    def train_models(self, features_df):
        """训练多个模型"""
        self.log.info('开始训练模型...')

        # 使用train_all_models方法训练所有模型
        trained_models = self.sales_predictor.train_all_models(features_df)

        # 获取模型比较结果
        comparison_df = self.sales_predictor.compare_models()
        self.log.info('模型性能比较:')
        for model_name, metrics in comparison_df.iterrows():
            self.log.info(f'{model_name}: {metrics.to_dict()}')

        # 使用可视化管理器创建模型比较可视化
        self.visualization_manager.create_model_comparison_visualization(comparison_df)

        # 获取最佳模型（基于RMSE指标）
        best_model_name = comparison_df.index[0]  # 第一个是最佳模型
        self.log.info(f'最佳模型: {best_model_name}')

        # 获取特征重要性（仅适用于基于树的模型）
        tree_models = ['lightgbm', 'xgboost', 'random_forest', 'gradient_boosting']
        for model_name in tree_models:
            if model_name in trained_models:
                try:
                    importance = self.sales_predictor.get_feature_importance(model_name, top_n=10)
                    if importance:
                        self.log.info(f'{model_name}特征重要性Top10:')
                        for feature, imp in importance.items():
                            self.log.info(f'  {feature}: {imp}')
                except Exception as e:
                    self.log.error(f'获取{model_name}特征重要性失败: {str(e)}')

        # 保存最佳模型
        if best_model_name in trained_models:
            model_path = self.params.model_dir_path / f'best_model_{best_model_name}.pkl'
            self.sales_predictor.save_model(best_model_name, str(model_path))
            self.log.info(f'最佳模型已保存到: {model_path}')

        return comparison_df, best_model_name, trained_models

    def make_predictions(self, features_df, model_name=None, model=None):
        """进行预测（仅在测试集上）"""
        self.log.info('开始预测...')

        # 如果没有指定模型，使用训练好的最佳模型
        if model is None:
            # 尝试加载已保存的最佳模型
            model_files = list(self.params.model_dir_path.glob('best_model_*.pkl'))
            if model_files:
                model_path = model_files[0]
                best_model_name = model_path.stem.replace('best_model_', '')
                self.log.info(f'加载已保存的模型: {best_model_name}')

                # 使用SalesPredictor的predict方法进行预测
                try:
                    # 关键修改：只对测试集进行预测，而不是整个数据集
                    # 首先需要获取测试集数据
                    X_train, X_test, y_train, y_test, feature_columns = self.sales_predictor.prepare_features_for_training(
                        features_df, target_col='销售数量'
                    )
                    X_test.to_csv("X_test.csv", encoding='utf-8')
                    # 创建测试集的特征DataFrame（包含原始列信息）
                    test_indices = X_test.index
                    test_features_df = features_df.loc[test_indices].copy()

                    # 只对测试集进行预测
                    predictions = self.sales_predictor.predict(test_features_df, model_name=best_model_name)

                    # 保存预测结果
                    predictions_path = self.params.result_dir_path / f'predictions_{best_model_name}_test_set.csv'
                    predictions.to_csv(predictions_path, index=False, encoding='utf-8')
                    self.log.info(f'测试集预测结果已保存到: {predictions_path}')

                    # 计算预测误差统计（测试集上的真实性能）
                    if '实际销量' in predictions.columns:
                        actual = predictions['实际销量']
                        predicted = predictions['预测销量']
                        error = actual - predicted

                        mae = np.mean(np.abs(error))
                        mse = np.mean(error ** 2)
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs(error / np.where(actual != 0, actual, 1))) * 100

                        self.log.info(f'测试集预测误差统计:')
                        self.log.info(f'  MAE: {mae:.4f}')
                        self.log.info(f'  MSE: {mse:.4f}')
                        self.log.info(f'  RMSE: {rmse:.4f}')
                        self.log.info(f'  MAPE: {mape:.4f}%')
                        self.log.info(f'  测试集样本数量: {len(predictions)}')

                    return predictions
                except Exception as e:
                    self.log.error(f'预测失败1: {str(e)}')
                    return None
            else:
                self.log.error('没有可用的模型进行预测')
                return None
        else:
            # 如果提供了模型，使用提供的模型进行预测
            try:
                # 同样只对测试集进行预测
                X_train, X_test, y_train, y_test, feature_columns = self.sales_predictor.prepare_features_for_training(
                    features_df, target_col='销售数量'
                )
                # X_test.to_csv("X_test.csv", encoding='utf-8')
                test_indices = X_test.index
                test_features_df = features_df.loc[test_indices].copy()

                predictions = self.sales_predictor.predict(test_features_df, model_name=model_name)

                # 保存预测结果
                predictions_path = self.params.result_dir_path / f'predictions_{model_name}_test_set.csv'
                predictions.to_csv(predictions_path, index=False, encoding='utf-8')
                self.log.info(f'测试集预测结果已保存到: {predictions_path}')

                return predictions
            except Exception as e:
                self.log.error(f'预测失败: {str(e)}')
                return None

    def run_discount_optimization(self, input_params):
        """
        运行折扣优化算法

        参数:
        - input_params: 输入参数字典

        返回:
        - 折扣方案结果
        """
        self.log.info('开始折扣方案优化...')

        # 初始化折扣优化器
        discount_optimizer = DiscountOptimizer(
            model_predictor=self.sales_predictor,
            feature_store=self.feature_store,
            data_preprocessor=self.data_preprocessor
        )

        try:
            # 生成折扣方案
            discount_plan = discount_optimizer.generate_discount_plan(input_params)

            self.log.info(f'折扣方案生成成功，商品: {input_params.get("product_code")}')
            self.log.info(f'预期清空率: {discount_plan.get("plan_metrics", {}).get("clearance_rate", 0):.1%}')
            self.log.info(f'预期总利润: {discount_plan.get("plan_metrics", {}).get("total_expected_profit", 0):.2f}')

            # 可视化折扣方案
            self.visualization_manager.visualize_discount_plan(discount_plan)

            # 保存方案结果
            result_path = self.params.result_dir_path / f'discount_plan_{input_params.get("product_code")}.json'
            import json
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(discount_plan, f, ensure_ascii=False, indent=2)

            self.log.info(f'折扣方案已保存到: {result_path}')

            return discount_plan

        except Exception as e:
            self.log.error(f'折扣方案优化失败: {str(e)}')
            return None

    def batch_discount_optimization(self, products_params):
        """
        批量折扣优化

        参数:
        - products_params: 商品参数列表

        返回:
        - 批量优化结果
        """
        self.log.info(f'开始批量折扣优化，共{len(products_params)}个商品')

        results = []
        for params in products_params:
            try:
                result = self.run_discount_optimization(params)
                results.append({
                    'product_code': params.get('product_code'),
                    'success': True if result else False,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'product_code': params.get('product_code'),
                    'success': False,
                    'error': str(e)
                })

        # 生成批量报告
        self._generate_batch_report(results)

        return results

    def _generate_batch_report(self, results):
        """生成批量优化报告"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        self.log.info(f'批量优化完成: 成功{len(successful)}个, 失败{len(failed)}个')

        if successful:
            # 计算总体指标
            total_expected_profit = sum(
                r['result'].get('plan_metrics', {}).get('total_expected_profit', 0)
                for r in successful
            )
            avg_clearance_rate = np.mean([
                r['result'].get('plan_metrics', {}).get('clearance_rate', 0)
                for r in successful
            ])

            self.log.info(f'预期总利润: {total_expected_profit:.2f}')
            self.log.info(f'平均清空率: {avg_clearance_rate:.1%}')

        # 保存报告
        report_data = {
            'summary': {
                'total_products': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(results) if results else 0
            },
            'details': results
        }

        report_path = self.params.result_dir_path / 'batch_discount_optimization_report.json'
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        self.log.info(f'批量优化报告已保存到: {report_path}')

    def predict(self, product_code: str, store_code: str,
                predict_date: datetime, time_windows: Optional[List[int]] = None, model_name: str = None):
        """
        HRMain的预测方法

        参数:
        - product_code: 商品编码
        - store_code: 门店编码
        - predict_date: 预测日期
        - model_name: 模型名称

        返回:
        - 预测结果DataFrame
        """
        # 处理日期参数
        self.log.info(f"开始预测: 商品={product_code}, 门店={store_code}, 日期={predict_date}")

        # 初始化预测管理器
        prediction_manager = PredictionManager(self)
        prediction_manager.initialize_prediction_builder()

        # 执行预测
        result = prediction_manager.predict_sales(
            product_code, store_code, predict_date, time_windows, model_name
        )

        # 保存预测结果
        if not result.empty:
            result_path = self.params.result_dir_path / f'prediction_{product_code}_{predict_date}.csv'.replace(":", '')
            result.to_csv(result_path, index=False, encoding='utf-8')
            self.log.info(f"预测结果已保存到: {result_path}")

        return result

    def main(self):
        """
        主函数，执行整个预测流程
        :return: 无
        """
        # 输出日志信息
        self.log.info('main')

        # 加载数据
        self.hsr_df, self.calendar_df, self.weather_df = self.load_data()

        # 特征工程
        self.features_df = self.create_features_pipeline()

        # 训练模型
        model_comparison, best_model_name, trained_models = self.train_models(self.features_df)

        # 进行预测
        if best_model_name in trained_models:
            self.best_model_name = best_model_name
            predictions = self.make_predictions(self.features_df,
                                                model_name=best_model_name,
                                                model=trained_models[best_model_name])

        # 使用可视化管理器可视化预测结果
        self.visualization_manager.visualize_results(predictions)

        # 重置目录路径
        self.params.reset_dir_path()
        try:
            # 输出日志信息，尝试清除工作目录
            self.log.info('clear work_dir')
            shutil.rmtree(self.params.work_dir_path)
        except Exception:
            # 输出异常信息
            self.log.exception()
        return


if __name__ == '__main__':
    # 导入参数
    from param import *

    # 实例化HRMain类
    m = HRMain(params)
    # 打印日志
    m.log.info('******** start')
    m.log.info(parser.parse_args())
    # 记录开始时间
    m.start_dt = datetime.datetime.now()

    # 打印参数设置
    m.log.info(m.params.setting)
    # 运行主函数
    m.main()

    # 进行预测
    result = m.predict(
        product_code=params.product_code,
        store_code=params.store_code,
        time_windows=list(range(36, 44)),
        predict_date=pd.to_datetime("2025-11-01"),
        model_name=m.best_model_name
    )

    print(result.head())

    # 打印结束时间和运行时间
    m.log.info(
        ['******** end', 'start_time', m.start_dt, 'process_time', datetime.datetime.now() - m.start_dt])
