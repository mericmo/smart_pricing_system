# 导入必要的库
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import logging  # 日志记录库
import datetime  # 日期时间处理库
import lightgbm as lgb  # LightGBM机器学习库
import random  # 随机数生成库
import os  # 操作系统接口库
import psutil  # 系统和进程信息库
import argparse  # 命令行参数解析库


def get_logger(log_name, log_dir_path, log_level=logging.DEBUG):
    """
    创建并配置日志记录器
    
    参数:
        log_name: 日志名称
        log_dir_path: 日志目录路径
        log_level: 日志级别，默认为DEBUG
    
    返回:
        配置好的logger对象
    """
    # 创建一个logger对象
    logger = logging.getLogger(log_name)
    # 创建一个输出到控制台的handler
    stream_handler = logging.StreamHandler()
    # 设置输出格式
    formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
    # 将格式应用到handler
    stream_handler.setFormatter(formatter)
    # 将handler添加到logger对象中
    logger.addHandler(stream_handler)

    # 设置日志文件路径
    log_file_path = str(log_dir_path / (log_name + datetime.datetime.today().strftime('_%Y_%m%d.log')))
    # 创建一个输出到文件的handler
    file_handler = logging.FileHandler(filename=log_file_path, encoding='utf-8')
    # 将格式应用到handler
    file_handler.setFormatter(formatter)
    # 将handler添加到logger对象中
    logger.addHandler(file_handler)

    # 设置logger对象的日志级别
    logger.setLevel(log_level)
    return logger


class Log(object):
    """
    日志记录封装类，提供便捷的日志记录方法
    """
    def __init__(self, logger):
        self.logger = logger

    def info(self, *messages):
        # 将消息记录到日志中，级别为INFO
        return self.logger.info(Log.format_message(messages))

    def debug(self, *messages):
        # 将消息记录到日志中，级别为DEBUG
        return self.logger.debug(Log.format_message(messages))

    def warning(self, *messages):
        # 将消息记录到日志中，级别为WARNING
        return self.logger.warning(Log.format_message(messages))

    def error(self, *messages):
        # 将消息记录到日志中，级别为ERROR
        return self.logger.error(Log.format_message(messages))

    def exception(self, *messages):
        # 将消息记录到日志中，级别为ERROR，并且记录异常信息
        return self.logger.exception(Log.format_message(messages))

    @staticmethod
    def format_message(messages):
        # 将多个消息拼接成一个字符串
        if len(messages) == 1 and isinstance(messages[0], list):
            messages = tuple(messages[0])
        return '\t'.join(map(str, messages))

    def log_evaluation(self, period=100, show_stdv=True, level=logging.INFO):
        # 定义一个回调函数，用于记录模型评估结果
        def _callback(env):
            if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
                result = '\t'.join(
                    [lgb.callback._format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
                self.logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))

        _callback.order = 10
        return _callback


class Util(object):
    """
    工具类，提供常用的辅助方法
    """
    @staticmethod
    def set_seed(seed):
        """
        设置随机种子，确保实验可重复性
        
        参数:
            seed: 随机种子值
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        return

    @staticmethod
    def get_memory_usage():
        """
        获取当前进程的内存使用情况
        
        返回:
            内存使用量(GB)
        """
        return np.round(psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30, 2)

    @staticmethod
    def reduce_mem_usage(df, verbose=False):
        """
        优化DataFrame的内存使用，通过降低数值类型的数据类型
        
        参数:
            df: 需要优化的DataFrame
            verbose: 是否打印优化信息
            
        返回:
            优化后的DataFrame
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
        return df

    @staticmethod
    def merge_by_concat(df1, df2, merge_on):
        """
        通过连接方式合并两个DataFrame，避免内存使用过多
        
        参数:
            df1: 第一个DataFrame
            df2: 第二个DataFrame
            merge_on: 合并的键列名
            
        返回:
            合并后的DataFrame
        """
        merged_gf = df1[merge_on]
        merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
        new_columns = [col for col in list(merged_gf) if col not in merge_on]
        df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
        return df1


class Params(object):
    """
    参数管理类，负责管理项目中的各种参数和路径配置
    """
    def __init__(self, setting):
        """
        初始化参数配置
        
        参数:
            setting: 包含配置信息的字典
        """
        self.setting = setting
        self.data_dir_path = Path(setting['data_dir_path'])  # 数据根目录路径

        # 原始数据目录
        self.raw_dir_path = self.data_dir_path / 'data'
        self.raw_dir_path.mkdir(parents=True, exist_ok=True)

        # 输出目录
        self.output_name = Path(setting['output_name'])
        self.output_dir_path = self.data_dir_path / 'output' / self.output_name
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

        # 日志配置
        self.log_name = setting['log_name']
        self.log_dir_path = self.output_dir_path / 'log'
        self.log_dir_path.mkdir(parents=True, exist_ok=True)
        self.log = Log(get_logger(self.log_name, self.log_dir_path))

        # 结果目录
        self.result_dir_path = self.output_dir_path / 'result'
        self.result_dir_path.mkdir(parents=True, exist_ok=True)

        # 工作目录
        self.work_dir_path = self.output_dir_path / 'work'
        self.work_dir_path.mkdir(parents=True, exist_ok=True)

        # 模型目录
        self.model_dir_path = self.output_dir_path / 'model'
        self.model_dir_path.mkdir(parents=True, exist_ok=True)

        # 随机种子设置
        self.seed = 42
        Util.set_seed(self.seed)

        # 数据采样率
        self.sampling_rate = setting['sampling_rate']
        self.export_all_flag = False  # 是否导出所有结果
        self.recursive_feature_flag = False  # 是否使用递归特征

        # 目标变量
        self.target = 'sales'
        self.start_train_day_x = 1  # 训练开始日期


        # 原始数据文件路径
        self.raw_sales_path = self.raw_dir_path / 'historical_transactions.csv'#'1月1日-10月31日-56个品.csv'
        self.raw_weather_path = self.raw_dir_path / 'weather_info.csv'
        self.raw_calendar_path = self.raw_dir_path / 'calender_info.csv'

        # 当前预测的产品信息
        self.product_code = setting['product_code']
        self.store_code = setting['store_code']
        return


    def reset_dir_path(self):
        """
        重置目录路径到默认值
        """
        self.result_dir_path = self.output_dir_path / 'result'
        self.result_dir_path.mkdir(parents=True, exist_ok=True)

        self.work_dir_path = self.output_dir_path / 'work'
        self.work_dir_path.mkdir(parents=True, exist_ok=True)

        self.model_dir_path = self.output_dir_path / 'work'
        self.model_dir_path.mkdir(parents=True, exist_ok=True)
        return



# 项目键值
project_key = 'hr'
# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='{}_main'.format(project_key))
from pathlib import Path

# 添加命令行参数
parser.add_argument('-ddp', '--data_dir_path', type=str, default='.')  # 数据目录路径
parser.add_argument('-opn', '--output_name', type=str, default='default')  # 输出名称
# parser.add_argument('-spr', '--sampling_rate', type=float, default=0.02)  # 采样率
parser.add_argument('-spr', '--sampling_rate', type=float, default=1)  # 备用采样率
# parser.add_argument('-plc', '--prediction_horizon_list_csv', type=str, default='7,14,21,28')  # 预测时间范围列表
parser.add_argument('-plc', '--prediction_horizon_list_csv', type=str, default='7')  # 预测时间范围列表
parser.add_argument('-sc', '--store_code', type=str, default='205625')  # 预测门店信息
parser.add_argument('-pc', '--product_code', type=str, default='8006144')  # 预测产品编码
# 解析命令行参数
args = parser.parse_args()
# 创建参数对象
params = Params(
    {
        'data_dir_path': args.data_dir_path,
        'output_name': args.output_name,
        'log_name': '{}_main'.format(project_key),
        'sampling_rate': args.sampling_rate,
        'prediction_horizon_list_csv': args.prediction_horizon_list_csv,
        'product_code': args.product_code,
        'store_code': args.store_code,
    })