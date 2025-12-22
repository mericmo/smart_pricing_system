import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VisualizationManager:
    """可视化管理器，负责所有预测结果的可视化"""
    
    def __init__(self, log=None, result_dir_path=None):
        """
        初始化可视化管理器
        
        参数:
        - log: 日志对象
        - result_dir_path: 结果保存路径
        """
        self.log = log
        self.result_dir_path = result_dir_path or Path("output/default/result")
        
    def clean_filename(self, filename):
        """清理文件名中的非法字符"""
        illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in illegal_chars:
            filename = filename.replace(char, '_')
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_')
        if len(filename) > 100:
            filename = filename[:100]
        return filename
    
    
    def visualize_results(self, results_df):
 
        """可视化预测结果（基于测试集）"""
        if results_df is None:
            return
        
        # 检查必要的列是否存在
        if '实际销量' not in results_df.columns or '预测销量' not in results_df.columns:
            self.log.warning('预测结果缺少实际值或预测值列，无法进行可视化')
            return
        
        # 计算误差
        results_df['预测误差'] = results_df['实际销量'] - results_df['预测销量']
        results_df['绝对误差'] = np.abs(results_df['预测误差'])
        results_df['相对误差百分比'] = (results_df['预测误差'] / np.where(results_df['实际销量'] != 0, results_df['实际销量'], 1)) * 100
        
        # 创建多个子图，更直观地展示结果
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 实际值与预测值散点图（测试集）
        plt.subplot(2, 3, 1)
        plt.scatter(results_df['实际销量'], results_df['预测销量'], alpha=0.5, s=20)
        plt.plot([results_df['实际销量'].min(), results_df['实际销量'].max()], 
                [results_df['实际销量'].min(), results_df['实际销量'].max()], 'r--', linewidth=2)
        plt.xlabel('实际销量')
        plt.ylabel('预测销量')
        plt.title('测试集：实际销量 vs 预测销量')
        plt.grid(True, alpha=0.3)
        
        # 2. 时间序列对比图（按日期排序，测试集）- 修复X轴显示问题
        plt.subplot(2, 3, 2)
        # 按日期排序
        if '日期' in results_df.columns:
            time_series_df = results_df.sort_values('日期')
            # 确保日期列是datetime类型
            time_series_df['日期'] = pd.to_datetime(time_series_df['日期'])
            
            # 限制样本数量避免图表过于密集，但保持时间连续性
            if len(time_series_df) > 100:
                # 按日期等间隔采样，保持时间分布
                date_range = pd.date_range(start=time_series_df['日期'].min(), 
                                         end=time_series_df['日期'].max(), 
                                         periods=min(100, len(time_series_df)))
                # 找到最接近采样日期的数据点
                sampled_indices = []
                for target_date in date_range:
                    closest_idx = (time_series_df['日期'] - target_date).abs().idxmin()
                    sampled_indices.append(closest_idx)
                sampled_df = time_series_df.loc[sorted(set(sampled_indices))]
            else:
                sampled_df = time_series_df
            
            # 使用实际日期作为X轴
            plt.plot(sampled_df['日期'], sampled_df['实际销量'], 'o-', label='实际销量', 
                    linewidth=2, markersize=4)
            plt.plot(sampled_df['日期'], sampled_df['预测销量'], 's-', label='预测销量', 
                    linewidth=2, markersize=4, alpha=0.8)
            plt.xlabel('日期')
            plt.ylabel('销量')
            plt.title('测试集：时间序列对比（9-10月）')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '无日期信息', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('测试集：时间序列对比（无日期信息）')
        
        # 3. 误差分布直方图（测试集）
        plt.subplot(2, 3, 3)
        plt.hist(results_df['预测误差'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('预测误差')
        plt.ylabel('频数')
        plt.title('测试集：预测误差分布')
        plt.grid(True, alpha=0.3)
        
        # 4. 相对误差百分比分布（测试集）
        plt.subplot(2, 3, 4)
        # 限制相对误差范围，避免极端值影响可视化
        relative_error_clipped = results_df['相对误差百分比'].clip(-100, 100)
        plt.hist(relative_error_clipped, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('相对误差百分比 (%)')
        plt.ylabel('频数')
        plt.title('测试集：相对误差百分比分布')
        plt.grid(True, alpha=0.3)
        
        # 5. 预测准确度热力图（按商品分组，测试集）
        plt.subplot(2, 3, 5)
        if '商品名称' in results_df.columns:
            # 按商品分组计算平均误差
            product_accuracy = results_df.groupby('商品名称').agg({
                '实际销量': 'mean',
                '预测销量': 'mean',
                '绝对误差': 'mean'
            }).reset_index()
            
            # 计算准确率（1 - 平均绝对误差/平均实际销量）
            product_accuracy['准确率'] = 1 - (product_accuracy['绝对误差'] / 
                                           np.where(product_accuracy['实际销量'] > 0, 
                                                   product_accuracy['实际销量'], 1))
            product_accuracy['准确率'] = product_accuracy['准确率'].clip(0, 1)
            
            # 取准确率最高的10个商品
            top_products = product_accuracy.nlargest(10, '准确率')
            
            plt.barh(range(len(top_products)), top_products['准确率'], 
                    color=plt.cm.viridis(top_products['准确率']))
            plt.yticks(range(len(top_products)), top_products['商品名称'])
            plt.xlabel('预测准确率')
            plt.title('测试集：各商品预测准确率Top10')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '无商品信息', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('测试集：商品预测准确率（无商品信息）')
        
        # 6. 误差统计信息（测试集）
        plt.subplot(2, 3, 6)
        # 清除当前子图，用于显示文本统计信息
        plt.axis('off')
        
        # 计算各种统计指标
        mae = results_df['绝对误差'].mean()
        rmse = np.sqrt((results_df['预测误差'] ** 2).mean())
        mape = results_df['相对误差百分比'].abs().mean()
        accuracy_within_10_percent = (results_df['相对误差百分比'].abs() <= 10).mean() * 100
        accuracy_within_20_percent = (results_df['相对误差百分比'].abs() <= 20).mean() * 100
        
        stats_text = f"""测试集预测性能统计:
        
        平均绝对误差 (MAE): {mae:.2f}
        均方根误差 (RMSE): {rmse:.2f}
        平均绝对百分比误差 (MAPE): {mape:.1f}%

        准确率统计:
        误差 ≤ 10%: {accuracy_within_10_percent:.1f}%
        误差 ≤ 20%: {accuracy_within_20_percent:.1f}%

        测试集样本总数: {len(results_df)}"""
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存可视化结果
        viz_path = self.result_dir_path / 'prediction_visualization_test_set.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info(f'测试集预测结果可视化已保存到: {viz_path}')
        
        # 创建专门的实际销量和预测销量对比图（测试集）
        self.create_comparison_chart(results_df)

    def create_comparison_chart(self, results_df):
        """创建实际销量和预测销量对比图"""
        if results_df is None or '实际销量' not in results_df.columns or '预测销量' not in results_df.columns:
            return
        
        # 创建对比图
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 实际销量和预测销量对比散点图（带对角线）
        plt.subplot(2, 2, 1)
        plt.scatter(results_df['实际销量'], results_df['预测销量'], alpha=0.6, s=30, 
                   c=results_df['预测误差'], cmap='coolwarm', vmin=-results_df['实际销量'].std(), 
                   vmax=results_df['实际销量'].std())
        plt.colorbar(label='预测误差')
        
        # 添加对角线（完美预测线）
        min_val = min(results_df['实际销量'].min(), results_df['预测销量'].min())
        max_val = max(results_df['实际销量'].max(), results_df['预测销量'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        
        plt.xlabel('实际销量')
        plt.ylabel('预测销量')
        plt.title('实际销量 vs 预测销量对比图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 实际销量和预测销量分布对比
        plt.subplot(2, 2, 2)
        plt.hist(results_df['实际销量'], bins=30, alpha=0.7, label='实际销量', color='blue')
        plt.hist(results_df['预测销量'], bins=30, alpha=0.7, label='预测销量', color='orange')
        plt.xlabel('销量')
        plt.ylabel('频数')
        plt.title('实际销量和预测销量分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 按销量大小分组的对比分析
        plt.subplot(2, 2, 3)
        # 修复：使用pd.cut而不是pd.qcut，避免重复边界问题
        try:
            # 尝试使用pd.qcut，如果失败则使用pd.cut
            results_df['销量等级'] = pd.qcut(results_df['实际销量'], q=5, labels=['很低', '低', '中等', '高', '很高'], duplicates='drop')
        except ValueError:
            # 如果pd.qcut失败，使用等距分箱
            unique_values = results_df['实际销量'].nunique()
            if unique_values >= 5:
                # 有足够的唯一值，使用等距分箱
                results_df['销量等级'] = pd.cut(results_df['实际销量'], bins=5, labels=['很低', '低', '中等', '高', '很高'])
            else:
                # 唯一值太少，直接使用唯一值作为等级
                unique_vals = sorted(results_df['实际销量'].unique())
                if len(unique_vals) >= 2:
                    bins = [unique_vals[0] - 1] + unique_vals + [unique_vals[-1] + 1]
                    labels = [f'等级{i+1}' for i in range(len(unique_vals))]
                    results_df['销量等级'] = pd.cut(results_df['实际销量'], bins=bins, labels=labels, include_lowest=True)
                else:
                    # 只有一个唯一值，无法分组
                    results_df['销量等级'] = '单一值'
        
        # 计算每个等级的平均实际销量和预测销量
        if '销量等级' in results_df.columns and results_df['销量等级'].nunique() > 1:
            level_comparison = results_df.groupby('销量等级').agg({
                '实际销量': 'mean',
                '预测销量': 'mean',
                '预测误差': 'mean'
            }).reset_index()
            
            x = range(len(level_comparison))
            width = 0.35
            plt.bar([i - width/2 for i in x], level_comparison['实际销量'], width, label='平均实际销量', alpha=0.7)
            plt.bar([i + width/2 for i in x], level_comparison['预测销量'], width, label='平均预测销量', alpha=0.7)
            plt.xticks(x, level_comparison['销量等级'])
            plt.xlabel('销量等级')
            plt.ylabel('平均销量')
            plt.title('不同销量等级的实际vs预测对比')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # 如果无法分组，显示提示信息
            plt.text(0.5, 0.5, '销量数据过于集中，无法分组对比', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('销量等级对比（数据过于集中）')
        
        # 4. 预测准确度分析
        plt.subplot(2, 2, 4)
        # 计算不同误差范围内的样本比例
        error_ranges = [
            ('误差≤10%', (results_df['相对误差百分比'].abs() <= 10).mean() * 100),
            ('10%<误差≤20%', ((results_df['相对误差百分比'].abs() > 10) & 
                            (results_df['相对误差百分比'].abs() <= 20)).mean() * 100),
            ('20%<误差≤50%', ((results_df['相对误差百分比'].abs() > 20) & 
                            (results_df['相对误差百分比'].abs() <= 50)).mean() * 100),
            ('误差>50%', (results_df['相对误差百分比'].abs() > 50).mean() * 100)
        ]
        
        labels = [er[0] for er in error_ranges]
        sizes = [er[1] for er in error_ranges]
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('预测准确度分布')
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_path = self.result_dir_path / 'prediction_comparison_chart.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info(f'实际销量和预测销量对比图已保存到: {comparison_path}')
        
        # 创建更详细的对比分析报告
        self.create_detailed_comparison_report(results_df)



    def create_detailed_comparison_report(self, results_df):
        """创建详细的对比分析报告"""
        if results_df is None:
            return
        
        # 创建更详细的对比图
        fig = plt.figure(figsize=(18, 15))
        
        # 1. 实际销量和预测销量时间序列对比（修复X轴显示问题）
        plt.subplot(3, 2, 1)
        if '日期' in results_df.columns:
            # 按日期排序
            time_series_df = results_df.sort_values('日期')
            # 确保日期列是datetime类型
            time_series_df['日期'] = pd.to_datetime(time_series_df['日期'])
            
            # 取前100个样本避免图表过于密集，但保持时间连续性
            if len(time_series_df) > 100:
                # 按日期等间隔采样
                date_range = pd.date_range(start=time_series_df['日期'].min(), 
                                         end=time_series_df['日期'].max(), 
                                         periods=min(100, len(time_series_df)))
                sampled_indices = []
                for target_date in date_range:
                    closest_idx = (time_series_df['日期'] - target_date).abs().idxmin()
                    sampled_indices.append(closest_idx)
                sampled_df = time_series_df.loc[sorted(set(sampled_indices))]
            else:
                sampled_df = time_series_df
            
            plt.plot(sampled_df['日期'], sampled_df['实际销量'], 'o-', label='实际销量', 
                    linewidth=2, markersize=4)
            plt.plot(sampled_df['日期'], sampled_df['预测销量'], 's-', label='预测销量', 
                    linewidth=2, markersize=4, alpha=0.7)
            plt.xlabel('日期')
            plt.ylabel('销量')
            plt.title('实际销量和预测销量时间序列对比（9-10月）')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        else:
            # 如果没有日期，使用索引
            sample_size = min(100, len(results_df))
            indices = np.linspace(0, len(results_df)-1, sample_size, dtype=int)
            sampled_df = results_df.iloc[indices]
            
            plt.plot(range(sample_size), sampled_df['实际销量'], 'o-', label='实际销量', 
                    linewidth=2, markersize=4)
            plt.plot(range(sample_size), sampled_df['预测销量'], 's-', label='预测销量', 
                    linewidth=2, markersize=4, alpha=0.7)
            plt.xlabel('样本序号')
            plt.ylabel('销量')
            plt.title('实际销量和预测销量对比')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. 误差与销量的关系
        plt.subplot(3, 2, 2)
        plt.scatter(results_df['实际销量'], results_df['预测误差'], alpha=0.6, s=20)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('实际销量')
        plt.ylabel('预测误差')
        plt.title('预测误差与实际销量的关系')
        plt.grid(True, alpha=0.3)
        
        # 3. 相对误差分布
        plt.subplot(3, 2, 3)
        relative_error = results_df['相对误差百分比'].clip(-100, 100)
        plt.hist(relative_error, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('相对误差百分比 (%)')
        plt.ylabel('频数')
        plt.title('相对误差百分比分布')
        plt.grid(True, alpha=0.3)
        
        # 4. 累积准确率曲线
        plt.subplot(3, 2, 4)
        # 按绝对误差排序
        sorted_errors = np.sort(results_df['绝对误差'])
        cumulative_accuracy = np.arange(1, len(sorted_errors)+1) / len(sorted_errors) * 100
        
        plt.plot(sorted_errors, cumulative_accuracy, linewidth=2)
        plt.xlabel('绝对误差阈值')
        plt.ylabel('累积准确率 (%)')
        plt.title('累积准确率曲线')
        plt.grid(True, alpha=0.3)
        
        # 5. 按商品分组的预测性能（如果存在商品信息）
        plt.subplot(3, 2, 5)
        if '商品名称' in results_df.columns:
            # 计算每个商品的预测性能
            product_stats = results_df.groupby('商品名称').agg({
                '实际销量': ['count', 'mean'],
                '预测销量': 'mean',
                '绝对误差': 'mean',
                '相对误差百分比': lambda x: np.mean(np.abs(x))
            }).round(2)
            
            product_stats.columns = ['样本数', '平均实际销量', '平均预测销量', '平均绝对误差', '平均相对误差%']
            product_stats['准确率'] = 1 - (product_stats['平均绝对误差'] / 
                                      np.where(product_stats['平均实际销量'] > 0, 
                                              product_stats['平均实际销量'], 1))
            
            # 取样本数最多的10个商品
            top_products = product_stats.nlargest(10, '样本数')
            
            x = range(len(top_products))
            width = 0.35
            plt.bar([i - width/2 for i in x], top_products['平均实际销量'], width, 
                   label='平均实际销量', alpha=0.7)
            plt.bar([i + width/2 for i in x], top_products['平均预测销量'], width, 
                   label='平均预测销量', alpha=0.7)
            plt.xticks(x, top_products.index, rotation=45)
            plt.xlabel('商品名称')
            plt.ylabel('平均销量')
            plt.title('主要商品的实际vs预测对比')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '无商品信息', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('商品对比分析（无商品信息）')
        
        # 6. 预测性能统计面板
        plt.subplot(3, 2, 6)
        plt.axis('off')
        
        # 计算详细统计指标
        mae = results_df['绝对误差'].mean()
        rmse = np.sqrt((results_df['预测误差'] ** 2).mean())
        mape = results_df['相对误差百分比'].abs().mean()
        
        # 计算不同误差范围内的准确率
        accuracy_10 = (results_df['相对误差百分比'].abs() <= 10).mean() * 100
        accuracy_20 = (results_df['相对误差百分比'].abs() <= 20).mean() * 100
        accuracy_30 = (results_df['相对误差百分比'].abs() <= 30).mean() * 100
        
        # 计算误差方向统计
        over_prediction = (results_df['预测误差'] > 0).mean() * 100  # 高估的比例
        under_prediction = (results_df['预测误差'] < 0).mean() * 100  # 低估的比例
        perfect_prediction = (results_df['预测误差'] == 0).mean() * 100  # 完美预测的比例
        
        stats_text = f"""详细预测性能分析:
        
        基本指标:
        • 平均绝对误差 (MAE): {mae:.2f}
        • 均方根误差 (RMSE): {rmse:.2f}
        • 平均绝对百分比误差 (MAPE): {mape:.1f}%
        
        准确率统计:
        • 误差 ≤ 10%: {accuracy_10:.1f}%
        • 误差 ≤ 20%: {accuracy_20:.1f}%
        • 误差 ≤ 30%: {accuracy_30:.1f}%
        
        预测偏差分析:
        • 高估预测 (预测 > 实际): {over_prediction:.1f}%
        • 低估预测 (预测 < 实际): {under_prediction:.1f}%
        • 完美预测: {perfect_prediction:.1f}%
        
        样本统计:
        • 总样本数: {len(results_df)}
        • 平均实际销量: {results_df['实际销量'].mean():.2f}
        • 平均预测销量: {results_df['预测销量'].mean():.2f}"""
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # 保存详细对比报告
        detailed_path = self.result_dir_path / 'prediction_detailed_comparison.png'
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info(f'详细对比分析报告已保存到: {detailed_path}')
        
        # 创建最准确产品的曲线图
        self.create_best_product_curve(results_df)

    def create_best_product_curve(self, results_df):
        
        """创建最准确产品的实际销量和预测销量曲线图"""
        if results_df is None or '商品名称' not in results_df.columns or '日期' not in results_df.columns:
            self.log.warning('缺少商品名称或日期信息，无法创建最准确产品曲线图')
            return
        
         # 新增：获取销售量最高的5个产品
        if '实际销量' in results_df.columns:
            top5_sales_products = results_df.groupby('商品名称')['实际销量'].sum().nlargest(5)
            self.log.info(f'销售量最高的5个产品: {top5_sales_products.to_dict()}')
        
        
        # 计算每个商品的预测准确率
        product_accuracy = results_df.groupby('商品名称').agg({
            '实际销量': 'count',
            '预测误差': lambda x: np.mean(np.abs(x)),
            '相对误差百分比': lambda x: np.mean(np.abs(x))
        }).reset_index()
        
        product_accuracy.columns = ['商品名称', '样本数', '平均绝对误差', '平均相对误差%']
        
        # 计算准确率（1 - 平均绝对误差/平均实际销量）
        product_means = results_df.groupby('商品名称')['实际销量'].mean().reset_index()
        product_accuracy = product_accuracy.merge(product_means, on='商品名称')
        product_accuracy['准确率'] = 1 - (product_accuracy['平均绝对误差'] / 
                                      np.where(product_accuracy['实际销量'] > 0, 
                                              product_accuracy['实际销量'], 1))
        product_accuracy['准确率'] = product_accuracy['准确率'].clip(0, 1)
        
        # 筛选样本数足够多的商品（至少5个样本）
        valid_products = product_accuracy[product_accuracy['样本数'] >= 5]
        
        if len(valid_products) == 0:
            self.log.warning('没有足够样本的商品，无法创建最准确产品曲线图')
            return
        
        # 改进选择逻辑：从样本量最多的前10个商品中选择最准确的3个
        if len(valid_products) > 10:
            # 先按样本数排序，取前10个样本量最多的商品
            top_sample_products = valid_products.nlargest(10, '样本数')
            # 再从这10个商品中按准确率排序，取最准确的前3个
            best_products = top_sample_products.nlargest(3, '准确率')
            self.log.info('从样本量最多的前10个商品中选择最准确的3个商品')
        else:
            # 如果商品数量不足10个，直接按准确率排序取前3个
            best_products = valid_products.nlargest(3, '准确率')
            self.log.info(f'商品数量较少（{len(valid_products)}个），直接选择最准确的3个商品')
        
        # 记录选择过程信息
        self.log.info(f'样本量最多的前5个商品: {valid_products.nlargest(5, "样本数")[["商品名称", "样本数", "准确率"]].to_string(index=False)}')
        self.log.info(f'最终选择的3个最准确商品: {best_products[["商品名称", "准确率", "样本数"]].to_string(index=False)}')
        
        # 创建曲线图
        fig, axes = plt.subplots(len(best_products), 1, figsize=(15, 5 * len(best_products)))
        
        # 如果只有一个商品，确保axes是列表形式
        if len(best_products) == 1:
            axes = [axes]
        
        for i, (idx, product) in enumerate(best_products.iterrows()):
            product_name = product['商品名称']
            accuracy = product['准确率']
            sample_count = product['样本数']
            
            # 筛选该商品的数据
            product_data = results_df[results_df['商品名称'] == product_name].sort_values('日期')
            
            # 创建子图
            ax = axes[i]
            ax.plot(product_data['日期'], product_data['实际销量'], 'o-', 
                   label='实际销量', linewidth=2, markersize=4, color='blue')
            ax.plot(product_data['日期'], product_data['预测销量'], 's-', 
                   label='预测销量', linewidth=2, markersize=4, color='red', alpha=0.7)
            
            # 设置标题和标签
            ax.set_title(f'{product_name} - 准确率: {accuracy:.2%} (样本数: {sample_count})', fontsize=12)
            ax.set_xlabel('日期')
            ax.set_ylabel('销量')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # 添加误差统计信息
            product_mae = product_data['预测误差'].abs().mean()
            product_rmse = np.sqrt((product_data['预测误差'] ** 2).mean())
            product_mape = product_data['相对误差百分比'].abs().mean()
            
            # 在图表右上角添加统计信息
            stats_text = f'MAE: {product_mae:.2f}\nRMSE: {product_rmse:.2f}\nMAPE: {product_mape:.1f}%'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存最准确产品曲线图
        best_product_path = self.result_dir_path / 'best_product_prediction_curves.png'
        plt.savefig(best_product_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info(f'最准确产品预测曲线图已保存到: {best_product_path}')
        
        # 创建单个最准确产品的详细曲线图
        if len(best_products) > 0:
            self.create_single_best_product_curve(results_df, best_products.iloc[0]['商品名称'])

    def create_single_best_product_curve(self, results_df, best_product_name):
        """创建单个最准确产品的详细曲线图"""
        # 筛选最准确产品的数据
        product_data = results_df[results_df['商品名称'] == best_product_name].sort_values('日期')
        
        if len(product_data) == 0:
            return
        
        # 创建更详细的单个产品曲线图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 实际销量和预测销量对比曲线
        ax1.plot(product_data['日期'], product_data['实际销量'], 'o-', 
                label='实际销量', linewidth=2, markersize=6, color='blue')
        ax1.plot(product_data['日期'], product_data['预测销量'], 's-', 
                label='预测销量', linewidth=2, markersize=4, color='red', alpha=0.8)
        
        # 填充实际和预测之间的区域
        ax1.fill_between(product_data['日期'], product_data['实际销量'], product_data['预测销量'],
                        alpha=0.3, color='orange', label='预测误差')
        
        ax1.set_title(f'最准确产品: {best_product_name} - 实际销量 vs 预测销量', fontsize=14)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('销量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 预测误差曲线
        ax2.plot(product_data['日期'], product_data['预测误差'], 'o-', 
                color='purple', linewidth=2, markersize=4)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='零误差线')
        
        # 填充正负误差区域
        ax2.fill_between(product_data['日期'], 0, product_data['预测误差'], 
                        where=(product_data['预测误差'] > 0), alpha=0.3, color='red', label='高估')
        ax2.fill_between(product_data['日期'], 0, product_data['预测误差'], 
                        where=(product_data['预测误差'] < 0), alpha=0.3, color='green', label='低估')
        
        ax2.set_title(f'{best_product_name} - 预测误差分析', fontsize=14)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('预测误差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 清理商品名称用于文件名
        clean_product_name = self.clean_filename(best_product_name)
        
        # 保存单个最准确产品详细曲线图
        single_product_path = self.result_dir_path / f'best_product_{clean_product_name}_detailed.png'
        plt.savefig(single_product_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info(f'单个最准确产品详细曲线图已保存到: {single_product_path}')


    def create_model_comparison_visualization(self, comparison_df):
        """创建模型性能比较可视化图表"""
        if comparison_df is None or len(comparison_df) == 0:
            self.log.warning('没有模型比较数据，无法创建可视化图表')
            return
        
        self.log.info('开始生成模型性能比较可视化...')
        
        # 创建综合对比图表
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 模型性能指标对比柱状图
        plt.subplot(2, 3, 1)
        
        # 提取主要性能指标
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
        available_metrics = [metric for metric in metrics_to_plot if metric in comparison_df.columns]
        
        if available_metrics:
            # 为每个指标创建子图
            x = np.arange(len(comparison_df))
            width = 0.25
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            
            for i, metric in enumerate(available_metrics):
                metric_values = comparison_df[metric].values
                plt.bar(x + i*width, metric_values, width, label=metric, color=colors[i], alpha=0.8)
            
            plt.xlabel('模型')
            plt.ylabel('指标值')
            plt.title('模型性能指标对比')
            plt.xticks(x + width*(len(available_metrics)-1)/2, comparison_df.index, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '无可用性能指标数据', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('模型性能指标对比')
        
        # 2. 模型排名热力图
        plt.subplot(2, 3, 2)
        
        # 为每个指标计算排名（越小越好）
        rank_df = comparison_df.copy()
        for col in rank_df.columns:
            if col in ['RMSE', 'MAE', 'MAPE']:  # 越小越好的指标
                rank_df[col] = rank_df[col].rank()
            elif col in ['R2', 'Accuracy']:  # 越大越好的指标
                rank_df[col] = rank_df[col].rank(ascending=False)
        
        # 只保留数值型列
        numeric_cols = rank_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # 创建热力图
            im = plt.imshow(rank_df[numeric_cols].values, cmap='RdYlGn_r', aspect='auto')
            plt.colorbar(im, label='排名（绿色越好）')
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
            plt.yticks(range(len(rank_df)), rank_df.index)
            plt.title('模型性能排名热力图')
            
            # 添加排名数值
            for i in range(len(rank_df)):
                for j in range(len(numeric_cols)):
                    plt.text(j, i, f'{int(rank_df.iloc[i, j])}', 
                            ha='center', va='center', fontsize=8, fontweight='bold')
        else:
            plt.text(0.5, 0.5, '无数值型指标数据', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('模型性能排名热力图')
        
        # 3. 模型相对性能雷达图
        plt.subplot(2, 3, 3, polar=True)
        
        # 选择几个关键指标进行雷达图展示
        key_metrics = ['RMSE', 'MAE', 'MAPE', 'R2'] if 'R2' in comparison_df.columns else ['RMSE', 'MAE', 'MAPE']
        available_key_metrics = [metric for metric in key_metrics if metric in comparison_df.columns]
        
        if len(available_key_metrics) >= 3:
            # 标准化指标（对于越小越好的指标，进行反向处理）
            normalized_df = comparison_df[available_key_metrics].copy()
            for metric in available_key_metrics:
                if metric in ['RMSE', 'MAE', 'MAPE']:
                    # 越小越好，所以用1-标准化值
                    max_val = normalized_df[metric].max()
                    min_val = normalized_df[metric].min()
                    if max_val != min_val:
                        normalized_df[metric] = 1 - (normalized_df[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_df[metric] = 0.5
                elif metric == 'R2':
                    # 越大越好，直接标准化
                    max_val = normalized_df[metric].max()
                    min_val = normalized_df[metric].min()
                    if max_val != min_val:
                        normalized_df[metric] = (normalized_df[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_df[metric] = 0.5
            
            # 设置雷达图角度
            angles = np.linspace(0, 2*np.pi, len(available_key_metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合雷达图
            
            # 为每个模型绘制雷达图
            colors = plt.cm.Set1(np.linspace(0, 1, len(comparison_df)))
            for i, (model_name, metrics) in enumerate(normalized_df.iterrows()):
                values = metrics.values.tolist()
                values += values[:1]  # 闭合雷达图
                plt.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
                plt.fill(angles, values, alpha=0.1, color=colors[i])
            
            # 设置雷达图标签
            plt.xticks(angles[:-1], available_key_metrics)
            plt.yticks([0.2, 0.4, 0.6, 0.8], ['20%', '40%', '60%', '80%'])
            plt.title('模型相对性能雷达图', size=14, y=1.08)
            plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        else:
            plt.text(0.5, 0.5, '指标数量不足，无法创建雷达图', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('模型相对性能雷达图')
        
        # 4. 模型性能散点图矩阵（如果指标足够多）
        plt.subplot(2, 3, 4)
        
        if len(comparison_df.columns) >= 2:
            # 选择前两个指标进行散点图展示
            metric1, metric2 = comparison_df.columns[:2]
            plt.scatter(comparison_df[metric1], comparison_df[metric2], s=100, alpha=0.7)
            
            # 添加模型标签
            for i, (model_name, metrics) in enumerate(comparison_df.iterrows()):
                plt.annotate(model_name, (metrics[metric1], metrics[metric2]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel(metric1)
            plt.ylabel(metric2)
            plt.title(f'{metric1} vs {metric2} 散点图')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '指标数量不足，无法创建散点图', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('模型性能散点图')
        
        # 5. 最佳模型性能展示
        plt.subplot(2, 3, 5)
        
        best_model_name = comparison_df.index[0]  # 第一个是最佳模型
        best_model_metrics = comparison_df.loc[best_model_name]
        
        # 选择几个关键指标进行展示
        key_metrics_for_best = ['RMSE', 'MAE', 'MAPE', 'R2'] if 'R2' in comparison_df.columns else ['RMSE', 'MAE', 'MAPE']
        available_best_metrics = [metric for metric in key_metrics_for_best if metric in best_model_metrics.index]
        
        if available_best_metrics:
            values = [best_model_metrics[metric] for metric in available_best_metrics]
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(available_best_metrics)]
            
            bars = plt.bar(available_best_metrics, values, color=colors, alpha=0.8)
            plt.xlabel('性能指标')
            plt.ylabel('指标值')
            plt.title(f'最佳模型 ({best_model_name}) 性能指标')
            plt.grid(True, alpha=0.3)
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            plt.text(0.5, 0.5, '无关键指标数据', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('最佳模型性能指标')
        
        # 6. 详细性能指标表格
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # 创建详细的性能指标表格
        table_data = []
        for model_name, metrics in comparison_df.iterrows():
            row = [model_name]
            for metric in comparison_df.columns:
                if isinstance(metrics[metric], (int, float)):
                    row.append(f'{metrics[metric]:.4f}')
                else:
                    row.append(str(metrics[metric]))
            table_data.append(row)
        
        # 创建表格
        table = plt.table(cellText=table_data,
                         colLabels=['模型'] + list(comparison_df.columns),
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        plt.title('详细模型性能指标对比', y=0.95)
        
        plt.tight_layout()
        
        # 保存模型性能比较可视化图表
        model_comparison_path = self.result_dir_path / 'model_comparison_visualization.png'
        plt.savefig(model_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细的性能指标到CSV文件
        csv_path = self.result_dir_path / 'model_performance_metrics.csv'
        comparison_df.to_csv(csv_path, encoding='utf-8')
        
        self.log.info(f'模型性能比较可视化图表已保存到: {model_comparison_path}')
        self.log.info(f'详细性能指标已保存到: {csv_path}')
        
        # 打印最佳模型详细信息
        self.log.info('=== 最佳模型性能详情 ===')
        self.log.info(f'最佳模型: {best_model_name}')
        for metric, value in best_model_metrics.items():
            self.log.info(f'  {metric}: {value}')

    def visualize_discount_plan(self, discount_result):
            """
            可视化折扣方案

            参数:
            - discount_result: 折扣方案结果字典
            """
            try:
                self.log.info(f"开始可视化折扣方案，商品: {discount_result.get('product_code', '未知')}")

                # 提取折扣计划数据
                if 'discount_plan' not in discount_result or not discount_result['discount_plan']:
                    self.log.warning('折扣计划数据为空，无法可视化')
                    return

                plan_df = pd.DataFrame(discount_result['discount_plan'])

                # 创建多子图可视化
                fig = plt.figure(figsize=(15, 10))

                # 1. 折扣率随时间变化
                plt.subplot(2, 2, 1)
                plt.plot(range(len(plan_df)), plan_df['discount_percentage'], 'o-', linewidth=2, markersize=8)
                plt.xlabel('时间段')
                plt.ylabel('折扣率 (%)')
                plt.title(f"{discount_result.get('product_code', '商品')} - 折扣率变化趋势")
                plt.grid(True, alpha=0.3)

                # 设置x轴标签
                if 'time_slot' in plan_df.columns:
                    plt.xticks(range(len(plan_df)), plan_df['time_slot'], rotation=45)

                # 2. 预期销量分布
                plt.subplot(2, 2, 2)
                plt.bar(range(len(plan_df)), plan_df['expected_sales'], alpha=0.7)
                plt.xlabel('时间段')
                plt.ylabel('预期销量')
                plt.title('各时间段预期销量分布')
                plt.grid(True, alpha=0.3)

                # 3. 累计销量曲线
                plt.subplot(2, 2, 3)
                cumulative_sales = plan_df['expected_sales'].cumsum()
                plt.plot(range(len(cumulative_sales)), cumulative_sales, 's-', linewidth=2, markersize=6)
                plt.axhline(y=discount_result.get('current_inventory', 0), color='r',
                            linestyle='--', alpha=0.5, label='总库存')
                plt.xlabel('时间段')
                plt.ylabel('累计销量')
                plt.title('累计销量曲线')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # 4. 预期利润分析
                plt.subplot(2, 2, 4)
                x = range(len(plan_df))
                width = 0.35

                plt.bar([i - width / 2 for i in x], plan_df['expected_revenue'], width,
                        label='预期收入', alpha=0.7)
                plt.bar([i + width / 2 for i in x], plan_df['expected_profit'], width,
                        label='预期利润', alpha=0.7)
                plt.xlabel('时间段')
                plt.ylabel('金额 (元)')
                plt.title('预期收入和利润对比')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.tight_layout()

                # 保存可视化结果
                product_code = discount_result.get('product_code', 'unknown')
                viz_path = self.result_dir_path / f'discount_plan_{product_code}.png'
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()

                self.log.info(f'折扣方案可视化已保存到: {viz_path}')

                # 创建详细分析报告
                self._create_discount_report(discount_result, plan_df)

            except Exception as e:
                self.log.error(f'折扣方案可视化失败: {str(e)}')

    def _create_discount_report(self, discount_result, plan_df):
        """创建折扣方案详细报告"""
        try:
            fig = plt.figure(figsize=(12, 8))

            # 创建文本报告
            plt.axis('off')

            # 提取关键信息
            product_code = discount_result.get('product_code', '未知')
            current_inventory = discount_result.get('current_inventory', 0)
            current_price = discount_result.get('current_price', 0)
            cost_price = discount_result.get('cost_price', 0)
            min_margin = discount_result.get('min_gross_margin', 0)

            # 计算汇总指标
            total_expected_sales = plan_df['expected_sales'].sum()
            total_expected_revenue = plan_df['expected_revenue'].sum()
            total_expected_profit = plan_df['expected_profit'].sum()
            clearance_rate = total_expected_sales / current_inventory if current_inventory > 0 else 0

            # 创建报告文本
            report_text = f"""折扣方案分析报告
商品编码: {product_code}
当前库存: {current_inventory}
当前售价: {current_price:.2f}元
成本价: {cost_price:.2f}元
最低毛利率: {min_margin * 100:.1f}%

方案汇总:
预期总销量: {total_expected_sales:.0f}个
预期总收入: {total_expected_revenue:.2f}元
预期总利润: {total_expected_profit:.2f}元
库存清空率: {clearance_rate:.1%}

各时间段方案:"""

            # 添加每个时间段的详细信息
            for i, row in plan_df.iterrows():
                report_text += f"\n\n时间段 {i + 1}: {row.get('time_slot', '未知')}"
                report_text += f"\n  折扣: {row.get('discount_percentage', 0):.1f}%"
                report_text += f"\n  最终售价: {row.get('final_price', 0):.2f}元"
                report_text += f"\n  预期销量: {row.get('expected_sales', 0):.0f}个"
                report_text += f"\n  预期收入: {row.get('expected_revenue', 0):.2f}元"
                report_text += f"\n  预期利润: {row.get('expected_profit', 0):.2f}元"

            # 可行性分析
            feasibility = discount_result.get('feasibility_analysis', {})
            if feasibility:
                report_text += "\n\n可行性分析:"
                for key, value in feasibility.items():
                    if isinstance(value, bool):
                        report_text += f"\n  {key}: {'✓' if value else '✗'}"
                    elif isinstance(value, (int, float)):
                        if 'rate' in key:
                            report_text += f"\n  {key}: {value:.1%}"
                        else:
                            report_text += f"\n  {key}: {value:.2f}"
                    else:
                        report_text += f"\n  {key}: {value}"

            # 添加总结
            summary = discount_result.get('summary', {})
            if 'recommendation' in summary:
                report_text += f"\n\n推荐建议:\n{summary['recommendation']}"

            # 显示报告文本
            plt.text(0.05, 0.95, report_text, transform=plt.gca().transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            plt.tight_layout()

            # 保存报告
            report_path = self.result_dir_path / f'discount_report_{product_code}.png'
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.log.info(f'折扣方案详细报告已保存到: {report_path}')

        except Exception as e:
            self.log.error(f'创建折扣报告失败: {str(e)}')