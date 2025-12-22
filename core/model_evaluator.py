# core/model_evaluator.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import os
import platform
import matplotlib.dates as mdates

@dataclass
class ModelEvaluationResult:
    """模型评估结果"""
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: Dict[str, float]
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None
    data_quality_warning: Optional[str] = None  # 新增：数据质量警告


class SimplifiedModelVisualizer:
    """简化模型可视化器"""

    def __init__(self, output_dir: str = "output", config = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置matplotlib参数
        # plt.rcParams['figure.dpi'] = 100
        # plt.rcParams['savefig.dpi'] = 150
        # plt.rcParams['figure.figsize'] = (10, 8)
        # 可视化配置
        self.viz_output_dir = config.get('visualization_output_dir', output_dir+"./viz") if config else 'output/viz'
        os.makedirs(self.viz_output_dir, exist_ok=True)
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        # 设置matplotlib样式
        # plt.style.use('default')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',  'DejaVu Sans', 'Arial Unicode MS', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        # print(f"可视化器初始化完成，使用字体: {FONT_NAME}")

    def _check_data_quality(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[str]:
        """检查数据质量，返回警告信息"""
        n_samples = len(y_true)

        # 检查样本数量
        if n_samples < 10:
            return f"数据样本过少（{n_samples}个），模型可能不可靠"

        # 检查预测值与真实值的差异
        if np.allclose(y_true, y_pred, rtol=1e-5):
            return "预测值与真实值完全相同，可能存在问题"

        # 检查R²值
        try:
            r2 = r2_score(y_true, y_pred)
            if r2 < 0:
                return f"R2值为负（{r2:.3f}），模型拟合效果差"
            elif r2 < 0.3:
                return f"R2值较低（{r2:.3f}），模型解释力弱"
        except:
            pass

        return None

    def _create_simple_scatter(self, y_true: np.ndarray, y_pred: np.ndarray,
                               title: str, ax: plt.Axes, color: str = 'blue'):
        """创建简单的散点图"""
        # 确保数据是数值类型
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # 创建散点图
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, color=color, edgecolors='w', linewidth=0.5)

        # 添加对角线
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想预测线')

        # 计算并显示R²
        try:
            r2 = r2_score(y_true, y_pred)
            ax.text(0.05, 0.95, f'R2 = {r2:.3f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except:
            pass

        ax.set_xlabel('实际销量', fontsize=12)
        ax.set_ylabel('预测销量', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def create_prediction_comparison_chart(self,
                                           y_true: np.ndarray,
                                           y_pred: np.ndarray,
                                           product_code: str,
                                           store_code: Optional[str] = None,
                                           save_path: Optional[str] = None) -> str:
        """
        创建预测值与实际值对比图 - 修复版
        """
        try:
            # 转换数据为数值类型
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

            # 检查数据长度
            if len(y_true) != len(y_pred):
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            # 检查数据质量
            warning = self._check_data_quality(y_true, y_pred)

            # 根据样本数量决定图表类型
            n_samples = len(y_true)

            if n_samples < 5:
                # 样本太少，创建简单图表
                fig, ax = plt.subplots(figsize=(8, 6))

                # 创建简单散点图
                self._create_simple_scatter(y_true, y_pred,
                                            '预测值与实际值对比（数据量不足）', ax)

                # 添加警告文本
                if warning:
                    ax.text(0.5, 0.02, f'警告: {warning}',
                            transform=ax.transAxes, ha='center',
                            fontsize=11, color='red',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

            else:
                # 创建标准图表
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # 1. 预测vs实际散点图
                ax1 = axes[0, 0]
                self._create_simple_scatter(y_true, y_pred, '预测值 vs 实际值', ax1)

                # 2. 残差图
                ax2 = axes[0, 1]
                residuals = y_true - y_pred
                ax2.scatter(y_pred, residuals, alpha=0.6, s=50, color='green',
                            edgecolors='w', linewidth=0.5)
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
                ax2.set_xlabel('预测值', fontsize=12)
                ax2.set_ylabel('残差', fontsize=12)
                ax2.set_title('残差分析', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)

                # 3. 误差分布直方图
                ax3 = axes[1, 0]
                absolute_errors = np.abs(residuals)
                ax3.hist(absolute_errors, bins=min(20, len(y_true) // 2),
                         alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(x=np.mean(absolute_errors), color='r', linestyle='--',
                            label=f'平均误差: {np.mean(absolute_errors):.2f}')
                ax3.set_xlabel('绝对误差', fontsize=12)
                ax3.set_ylabel('频数', fontsize=12)
                ax3.set_title('误差分布', fontsize=14, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                # 4. 时间序列对比
                ax4 = axes[1, 1]
                indices = np.arange(len(y_true))

                # 限制显示的点数
                max_points = min(50, len(y_true))
                if len(y_true) > max_points:
                    step = len(y_true) // max_points
                    sample_idx = indices[::step]
                    ax4.plot(sample_idx, y_true[::step], 'o-', label='实际值', alpha=0.7, markersize=4)
                    ax4.plot(sample_idx, y_pred[::step], 's--', label='预测值', alpha=0.7, markersize=4)
                else:
                    ax4.plot(indices, y_true, 'o-', label='实际值', alpha=0.7, markersize=4)
                    ax4.plot(indices, y_pred, 's--', label='预测值', alpha=0.7, markersize=4)

                ax4.set_xlabel('样本索引', fontsize=12)
                ax4.set_ylabel('销量', fontsize=12)
                ax4.set_title('实际值与预测值对比', fontsize=14, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

                # 添加警告（如果有）
                if warning:
                    fig.text(0.5, 0.01, f'注意: {warning}',
                             ha='center', fontsize=11, color='red',
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

            # 设置总标题
            store_info = f" - 门店: {store_code}" if store_code else ""
            title_text = f"模型评估 - 商品: {product_code}{store_info}"

            plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()

            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir,
                                         f"model_eval_{product_code}_{store_code or 'all'}_{timestamp}.png")

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"预测对比图已保存: {save_path}")
            return save_path

        except Exception as e:
            print(f"创建预测对比图失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    # 创建误差比较图
    def create_metrics_card(self, metrics: Dict[str, any],
                            product_code: str,
                            save_path: Optional[str] = None) -> str:
        """
        创建模型性能指标卡片 - 修复版
        """
        try:
            # 确保指标是数值类型
            float_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    float_metrics[key] = float(value)
                elif isinstance(value, str):
                    # 处理百分比字符串
                    if '%' in value:
                        try:
                            float_metrics[key] = float(value.replace('%', '').strip())
                        except:
                            float_metrics[key] = 0.0
                    else:
                        try:
                            float_metrics[key] = float(value)
                        except:
                            float_metrics[key] = 0.0
                else:
                    float_metrics[key] = 0.0

            # 检查R²值
            r2 = float_metrics.get('r2', 0)

            fig, ax = plt.subplots(figsize=(10, 6))

            # 创建指标数据
            metric_names = ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE']
            metric_values = [
                float_metrics.get('mae', 0),
                float_metrics.get('mse', 0),
                float_metrics.get('rmse', 0),
                r2,
                float_metrics.get('mape', 0)
            ]

            # 设置颜色（R²为负时用红色突出显示）
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            if r2 < 0:
                colors[3] = '#FF0000'  # 红色突出显示

            # 创建柱状图
            bars = ax.bar(metric_names, metric_values, color=colors)

            # 添加数值标签
            for bar, name, value in zip(bars, metric_names, metric_values):
                height = bar.get_height()
                if name == 'R2':
                    # R²特殊格式
                    label = f'{value:.3f}'
                    if value < 0:
                        label = f'{value:.3f} (警告!)'
                elif name == 'MAPE':
                    label = f'{value:.1f}%'
                else:
                    label = f'{value:.3f}'

                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        label, ha='center', va='bottom', fontsize=11, fontweight='bold')

            # 设置Y轴标签和标题
            ylabel = '指标值'
            title = f'模型性能指标 - {product_code}'

            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # 添加性能评级
            if r2 > 0.8:
                rating = "优秀 ✓"
                rating_color = "green"
            elif r2 > 0.6:
                rating = "良好 ~"
                rating_color = "orange"
            elif r2 > 0:
                rating = "需改进 ✗"
                rating_color = "red"
            else:
                rating = "警告: R2为负!"
                rating_color = "darkred"

            rating_text = f'模型评级: {rating}'
            ax.text(0.02, 0.98, rating_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    color=rating_color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            plt.tight_layout()

            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir,
                                         f"metrics_card_{product_code}_{timestamp}.png")

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"指标卡片已保存: {save_path}")
            return save_path

        except Exception as e:
            print(f"创建指标卡片失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def create_comprehensive_report(self,
                                    evaluation_result: ModelEvaluationResult,
                                    product_code: str,
                                    store_code: Optional[str] = None) -> Dict[str, str]:
        """
        创建综合评估报告
        """
        plot_paths = {}
        try:
            # 1. 预测对比图
            plot_paths['prediction_comparison'] = self.create_prediction_comparison_chart(
                y_true=evaluation_result.y_true,
                y_pred=evaluation_result.y_pred,
                product_code=product_code,
                store_code=store_code
            )

            # 2. 指标卡片
            plot_paths['metrics_card'] = self.create_metrics_card(
                metrics=evaluation_result.metrics,
                product_code=product_code
            )

        except Exception as e:
            print(f"创建综合报告失败: {e}")

        return plot_paths

    def _generate_strategy_visualizations_with_pil(self, strategy_id: str, pricing_schedule: List,
                                                   product_info: Dict, features: Dict,
                                                   evaluation: Dict, training_history,
                                                   total_sales: int, total_profit: float,
                                                   confidence_score: float) -> Dict[str, str]:
        """使用PIL生成定价策略可视化图表"""

        plot_paths = {}
        try:
            # 1. 创建策略报告主图像
            img_width = 1200
            img_height = 1600  # 增加高度以容纳更多内容

            # 创建白色背景图像
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)

            def get_font_path():
                system = platform.system()

                # 优先尝试的字体列表（按优先级排序）
                font_candidates = []

                if system == "Windows":
                    font_candidates = [
                        r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑（支持货币符号）
                        r"C:\Windows\Fonts\msyhbd.ttc",  # 微软雅黑粗体
                        r"C:\Windows\Fonts\Arial.ttf",  # Arial（支持货币符号）
                        r"C:\Windows\Fonts\Arial Unicode MS.ttf",  # Arial Unicode MS（支持最全）
                        r"C:\Windows\Fonts\Segoe UI.ttf",  # Segoe UI
                        r"C:\Windows\Fonts\YuGothM.ttc",  # 游哥特体
                    ]
                elif system == "Darwin":  # macOS
                    font_candidates = [
                        "/System/Library/Fonts/PingFang.ttc",  # 苹方
                        "/System/Library/Fonts/AppleGothic.ttf",  # 苹果哥特体
                        "/Library/Fonts/Arial Unicode.ttf",  # Arial Unicode
                        "/System/Library/Fonts/Helvetica.ttc",  # 赫尔维蒂卡
                    ]
                else:  # Linux
                    font_candidates = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
                    ]

                # 检查字体文件是否存在
                for font_path in font_candidates:
                    if os.path.exists(font_path):
                        return font_path

                return None

            # 加载字体
            font_path = get_font_path()

            if font_path:
                # 使用不同大小的字体
                font_title = ImageFont.truetype(font_path, 40)
                font_subtitle = ImageFont.truetype(font_path, 28)
                font_normal = ImageFont.truetype(font_path, 24)
                font_small = ImageFont.truetype(font_path, 20)
            else:
                # 如果找不到系统字体，可以下载字体文件到本地
                print("警告：未找到系统字体，将尝试在线下载或使用备用字体")
                # 这里可以添加下载字体的代码
                font_title = ImageFont.load_default()
                font_subtitle = ImageFont.load_default()
                font_normal = ImageFont.load_default()
                font_small = ImageFont.load_default()
            # 绘制标题
            # title = f"定价策略分析报告 - {strategy_id}"
            title = f"定价策略分析报告"
            draw.text((img_width // 2, 40), title, fill='darkblue', font=font_title, anchor='mm')

            # 绘制基本信息
            info_y = 100
            draw.text((50, info_y), f"商品名称: {product_info.get('product_name', '未知商品')}", fill='black',
                      font=font_normal)
            draw.text((50, info_y + 30), f"商品编码: {product_info.get('product_code', 'N/A')}", fill='black',
                      font=font_normal)
            draw.text((50, info_y + 60), f"原价: ¥{product_info.get('original_price', 0):.2f}", fill='black',
                      font=font_normal)
            draw.text((50, info_y + 90), f"成本价: ¥{product_info.get('cost_price', 0):.2f}", fill='black',
                      font=font_normal)
            draw.text((50, info_y + 120), f"初始库存: {product_info.get('initial_stock', 0)}", fill='black',
                      font=font_normal)

            # 绘制右侧信息
            right_x = img_width - 300
            draw.text((right_x, info_y), f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fill='darkblue',
                      font=font_small)
            draw.text((right_x, info_y + 25), f"预期总销量: {total_sales}", fill='darkgreen', font=font_small)
            draw.text((right_x, info_y + 50), f"预期总利润: ¥{total_profit:.2f}", fill='darkgreen', font=font_small)
            draw.text((right_x, info_y + 75), f"售罄概率: {evaluation.get('sell_out_probability', 0) * 100:.1f}%",
                      fill='darkgreen', font=font_small)
            draw.text((right_x, info_y + 100), f"策略置信度: {confidence_score * 100:.1f}%", fill='darkgreen',
                      font=font_small)

            # 2. 绘制定价方案表格
            # table_y = 200
            table_y = 300
            # draw.text((img_width//2, table_y), "定价方案明细", fill='darkred', font=font_subtitle, anchor='mm')
            draw.text((50, table_y), "定价方案明细", fill='darkred', font=font_subtitle)
            # 绘制表格标题
            table_start_y = table_y + 40
            headers = ["时间段", "价格", "折扣", "预期销量", "预期利润"]
            col_widths = [200, 100, 100, 150, 150]
            col_x = 50

            # 绘制表头
            for i, header in enumerate(headers):
                draw.rectangle([col_x, table_start_y, col_x + col_widths[i], table_start_y + 40],
                               fill='lightgray', outline='black')
                draw.text((col_x + col_widths[i] // 2, table_start_y + 20), header,
                          fill='black', font=font_normal, anchor='mm')
                col_x += col_widths[i]

            # 绘制表格内容
            row_y = table_start_y + 40
            for i, segment in enumerate(pricing_schedule):
                if i >= 8:  # 最多显示8行
                    break

                col_x = 50
                row_data = [
                    f"{segment.start_time}-{segment.end_time}",
                    f"¥{segment.price:.2f}",
                    f"{round((1 - segment.discount) * 100, 1)}%",
                    f"{segment.expected_sales}",
                    f"¥{segment.profit:.2f}"
                ]

                for j, cell in enumerate(row_data):
                    # 交替行背景色
                    fill_color = 'white' if i % 2 == 0 else 'lightblue'
                    draw.rectangle([col_x, row_y, col_x + col_widths[j], row_y + 30],
                                   fill=fill_color, outline='black')
                    draw.text((col_x + col_widths[j] // 2, row_y + 15), cell,
                              fill='black', font=font_small, anchor='mm')
                    col_x += col_widths[j]

                row_y += 30

            # 3. 绘制价格变化趋势图
            chart_y = row_y + 50

            # 创建价格变化图
            fig = self._create_price_trend_chart(pricing_schedule)
            if fig:
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                chart_img = Image.open(buf)

                chart_width = 600
                chart_height = 350
                chart_img = chart_img.resize((chart_width, chart_height), Image.Resampling.LANCZOS)
                img.paste(chart_img, (img_width // 2 - chart_width // 2, chart_y))
                buf.close()

            # 4. 绘制模型性能指标
            metrics_y = chart_y + 400
            if training_history and training_history.performance_metrics:
                metrics = training_history.performance_metrics

                draw.text((50, metrics_y), "模型性能指标:", fill='darkgreen', font=font_subtitle)

                # 绘制指标卡片
                card_width = 200
                card_height = 80
                card_spacing = 220

                metrics_list = [
                    ("R²得分", f"{metrics.r2:.3f}", "lightgreen" if metrics.r2 > 0.7 else "lightyellow"),
                    ("RMSE", f"{metrics.rmse:.3f}", "lightblue"),
                    ("MAE", f"{metrics.mae:.3f}", "lightcoral"),
                    ("MAPE", f"{metrics.mape:.1f}%", "lightpink")
                ]

                for i, (name, value, color) in enumerate(metrics_list):
                    card_x = 50 + (i % 2) * card_spacing
                    card_y = metrics_y + 40 + (i // 2) * (card_height + 20)

                    # 绘制指标卡片
                    draw.rectangle([card_x, card_y, card_x + card_width, card_y + card_height],
                                   fill=color, outline='gray', width=2)

                    # 绘制指标名称
                    draw.text((card_x + card_width // 2, card_y + 20), name,
                              fill='black', font=font_normal, anchor='mm')

                    # 绘制指标值
                    draw.text((card_x + card_width // 2, card_y + 50), value,
                              fill='darkblue', font=font_title, anchor='mm')

            # 5. 绘制特征使用情况
            # features_y = metrics_y + 200
            features_y = metrics_y + 260
            if features:
                draw.text((50, features_y), "使用的主要特征:", fill='darkgreen', font=font_subtitle)

                # 显示前10个特征
                feature_keys = list(features.keys())[:10]
                feature_y = features_y + 40

                for i, key in enumerate(feature_keys):
                    if i >= 10:
                        break

                    # 每行显示两个特征
                    col = i % 2
                    row = i // 2

                    x_pos = 50 + col * 350
                    y_pos = feature_y + row * 30

                    value = features[key]
                    if isinstance(value, float):
                        value_str = f"{value:.3f}"
                    else:
                        value_str = str(value)

                    # 截断过长的特征名
                    display_key = key[:25] + "..." if len(key) > 25 else key

                    draw.text((x_pos, y_pos), f"• {display_key}: {value_str}",
                              fill='black', font=font_small)

            # 6. 绘制策略建议
            # advice_y = features_y + 200
            advice_y = features_y + 250
            draw.rectangle([40, advice_y, img_width - 40, advice_y + 120],
                           fill='lightyellow', outline='gold', width=2)

            draw.text((img_width // 2, advice_y + 20), "策略建议",
                      fill='darkred', font=font_subtitle, anchor='mm')

            # 根据售罄概率给出建议
            sell_out_prob = evaluation.get('sell_out_probability', 0)
            if sell_out_prob > 0.9:
                advice = "售罄概率较高，建议适当提高价格或增加库存"
                advice_color = 'red'
            elif sell_out_prob > 0.7:
                advice = "售罄概率适中，当前定价策略较为合理"
                advice_color = 'orange'
            else:
                advice = "售罄概率较低，建议加大折扣力度促进销售"
                advice_color = 'blue'

            draw.text((img_width // 2, advice_y + 60), advice,
                      fill=advice_color, font=font_normal, anchor='mm')

            # 根据置信度给出建议
            if confidence_score > 0.8:
                conf_advice = "策略置信度高，可放心执行"
            elif confidence_score > 0.6:
                conf_advice = "策略置信度中等，建议监控执行效果"
            else:
                conf_advice = "策略置信度较低，建议谨慎执行并准备备选方案"

            draw.text((img_width // 2, advice_y + 90), conf_advice,
                      fill='darkgreen', font=font_small, anchor='mm')

            # 7. 绘制页脚
            footer_y = img_height - 50
            draw.text((img_width // 2, footer_y), f"定价策略分析系统 - {strategy_id}",
                      fill='gray', font=font_small, anchor='mm')
            draw.text((img_width // 2, footer_y + 25), f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      fill='gray', font=font_small, anchor='mm')

            # 保存图像
            filename = f"strategy_report_{strategy_id.replace(':', '')}.png"
            filepath = os.path.join(self.viz_output_dir, filename)
            img.save(filepath, 'PNG')

            plot_paths['strategy_report'] = filepath
            print(f"策略报告已保存到: {filepath}")

        except Exception as e:
            print(f"使用PIL生成策略可视化图表失败: {e}")
            import traceback
            traceback.print_exc()

        return plot_paths

    def _create_price_trend_chart(self, pricing_schedule: List):
        """创建价格变化趋势图"""
        try:
            times = []
            prices = []
            discounts = []

            for segment in pricing_schedule:
                # 解析时间
                start_time = datetime.strptime(segment.start_time, "%H:%M")
                end_time = datetime.strptime(segment.end_time, "%H:%M")
                mid_time = start_time + (end_time - start_time) / 2

                # times.append(mid_time.strftime("%H:%M"))
                times.append(mid_time)
                prices.append(segment.price)
                discounts.append((1 - segment.discount) * 100)  # 转换为折扣百分比

            fig, ax1 = plt.subplots(figsize=(8, 4))

            # 绘制价格折线图
            color1 = 'tab:blue'
            ax1.set_xlabel('时间段', fontsize=18)
            ax1.set_ylabel('价格 (元)', color=color1, fontsize=18)
            ax1.plot(times, prices, 'o-', color=color1, linewidth=2, markersize=18)
            ax1.tick_params(axis='y', labelcolor=color1)

            # 创建第二个Y轴用于折扣
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('折扣 (%)', color=color2, fontsize=18)
            ax2.plot(times, discounts, 's--', color=color2, linewidth=2, markersize=18)
            ax2.tick_params(axis='y', labelcolor=color2)

            ax1.set_title('价格与折扣变化趋势', fontsize=22, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # 格式化x轴显示
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # 旋转X轴标签
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"创建价格趋势图失败: {e}")
            return None
