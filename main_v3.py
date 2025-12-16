# main.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from core.config_v1 import ConfigManager
from core.pricing_strategy_generator import PricingStrategyGenerator
from core.real_time_adjuster import RealTimeAdjuster, SalesUpdate
from core.emergency_controller import EmergencyClearanceController
def load_transaction_data(filepath: str) -> pd.DataFrame:
    """加载交易数据"""

    # 读取CSV文件
    df = pd.read_csv(filepath, encoding='utf-8',parse_dates=["日期", "交易时间"], dtype={"商品编码":str,"门店编码":str})

    # 确保必要的列存在
    required_columns = ['商品编码', '商品名称', '售价', '销售数量', '销售金额']
    for col in required_columns:
        if col not in df.columns:
            print(f"警告：缺少必要列 {col}")

    return df

def main():
    """主函数 - 出清优先版本"""
    
    print("=" * 70)
    print("智能打折促销系统 - 日清品出清优先定价")
    print("=" * 70)
    print("核心原则：优先保障商品售罄，在可出清前提下优化利润")
    print()
    
    # 1. 加载数据
    print("1. 加载交易数据...")
    transaction_data = load_transaction_data()
    
    # 2. 初始化系统
    print("2. 初始化系统组件...")
    config_manager = ConfigManager()
    strategy_generator = PricingStrategyGenerator(transaction_data, config_manager)
    emergency_controller = EmergencyClearanceController(config_manager)
    
    # 3. 用户输入
    print("\n3. 请输入商品信息和促销参数:")
    print("-" * 50)
    
    product_code = input("   商品编码: ").strip()
    initial_stock = int(input("   当前库存数量: "))
    
    # 获取商品信息
    product_info = strategy_generator._get_product_info(product_code)
    print(f"   商品名称: {product_info.get('product_name', '未知')}")
    print(f"   原价: ¥{product_info['original_price']:.2f}")
    print(f"   成本价: ¥{product_info['cost_price']:.2f}")
    
    print("\n   促销时间段设置:")
    promotion_start = input("   开始时间 (HH:MM, 默认20:00): ").strip() or "20:00"
    promotion_end = input("   结束时间 (HH:MM, 默认22:00): ").strip() or "22:00"
    
    print("\n   折扣范围设置:")
    min_discount = float(input("   最低折扣 (如0.4表示4折): ") or "0.4")
    max_discount = float(input("   最高折扣 (如0.9表示9折): ") or "0.9")
    
    # 4. 生成出清策略
    print(f"\n4. 正在生成出清优先定价策略...")
    print("   原则：优先保障 {promotion_end} 前全部售罄")
    
    strategy = strategy_generator.generate_clearance_strategy(
        product_code=product_code,
        initial_stock=initial_stock,
        promotion_start=promotion_start,
        promotion_end=promotion_end,
        min_discount=min_discount,
        max_discount=max_discount
    )
    
    # 5. 显示结果
    display_clearance_strategy(strategy)
    
    # 6. 显示可行性分析
    display_feasibility_analysis(strategy['feasibility'])
    
    # 7. 显示建议
    display_recommendations(strategy['recommendations'])
    
    # 8. 备选策略
    if strategy['feasibility']['clearance_probability'] < 0.7:
        print("\n⚠️  出清概率较低，建议考虑备选方案:")
        fallback_strategies = strategy_generator.generate_fallback_strategies(
            product_code, initial_stock, promotion_start, promotion_end,
            min_discount, max_discount
        )
        display_fallback_strategies(fallback_strategies)
    
    # 9. 监控选项
    enable_monitoring = input("\n是否启用实时监控？(y/n): ").strip().lower()
    if enable_monitoring == 'y':
        simulate_real_time_monitoring(strategy, emergency_controller)
    
    print("\n" + "=" * 70)
    print("程序执行完成！")
    print("=" * 70)

def display_clearance_strategy(strategy: Dict):
    """显示出清策略"""
    
    print("\n" + "=" * 70)
    print("出清优先定价策略")
    print("=" * 70)
    
    print(f"\n商品信息:")
    print(f"  商品编码: {strategy['product_code']}")
    print(f"  商品名称: {strategy['product_name']}")
    print(f"  原价: ¥{strategy['original_price']:.2f}")
    print(f"  成本价: ¥{strategy['cost_price']:.2f}")
    print(f"  初始库存: {strategy['initial_stock']}件")
    
    print(f"\n促销设置:")
    print(f"  促销时段: {strategy['promotion_start']} - {strategy['promotion_end']}")
    print(f"  折扣范围: {strategy['min_discount']:.1%} - {strategy['max_discount']:.1%}")
    print(f"  策略类型: {strategy['strategy_type']}")
    
    print(f"\n阶梯定价方案 (出清优先级):")
    print("-" * 90)
    print(f"{'时间段':<18} {'折扣':<10} {'价格':<10} {'预期销量':<10} {'出清优先级':<12} {'紧迫程度':<10}")
    print("-" * 90)
    
    for stage in strategy['pricing_schedule']:
        clearance_priority = stage['clearance_priority']
        urgency_level = stage['urgency_level']
        
        # 可视化优先级
        priority_bar = "▓" * int(clearance_priority * 10) + "░" * (10 - int(clearance_priority * 10))
        urgency_bar = "●" * int(urgency_level * 5) + "○" * (5 - int(urgency_level * 5))
        
        print(f"{stage['start_time']}-{stage['end_time']:<18} "
              f"{stage['discount_percentage']:<10} "
              f"¥{stage['price']:<9.2f} "
              f"{stage['expected_sales']:<10} "
              f"{priority_bar:<12} "
              f"{urgency_bar:<10}")
    
    print("-" * 90)
    
    # 显示评估结果
    eval_result = strategy['evaluation']
    print(f"\n策略评估:")
    print(f"  预期总销量: {eval_result['total_expected_sales']}件")
    print(f"  剩余库存: {eval_result['remaining_stock']}件")
    print(f"  售罄率: {eval_result['clearance_rate']:.1%}")
    
    if eval_result['success']:
        print(f"  ✅ 预计能成功出清 (>{config_manager.clearance_config.clearance_threshold:.0%})")
    else:
        print(f"  ⚠️  出清概率较低 (<{config_manager.clearance_config.clearance_threshold:.0%})")
    
    print(f"  预期总收入: ¥{eval_result['total_revenue']:.2f}")
    print(f"  预期总利润: ¥{eval_result['total_profit']:.2f}")
    print(f"  利润率: {eval_result['profit_margin']:.1%}")
    print(f"  预计售罄时间: {eval_result.get('expected_clearance_time', '未知')}")

def display_feasibility_analysis(feasibility: Dict):
    """显示可行性分析"""
    
    print("\n" + "=" * 70)
    print("出清可行性分析")
    print("=" * 70)
    
    clearance_prob = feasibility['clearance_probability']
    
    # 可视化概率
    prob_bar_length = 20
    filled_length = int(clearance_prob * prob_bar_length)
    prob_bar = "█" * filled_length + "░" * (prob_bar_length - filled_length)
    
    print(f"\n出清概率: {clearance_prob:.1%}")
    print(f"  [{prob_bar}]")
    
    if clearance_prob >= 0.8:
        print("  ✅ 出清可行性高")
    elif clearance_prob >= 0.6:
        print("  ⚠️  出清可行性中等")
    elif clearance_prob >= 0.4:
        print("  ⚠️  出清可行性较低")
    else:
        print("  ❌ 出清可行性低")
    
    print(f"\n详细分析:")
    print(f"  初始库存: {feasibility['initial_stock']}件")
    print(f"  促销时长: {feasibility['promotion_duration_hours']:.1f}小时")
    print(f"  最大可能销量: {feasibility['max_possible_sales']}件")
    print(f"  所需销售速率: {feasibility['required_sales_rate']:.1f}件/小时")
    print(f"  可能销售速率: {feasibility['possible_sales_rate']:.1f}件/小时")
    print(f"  库存压力系数: {feasibility['stock_pressure']:.2f}")
    print(f"  时间压力系数: {feasibility['time_pressure']:.2f}")
    print(f"  价格效应系数: {feasibility['price_effect']:.2f}")

def display_recommendations(recommendations: List[str]):
    """显示建议"""
    
    print("\n" + "=" * 70)
    print("优化建议")
    print("=" * 70)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

def display_fallback_strategies(fallback_strategies: Dict):
    """显示备选策略"""
    
    print("\n备选促销方案:")
    print("-" * 60)
    
    for strategy_type, strategy in fallback_strategies.items():
        if strategy_type == 'recommendation':
            continue
            
        print(f"\n{strategy['strategy_type'].upper()}策略:")
        print(f"  描述: {strategy['description']}")
        print(f"  预期售罄率: {strategy['expected_clearance_rate']:.0%}")
        
        if 'effective_discount' in strategy:
            print(f"  实际折扣: {strategy['effective_discount']:.1%}")
        
        print(f"  优势: {', '.join(strategy['advantages'])}")
        print(f"  劣势: {', '.join(strategy['disadvantages'])}")
    
    print(f"\n推荐方案: {fallback_strategies['recommendation']['selected_strategy']['description']}")
    print(f"推荐理由: {fallback_strategies['recommendation']['reason']}")

def simulate_real_time_monitoring(strategy: Dict, emergency_controller):
    """模拟实时监控"""
    
    print("\n" + "=" * 70)
    print("实时监控模拟")
    print("=" * 70)
    
    strategy_id = strategy['strategy_id']
    promotion_end = strategy['promotion_end']
    
    # 模拟几个时间点的销售
    time_points = [
        (datetime.strptime("20:30", "%H:%M"), 15, 85),   # 30分钟后
        (datetime.strptime("21:00", "%H:%M"), 25, 60),   # 1小时后
        (datetime.strptime("21:30", "%H:%M"), 20, 40),   # 1.5小时后
        (datetime.strptime("21:45", "%H:%M"), 10, 30),   # 1.75小时后
    ]
    
    for i, (current_time, sales, remaining) in enumerate(time_points):
        print(f"\n检查点 {i+1}: {current_time.strftime('%H:%M')}")
        print(f"  实际销售: {sales}件")
        print(f"  剩余库存: {remaining}件")
        
        # 监控进度
        monitoring_result = emergency_controller.monitor_sales_progress(
            strategy_id=strategy_id,
            current_time=current_time,
            actual_sales=sales,
            remaining_stock=remaining,
            promotion_end=promotion_end
        )
        
        progress = monitoring_result['progress_analysis']
        print(f"  销售进度: {progress['status']} (比率: {progress['progress_rate']})")
        print(f"  销售速率: {progress['sales_rate']}件/小时")
        print(f"  所需速率: {progress['required_rate']}件/小时")
        print(f"  剩余时间: {progress['time_to_close']:.1f}小时")
        
        if monitoring_result['adjustment_needed']:
            print(f"  ⚠️  需要调整: {monitoring_result['adjustment_type']}")
            for rec in monitoring_result['recommendations']:
                print(f"    建议: {rec}")
            
            # 模拟调整
            if i < len(time_points) - 1:  # 不是最后一次
                adjust = input("    是否执行调整？(y/n): ").strip().lower()
                if adjust == 'y':
                    print("    ✅ 已执行调整")
        else:
            print(f"  ✅ 进度正常，无需调整")
        
        print("-" * 50)
    
    # 显示监控摘要
    summary = emergency_controller.get_monitoring_summary(strategy_id)
    print(f"\n监控摘要:")
    print(f"  总检查点: {summary['checkpoints']}")
    print(f"  总调整次数: {summary['adjustments_made']}")