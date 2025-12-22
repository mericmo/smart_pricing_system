# main.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
from models.demand_predictor import EnhancedDemandPredictor as DemandPredictor
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.calender_helper import create_china_holidays_from_date_list
from core.pricing_strategy_generator import EnhancedPricingStrategyGenerator
from core.config import *
def load_data_files(transaction_file: str, 
                   weather_file: str = None,
                   calendar_file: str = None) -> tuple:
    """加载数据文件"""
    
    print("正在加载数据文件...")
    
    # 1. 加载交易数据
    try:
        transaction_data = pd.read_csv(transaction_file, encoding='utf-8', parse_dates=['日期', "交易时间"], dtype={"商品编码": str, "门店编码": str})
        print(f"交易数据加载成功: {len(transaction_data)} 条记录")
    except Exception as e:
        print(f"加载交易数据失败: {e}")
        return None, None, None
    
    # 2. 加载天气数据（可选）

    weather_data = None
    if weather_file and os.path.exists(weather_file):
        try:
            weather_data = pd.read_csv(weather_file, encoding='utf-8', parse_dates=['date'])
            print(f"天气数据加载成功: {len(weather_data)} 条记录")
        except Exception as e:
            print(f"加载天气数据失败: {e}")
    
    # 3. 加载日历数据（可选）
    calendar_data = None
    date_series = transaction_data['日期'].unique()
    calendar_data = create_china_holidays_from_date_list(date_series=date_series)
    # if calendar_file and os.path.exists(calendar_file):
    #     try:
    #         calendar_data = pd.read_csv(calendar_file, encoding='utf-8')
    #         print(f"日历数据加载成功: {len(calendar_data)} 条记录")
    #     except Exception as e:
    #         print(f"加载日历数据失败: {e}")
    
    return transaction_data, weather_data, calendar_data

def create_sample_data() -> tuple:
    """创建示例数据"""
    print("创建示例数据...")
    
    # 创建交易数据示例
    np.random.seed(42)
    
    dates = pd.date_range('2024-12-01', '2024-12-31', freq='D')
    product_codes = ['4834512', '4701098', '5012345', '5123456']
    product_names = ['福荫川式豆花380g', '福荫韧豆腐380g', '鲜奶面包500g', '酸奶200ml']
    prices = [7.99, 5.99, 12.99, 8.99]
    categories = ['20010101', '20010101', '20020101', '20030101']
    
    records = []
    
    for date in dates:
        for i, (product_code, product_name, price, category) in enumerate(
            zip(product_codes, product_names, prices, categories)):
            
            # 每天生成3-8条交易记录
            num_transactions = np.random.randint(3, 9)
            
            for _ in range(num_transactions):
                # 交易时间在8:00-22:00之间
                hour = np.random.randint(8, 22)
                minute = np.random.randint(0, 60)
                
                # 折扣概率（晚上20点后概率更高）
                if hour >= 20:
                    discount_prob = 0.6
                else:
                    discount_prob = 0.2
                
                if np.random.random() < discount_prob:
                    discount = np.random.choice([0.7, 0.8, 0.9])
                    discount_type = f"促销{discount*10}折"
                else:
                    discount = 1.0
                    discount_type = "n-无折扣促销"
                
                # 销售数量
                if discount < 1.0:
                    quantity = np.random.randint(1, 4)  # 促销时买的多
                else:
                    quantity = np.random.randint(1, 2)
                
                # 计算金额
                sales_amount = price * discount * quantity
                discount_amount = price * quantity - sales_amount if discount < 1.0 else 0
                
                record = {
                    '日期': date.strftime('%Y/%m/%d'),
                    '门店编码': '205625',
                    '流水单号': f"205625P{np.random.randint(100, 999)}{date.strftime('%Y%m%d')}{np.random.randint(1000, 9999)}",
                    '会员id': f"{np.random.randint(1000000000000000000, 9999999999999999999)}" if np.random.random() < 0.5 else None,
                    '交易时间': f"{date.strftime('%Y/%m/%d')} {hour:02d}:{minute:02d}",
                    '渠道名称': '线下销售',
                    '平台触点名称': np.random.choice(['智能POS人工', '智能POS自助']),
                    '小类编码': category,
                    '商品编码': product_code,
                    '商品名称': product_name,
                    '售价': price,
                    '折扣类型': discount_type,
                    '税率': 13,
                    '销售数量': quantity,
                    '销售金额': round(sales_amount, 2),
                    '销售净额': round(sales_amount * 0.87, 2),  # 扣除税
                    '折扣金额': round(discount_amount, 2)
                }
                
                records.append(record)
    
    transaction_data = pd.DataFrame(records)
    
    # 创建天气数据示例
    weather_records = []
    for date in dates:
        weather_records.append({
            'date': date.strftime('%Y/%m/%d'),
            'text_day': np.random.choice(['晴', '多云', '阴', '阵雨']),
            'code_day': np.random.choice(['01', '02', '03', '04']),
            'text_night': np.random.choice(['晴', '多云', '阴']),
            'code_night': np.random.choice(['01', '02', '03']),
            'high': np.random.randint(20, 35),
            'low': np.random.randint(15, 25)
        })
    
    weather_data = pd.DataFrame(weather_records)
    
    # 创建日历数据示例
    calendar_records = []
    for date in dates:
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday = 0
        holiday_name = None
        
        # 模拟节假日
        if date.day in [1, 2, 3]:  # 月初几天模拟元旦
            is_holiday = 1
            holiday_name = '元旦'
        elif date.day in [24, 25, 26]:  # 模拟圣诞节
            is_holiday = 1
            holiday_name = '圣诞节'
        
        calendar_records.append({
            'date': date.strftime('%Y/%m/%d'),
            'is_holiday': is_holiday,
            'is_weekend': is_weekend,
            'holiday_name': holiday_name,
            'special_event': '双12' if date.day == 12 else None
        })
    
    calendar_data = pd.DataFrame(calendar_records)
    
    print(f"示例数据创建完成: {len(transaction_data)} 条交易记录")
    
    return transaction_data, weather_data, calendar_data

def main():
    """主函数"""
    
    print("=" * 70)
    print("智能打折促销系统 - 日清品阶梯定价优化")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n1. 数据加载")
    print("-" * 40)
    
    # 尝试加载实际数据文件
    transaction_file = "data/historical_transactions.csv"
    weather_file = "data/weather_info.csv"
    calendar_file = "data/calender_info.csv"
    
    if os.path.exists(transaction_file):
        transaction_data, weather_data, calendar_data = load_data_files(
            transaction_file, weather_file, calendar_file
        )
    else:
        print("未找到数据文件，使用示例数据")
        transaction_data, weather_data, calendar_data = create_sample_data()
    
    if transaction_data is None:
        print("无法加载数据，程序退出")
        return
    
    # 2. 初始化系统
    print("\n2. 系统初始化")
    print("-" * 40)
    # 获取用户输入
    product_code = "8006144"  # input("\n请输入商品编码: ").strip()
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        config['product_code'] = product_code
        strategy_generator = EnhancedPricingStrategyGenerator(
            transaction_data=transaction_data,
            weather_data=weather_data,
            calendar_data=calendar_data,
            config=config  # 可以使用配置管理器
        )
        print("系统初始化成功")
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return
    
    # 3. 用户输入
    print("\n3. 定价参数输入")
    print("-" * 40)
    
    # 显示可用的商品
    available_products = transaction_data['商品编码'].unique()[:5]  # 显示前5个
    available_product_names = transaction_data.drop_duplicates('商品编码').set_index('商品编码')['商品名称'].to_dict()
    
    print("可选的商品编码:")
    for code in available_products:
        name = available_product_names.get(code, '未知商品')
        avg_price = transaction_data[transaction_data['商品编码'] == code]['售价'].mean()
        print(f"  {code}: {name} (平均价格: ¥{avg_price:.2f})")
    

    
    # 验证商品编码
    if product_code not in transaction_data['商品编码'].values:
        print(f"警告: 商品编码 {product_code} 不在数据中，将继续使用")
    
    try:
        initial_stock = 5 # int(input("请输入当前库存数量: "))
        # original_price = float(input("请输入商品原价: "))
        # cost_price = float(input("请输入商品成本价: "))
    except ValueError:
        print("输入错误，请重新运行程序")
        return
    
    # 促销设置
    print("\n促销设置:")
    promotion_start = "20:00" #input("开始时间 (HH:MM, 默认20:00): ").strip() or "20:00"
    promotion_end = "22:00" #input("结束时间 (HH:MM, 默认22:00): ").strip() or "22:00"
    
    # 折扣范围
    print("\n折扣范围:")
    min_discount = 0.4 # float(input("最低折扣 (如0.4表示4折): ") or "0.4")
    max_discount = 0.8 # float(input("最高折扣 (如0.9表示9折): ") or "0.9")
    
    # 时间段
    time_segments = 2 # int(input("\n时间段数量 (默认4): ") or "4")
    
    # 门店编码（可选）
    store_code = "205625" #input("门店编码 (可选，直接回车跳过): ").strip() or None
    
    # 是否使用外部数据
    use_weather = True # input("\n是否使用天气数据? (y/n, 默认y): ").strip().lower() in ['y', '']
    use_calendar = True # input("是否使用日历数据? (y/n, 默认y): ").strip().lower() in ['y', '']
    
    # 4. 生成策略
    print(f"\n4. 正在生成定价策略...")
    print("-" * 40)
    
    try:
        strategy = strategy_generator.generate_pricing_strategy(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount,
            time_segments=time_segments,
            store_code=store_code,
            current_time=pd.to_datetime('2025-10-31 10:21:25'),
            use_weather=use_weather,
            use_calendar=use_calendar,
            generate_visualizations=True
        )
    except Exception as e:
        print(f"生成策略失败: {e}")
        return
    
    # 5. 显示结果
    print("\n" + "=" * 70)
    print("定价策略生成完成!")
    print("=" * 70)
    
    # 基本信息
    print(f"\n策略ID: {strategy.strategy_id}")
    print(f"生成时间: {strategy.generated_time}")
    print(f"置信度: {strategy.confidence_score:.1%}")
    
    # 商品信息
    print(f"\n商品信息:")
    print(f"  商品编码: {strategy.product_code}")
    print(f"  商品名称: {strategy.product_name}")
    print(f"  原价: ¥{strategy.original_price:.2f}")
    print(f"  成本价: ¥{strategy.cost_price:.2f}")
    print(f"  初始库存: {strategy.initial_stock}件")
    
    # 促销设置
    print(f"\n促销设置:")
    print(f"  促销时段: {strategy.promotion_start} - {strategy.promotion_end}")
    print(f"  折扣范围: {strategy.min_discount:.1%} - {strategy.max_discount:.1%}")
    print(f"  时间段数: {strategy.time_segments}")
    print(f"  使用天气数据: {'是' if strategy.weather_consideration else '否'}")
    print(f"  使用日历数据: {'是' if strategy.calendar_consideration else '否'}")
    
    # 阶梯定价方案
    print(f"\n阶梯定价方案:")
    print("-" * 85)
    print(f"{'时间段':<18} {'折扣':<12} {'价格':<10} {'预期销量':<12} {'预期收入':<12} {'预期利润':<12}")
    print("-" * 85)
    
    total_sales = 0
    total_revenue = 0
    total_profit = 0
    
    for stage in strategy.pricing_schedule:
        print(f"{stage['start_time']}-{stage['end_time']:<18} "
              f"{stage['discount_percentage']:<12} "
              f"¥{stage['price']:<9.2f} "
              f"{stage['expected_sales']:<12} "
              f"¥{stage['expected_revenue']:<11.2f} "
              f"¥{stage['expected_profit']:<11.2f}")
        
        total_sales += stage['expected_sales']
        total_revenue += stage['expected_revenue']
        total_profit += stage['expected_profit']
    
    print("-" * 85)
    print(f"{'总计':<18} {'':<12} {'':<10} "
          f"{total_sales:<12} "
          f"¥{total_revenue:<11.2f} "
          f"¥{total_profit:<11.2f}")
    
    # 方案评估
    print(f"\n方案评估:")
    eval_result = strategy.evaluation
    print(f"  预期总销量: {eval_result['total_expected_sales']}件")
    print(f"  预期总收入: ¥{eval_result['total_revenue']:.2f}")
    print(f"  预期总利润: ¥{eval_result['total_profit']:.2f}")
    print(f"  剩余库存: {eval_result['remaining_stock']}件")
    print(f"  售罄概率: {eval_result['sell_out_probability']:.1%}")
    print(f"  利润率: {eval_result['profit_margin']:.1%}")
    print(f"  平均折扣: {eval_result['average_discount']:.1%}")
    print(f"  库存清空率: {eval_result['stock_clearance_rate']:.1%}")
    if 'recommendation' in eval_result:
        print(f"  推荐建议: {eval_result['recommendation']}")

    # 访问可视化结果
    if strategy.visualization_paths:
        print(f"策略报告: {strategy.visualization_paths.get('strategy_report')}")
        print(f"训练报告: {strategy.model_performance.get('plot_paths', {}).get('training_report')}")
    # print(f"可视化输出目录: {strategy_generator.viz_output_dir}")
    # print(f"目录是否存在: {os.path.exists(strategy_generator.viz_output_dir)}")

    # 6. 保存策略
    print("\n6. 策略保存")
    print("-" * 40)
    
    save_option = 'n' #input("是否保存策略到文件? (y/n): ").strip().lower()
    if save_option == 'y':
        # 创建输出目录
        output_dir = "output/strategies"
        os.makedirs(output_dir, exist_ok=True)
        safe_strategy_id = strategy.strategy_id.replace(':', '_')
        filename = f"{safe_strategy_id}.json"
        # filename = f"{strategy.strategy_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        strategy_generator.save_strategy(strategy, filepath)
        print(f"策略已保存到: {filepath}")
        
        # 同时保存为CSV格式（便于查看）
        csv_filepath = filepath.replace('.json', '.csv')
        schedule_df = pd.DataFrame(strategy.pricing_schedule)
        schedule_df.to_csv(csv_filepath, index=False, encoding='utf-8')
        print(f"定价方案CSV已保存到: {csv_filepath}")
    
    # 7. 导出执行计划
    print("\n7. 执行计划导出")
    print("-" * 40)
    
    export_option = 'n' # input("是否导出执行计划表? (y/n): ").strip().lower()
    if export_option == 'y':
        execution_plan = []
        # current_time = datetime.strptime(promotion_start, "%H:%M")
        
        for stage in strategy.pricing_schedule:
            # 解析时间段
            # start_hour, start_minute = map(int, stage['start_time'].split(':'))
            # end_hour, end_minute = map(int, stage['end_time'].split(':'))
            
            # 创建执行计划项
            plan_item = {
                '执行时间': stage['start_time'],
                '结束时间': stage['end_time'],
                '折扣力度': stage['discount_percentage'],
                '执行价格': f"¥{stage['price']:.2f}",
                '价签更新': "是",
                '系统同步': "是",
                '负责人': "店长/值班经理",
                '检查时间': stage['end_time'],
                '备注': f"预期销量: {stage['expected_sales']}件"
            }
            
            execution_plan.append(plan_item)
        
        # 保存执行计划
        execution_df = pd.DataFrame(execution_plan)
        execution_file = f"output/execution_plan_{strategy.strategy_id.replace(':', '_')}.csv"
        os.makedirs("output", exist_ok=True)
        execution_df.to_csv(execution_file, index=False, encoding='utf-8')
        
        print(f"执行计划已保存到: {execution_file}")
        print("\n执行计划预览:")
        print(execution_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("程序执行完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()