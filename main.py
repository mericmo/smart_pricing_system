# main.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from core.pricing_strategy_generator import PricingStrategyGenerator
from core.real_time_adjuster import RealTimeAdjuster, SalesUpdate


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
    """主函数"""

    print("=" * 60)
    print("智能打折促销系统 - 日清品阶梯定价优化")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载交易数据...")
    try:
        transaction_data = load_transaction_data('data/historical_transactions.csv')
        print(f"   加载成功！共 {len(transaction_data)} 条交易记录")
    except FileNotFoundError:
        print("   警告：数据文件未找到，使用示例数据")
        # 创建示例数据
        transaction_data = create_sample_data()
        print(f"   生成示例数据，共 {len(transaction_data)} 条记录")

    # 2. 初始化系统
    print("\n2. 初始化系统组件...")
    strategy_generator = PricingStrategyGenerator(transaction_data)
    real_time_adjuster = RealTimeAdjuster(strategy_generator)

    # 3. 用户输入
    print("\n3. 请输入定价参数:")

    product_code = "8006148" #input("   商品编码: ").strip()
    initial_stock = 60 #int(input("   当前库存数量: "))
    original_price = 35 #float(input("   商品原价: "))
    cost_price = 12 #float(input("   商品成本价: "))

    print("\n   促销时间段设置:")
    promotion_start = "20:00" #input("   开始时间 (HH:MM, 默认20:00): ").strip() or "20:00"
    promotion_end = "22:00" #input("   结束时间 (HH:MM, 默认22:00): ").strip() or "22:00"

    print("\n   折扣范围设置:")
    min_discount = 0.4 #float(input("   最低折扣 (如0.4表示4折): ") or "0.4")
    max_discount = 0.9 #float(input("   最高折扣 (如0.9表示9折): ") or "0.9")

    time_segments = 2 #int(input("\n   时间段数量 (默认4): ") or "4")

    # 4. 验证可行性
    print("\n4. 验证策略可行性...")
    feasibility = strategy_generator.validate_strategy_feasibility(
        product_code=product_code,
        initial_stock=initial_stock,
        promotion_start=promotion_start,
        promotion_end=promotion_end
    )

    print(f"   可行性评估: {feasibility['feasibility']} (得分: {feasibility['feasibility_score']})")
    print(f"   建议: {feasibility['recommendation']}")

    if feasibility['feasibility'] == '高风险':
        proceed = input("\n   风险较高，是否继续? (y/n): ").strip().lower()
        if proceed != 'y':
            print("   已取消生成策略")
            return

    # 5. 生成定价策略
    print(f"\n5. 正在生成定价策略...")

    strategy = strategy_generator.generate_pricing_strategy(
        product_code=product_code,
        initial_stock=initial_stock,
        promotion_start=promotion_start,
        promotion_end=promotion_end,
        min_discount=min_discount,
        max_discount=max_discount,
        time_segments=time_segments
    )

    # 6. 显示结果
    print("\n" + "=" * 60)
    print("定价策略生成完成!")
    print("=" * 60)

    print(f"\n商品信息:")
    print(f"  商品编码: {strategy.product_code}")
    print(f"  商品名称: {strategy.product_name}")
    print(f"  原价: ¥{strategy.original_price:.2f}")
    print(f"  成本价: ¥{strategy.cost_price:.2f}")
    print(f"  初始库存: {strategy.initial_stock}件")

    print(f"\n促销设置:")
    print(f"  促销时段: {strategy.promotion_start} - {strategy.promotion_end}")
    print(f"  折扣范围: {strategy.min_discount:.1%} - {strategy.max_discount:.1%}")
    print(f"  时间段数: {strategy.time_segments}")

    print(f"\n阶梯定价方案:")
    print("-" * 80)
    print(f"{'时间段':<15} {'折扣':<10} {'价格':<10} {'预期销量':<12} {'预期收入':<12} {'预期利润':<12}")
    print("-" * 80)

    total_expected_sales = 0
    total_revenue = 0
    total_profit = 0

    for i, stage in enumerate(strategy.pricing_schedule, 1):
        print(f"{stage['start_time']}-{stage['end_time']:<15} "
              f"{stage['discount_percentage']:<10} "
              f"¥{stage['price']:<9.2f} "
              f"{stage['expected_sales']:<12} "
              f"¥{stage['expected_revenue']:<11.2f} "
              f"¥{stage['expected_profit']:<11.2f}")

        total_expected_sales += stage['expected_sales']
        total_revenue += stage['expected_revenue']
        total_profit += stage['expected_profit']

    print("-" * 80)
    print(f"{'总计':<15} {'':<10} {'':<10} "
          f"{total_expected_sales:<12} "
          f"¥{total_revenue:<11.2f} "
          f"¥{total_profit:<11.2f}")

    print(f"\n方案评估:")
    eval_result = strategy.evaluation
    print(f"  预期总销量: {eval_result['total_expected_sales']}件")
    print(f"  预期总收入: ¥{eval_result['total_revenue']:.2f}")
    print(f"  预期总利润: ¥{eval_result['total_profit']:.2f}")
    print(f"  剩余库存: {eval_result['remaining_stock']}件")
    print(f"  售罄概率: {eval_result['sell_out_probability']:.1%}")
    print(f"  利润率: {eval_result['profit_margin']:.1%}")
    print(f"  平均折扣: {eval_result['average_discount']:.1%}")
    print(f"  推荐建议: {eval_result['recommendation']}")

    # 7. 保存策略
    save_option = input("\n是否保存策略到文件? (y/n): ").strip().lower()
    if save_option == 'y':
        filename = f"strategy_{strategy.product_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        strategy_generator.save_strategy(strategy, filename)
        print(f"   策略已保存到: {filename}")

    # 8. 备选策略
    see_alternatives = input("\n是否查看备选策略? (y/n): ").strip().lower()
    if see_alternatives == 'y':
        print("\n生成备选策略...")
        alternatives = strategy_generator.generate_alternative_strategies(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount
        )

        print("\n备选策略:")
        for name, alt in alternatives.items():
            print(f"\n{name.upper()}策略:")
            print(f"  描述: {alt['description']}")
            print(f"  适用场景: {alt['suitable_for']}")

            if name == 'single_price':
                print(f"  统一折扣: {alt['discount']:.1%}")
                print(f"  统一价格: ¥{alt['price']:.2f}")
            else:
                strat = alt['strategy']
                print(f"  预期利润: ¥{strat.evaluation['total_profit']:.2f}")
                print(f"  售罄概率: {strat.evaluation['sell_out_probability']:.1%}")

    # 9. 模拟实时调整
    simulate_realtime = input("\n是否模拟实时调整过程? (y/n): ").strip().lower()
    if simulate_realtime == 'y':
        simulate_real_time_adjustment(strategy, real_time_adjuster)

    print("\n" + "=" * 60)
    print("程序执行完成!")
    print("=" * 60)


def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""

    np.random.seed(42)

    # 生成30天的数据
    dates = pd.date_range('2024-01-01', '2024-01-30', freq='D')

    records = []
    product_codes = ['PROD001', 'PROD002', 'PROD003']
    product_names = ['鲜奶面包', '酸奶', '三明治']
    prices = [25.0, 15.0, 20.0]

    for date in dates:
        for product_idx, (product_code, product_name, base_price) in enumerate(
                zip(product_codes, product_names, prices)):
            # 每天生成5-15条交易记录
            num_transactions = np.random.randint(5, 16)

            for _ in range(num_transactions):
                # 交易时间在8:00-22:00之间
                hour = np.random.randint(8, 22)
                minute = np.random.randint(0, 60)

                # 折扣概率
                if np.random.random() < 0.3:  # 30%的概率有折扣
                    discount = np.random.choice([0.7, 0.8, 0.9])
                else:
                    discount = 1.0

                # 销售数量
                if discount < 1.0:
                    quantity = np.random.randint(1, 4)  # 促销时买的多
                else:
                    quantity = np.random.randint(1, 3)

                # 计算金额
                price = base_price * discount
                sales_amount = price * quantity
                discount_amount = base_price * quantity - sales_amount if discount < 1.0 else 0

                record = {
                    '日期': date.strftime('%Y-%m-%d'),
                    '门店编码': 'STORE001',
                    '流水单号': f"INV{date.strftime('%Y%m%d')}{np.random.randint(1000, 9999)}",
                    '会员id': f"M{np.random.randint(10000, 99999)}" if np.random.random() < 0.5 else None,
                    '交易时间': f"{date.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}:00",
                    '渠道名称': '门店',
                    '平台触点名称': '收银台',
                    '小类编码': f"CAT{product_idx + 1:03d}",
                    '商品编码': product_code,
                    '商品名称': product_name,
                    '售价': base_price,
                    '折扣类型': '促销折扣' if discount < 1.0 else '无折扣',
                    '税率': 0.13,
                    '销售数量': quantity,
                    '销售金额': sales_amount,
                    '销售净额': sales_amount,
                    '折扣金额': discount_amount
                }

                records.append(record)

    return pd.DataFrame(records)


def simulate_real_time_adjustment(strategy, real_time_adjuster):
    """模拟实时调整"""

    print("\n模拟实时调整过程...")
    print("-" * 60)

    # 模拟销售数据
    current_time = datetime.strptime(strategy.promotion_start, "%H:%M")

    # 第一阶段
    print(f"\n第一阶段开始 ({strategy.pricing_schedule[0]['start_time']}-{strategy.pricing_schedule[0]['end_time']})")
    print(f"  预期销量: {strategy.pricing_schedule[0]['expected_sales']}件")

    # 模拟销售（假设销量低于预期）
    actual_sales = int(strategy.pricing_schedule[0]['expected_sales'] * 0.6)  # 只完成60%

    print(f"  实际销量: {actual_sales}件 (完成预期{actual_sales / strategy.pricing_schedule[0]['expected_sales']:.0%})")

    # 记录销售
    sales_update = SalesUpdate(
        timestamp=current_time + timedelta(minutes=15),
        product_code=strategy.product_code,
        quantity_sold=actual_sales,
        actual_price=strategy.pricing_schedule[0]['price'],
        discount_applied=strategy.pricing_schedule[0]['discount'],
        remaining_stock=strategy.initial_stock - actual_sales
    )

    real_time_adjuster.record_sales(strategy.strategy_id, sales_update)

    # 检查是否需要调整
    adjusted_strategy = real_time_adjuster.check_and_adjust(
        strategy,
        current_time + timedelta(minutes=15)
    )

    if adjusted_strategy:
        print(f"\n检测到销售滞后，已生成调整后策略:")
        print(f"  新策略ID: {adjusted_strategy.strategy_id}")
        print(f"  调整后预期利润: ¥{adjusted_strategy.evaluation['total_profit']:.2f}")
        print(f"  调整后售罄概率: {adjusted_strategy.evaluation['sell_out_probability']:.1%}")

        print(f"\n调整后的阶梯定价:")
        for stage in adjusted_strategy.pricing_schedule:
            print(
                f"  {stage['start_time']}-{stage['end_time']}: {stage['discount_percentage']} (¥{stage['price']:.2f})")
    else:
        print(f"\n销售正常，无需调整")

    # 获取调整历史
    adjustment_summary = real_time_adjuster.get_adjustment_summary(strategy.strategy_id)
    print(f"\n调整历史: 共{adjustment_summary['adjustment_count']}次调整")


if __name__ == "__main__":
    main()