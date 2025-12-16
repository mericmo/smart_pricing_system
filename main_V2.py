# main.py (refactored MVP)
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.pricing_strategy_generator import PricingStrategyGenerator
from core.real_time_adjuster import RealTimeAdjuster, SalesUpdate

DATA_PATH = "data/historical_transactions.csv"


def load_transaction_data(filepath: str) -> pd.DataFrame:
    """加载交易数据，返回 pandas.DataFrame；若文件缺失返回空 DataFrame"""
    try:
        df = pd.read_csv(filepath, encoding="utf-8", parse_dates=["日期", "交易时间"], dtype={"商品编码": str, "门店编码": str})
    except FileNotFoundError:
        return pd.DataFrame()
    # 基本字段检查
    expected = {"商品编码", "商品名称", "售价", "销售数量", "销售金额", "交易时间"}
    missing = expected.difference(set(df.columns))
    if missing:
        print(f"警告：历史数据缺少列: {missing}")
    return df


def create_sample_data() -> pd.DataFrame:
    """如果没有历史数据，生成示例数据（用于本地回测）"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2024-01-30", freq="D")
    records = []
    product_codes = ["8006148", "PROD002", "PROD003"]
    product_names = ["鲜奶面包", "酸奶", "三明治"]
    base_prices = [35.0, 15.0, 20.0]

    for date in dates:
        for idx, (code, name, base_price) in enumerate(zip(product_codes, product_names, base_prices)):
            num_tx = np.random.randint(5, 16)
            for _ in range(num_tx):
                hour = np.random.randint(8, 22)
                minute = np.random.randint(0, 60)
                if np.random.random() < 0.25:
                    discount = np.random.choice([0.1, 0.2, 0.3, 0.4])  # 折扣幅度
                else:
                    discount = 0.0
                qty = np.random.randint(1, 4)
                price = base_price * (1 - discount)
                sales_amount = price * qty
                records.append({
                    "日期": date,
                    "交易时间": pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=hour, minute=minute),
                    "门店编码": "STORE001",
                    "商品编码": code,
                    "商品名称": name,
                    "售价": base_price,
                    "折扣": discount,
                    "单价": price,
                    "销售数量": qty,
                    "销售金额": sales_amount
                })
    return pd.DataFrame(records)


def print_strategy(strategy):
    print("\n" + "=" * 60)
    print("定价策略生成完成!")
    print("=" * 60)
    print(f"商品编码: {strategy.product_code}  商品名称: {strategy.product_name}")
    print(f"原价: ¥{strategy.original_price:.2f}  成本: ¥{strategy.cost_price:.2f}  初始库存: {strategy.initial_stock}")
    print(f"促销时段: {strategy.promotion_start} - {strategy.promotion_end}  时间段数: {strategy.time_segments}")
    print("\n阶梯定价方案:")
    print("-" * 80)
    print(f"{'时间段':<20}{'折扣':<10}{'价格':<10}{'预期销量':<12}{'预期收入':<12}{'预期利润':<12}")
    print("-" * 80)
    for stage in strategy.pricing_schedule:
        print(f"{stage['start_time']}-{stage['end_time']:<10}{stage['discount']:<9.1%} ¥{stage['price']:<9.2f} "
              f"{stage['expected_sales']:<12.1f} ¥{stage['expected_revenue']:<11.2f} ¥{stage['expected_profit']:<11.2f}")
    print("-" * 80)
    ev = strategy.evaluation
    print(f"预期总销量: {ev['total_expected_sales']:.1f} 件  预期总收入: ¥{ev['total_revenue']:.2f}  预期总利润: ¥{ev['total_profit']:.2f}")
    print(f"剩余库存: {ev['remaining_stock']}  售罄概率(估计): {ev['sell_out_probability']:.1%}  平均折扣: {ev['average_discount']:.1%}")
    print(f"建议: {ev.get('recommendation','-')}")


def main():
    print("=" * 60)
    print("智能打折促销系统 - 日清品阶梯定价优化（MVP）")
    print("=" * 60)

    df = load_transaction_data(DATA_PATH)
    if df.empty:
        print("未找到历史数据，使用示例数据")
        df = create_sample_data()

    generator = PricingStrategyGenerator(df)
    adjuster = RealTimeAdjuster(generator)

    # 示例输入（MVP：把交互换成预设参数，便于自动运行/测试）
    product_code = "8006148"
    initial_stock = 60
    original_price = 35.0
    cost_price = 12.0
    promotion_start = "18:00"
    promotion_end = "22:00"
    # 折扣表示折扣幅度（0.0 无折扣，0.6 表示 6 折/60%）
    min_discount = 0.4
    max_discount = 0.7
    time_segments = 2  # 推荐 2~4 段

    feas = generator.validate_strategy_feasibility(product_code, initial_stock, promotion_start, promotion_end)
    print(f"可行性: {feas['feasibility']}  建议: {feas['recommendation']}")

    strategy = generator.generate_pricing_strategy(
        product_code=product_code,
        initial_stock=initial_stock,
        promotion_start=promotion_start,
        promotion_end=promotion_end,
        min_discount=min_discount,
        max_discount=max_discount,
        time_segments=time_segments
    )

    print_strategy(strategy)

    # 保存策略示例
    filename = f"strategy_{strategy.product_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    generator.save_strategy(strategy, filename)
    print(f"策略已保存到: {filename}")

    # 模拟实时调整（MVP）
    print("\n模拟实时调整（演示第一阶段销量滞后触发调整）")
    # 假设第一阶段实际完成 60% 预期
    first_expected = strategy.pricing_schedule[0]['expected_sales']
    actual_first_sales = int(max(0, round(first_expected * 0.6)))
    timestamp = datetime.strptime(strategy.promotion_start, "%H:%M") + timedelta(minutes=15)
    upd = SalesUpdate(timestamp=timestamp, product_code=strategy.product_code, quantity_sold=actual_first_sales,
                      actual_price=strategy.pricing_schedule[0]['price'], discount_applied=strategy.pricing_schedule[0]['discount'],
                      remaining_stock=strategy.initial_stock - actual_first_sales)
    adjuster.record_sales(strategy.strategy_id, upd)
    adjusted = adjuster.check_and_adjust(strategy, timestamp)
    if adjusted:
        print("\n检测到需要调整，新的策略如下：")
        print_strategy(adjusted)
    else:
        print("\n不需要调整。")

    print("\n程序执行完成。")


if __name__ == "__main__":
    main()