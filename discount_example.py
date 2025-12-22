# discount_example.py
"""
折扣方案优化使用示例
"""
import json
import numpy as np
from datetime import datetime, date
import pandas as pd
from main import HRMain
from param import params


def example_single_product():
    """单个商品折扣优化示例"""

    # 创建HRMain实例
    hr_main = HRMain(params)

    # 输入参数 - 修正日期格式
    input_params = {
        'product_code': '4834512',  # 商品编码
        'product_name': '福荫川式豆花380g',  # 商品名称
        'current_inventory': 50,  # 当前库存
        'promotion_window': ['20:00', '22:00'],  # 促销时间窗口
        'min_gross_margin': 0.15,  # 最低毛利率15%
        'allow_staggered_pricing': True,  # 允许阶梯定价
        'current_date': date(2025, 11, 1),  # 使用date对象或datetime对象
        'base_price': 7.99,  # 基准价格
        'cost_price': 5.59,  # 成本价 (70% of base_price)
        'historical_sales_data': None  # 可选，如果不提供会使用默认值
    }

    # 运行折扣优化
    result = hr_main.run_discount_optimization(input_params)

    if result:
        print("折扣方案生成成功:")
        print(f"商品: {result['product_code']}")
        print(f"当前库存: {result['current_inventory']}")
        print(f"促销窗口: {result['promotion_window'][0]} - {result['promotion_window'][1]}")
        print(f"\n折扣方案:")

        for i, plan in enumerate(result['discount_plan'], 1):
            print(f"{i}. {plan['time_slot']}: {plan['discount_percentage']}折")
            print(f"   最终价格: {plan['final_price']}元")
            print(f"   预期销量: {plan['expected_sales']}个")
            print(f"   预期利润: {plan['expected_profit']}元")

        print(f"\n方案指标:")
        metrics = result['plan_metrics']
        print(f"预期总销量: {metrics['total_expected_sales']}个")
        print(f"预期总收入: {metrics['total_expected_revenue']:.2f}元")
        print(f"预期总利润: {metrics['total_expected_profit']:.2f}元")
        print(f"库存清空率: {metrics['clearance_rate']:.1%}")
        print(f"平均毛利率: {metrics['avg_gross_margin']:.1%}")

        print(f"\n可行性分析:")
        feasibility = result['feasibility_analysis']
        for key, value in feasibility.items():
            print(f"  {key}: {'✓' if value else '✗'}")

        print(f"\n推荐建议: {result['recommendation']}")

    return result

def example_batch_products():
    """批量商品折扣优化示例"""
    
    hr_main = HRMain(params)
    
    # 批量商品参数
    products_params = [
        {
            'product_code': '4834512',
            'current_inventory': 50,
            'promotion_window': ['20:00', '22:00'],
            'min_gross_margin': 0.15,
            'allow_staggered_pricing': True,
            'base_price': 7.99,
            'cost_price': 5.59
        },
        {
            'product_code': '4701098',
            'current_inventory': 30,
            'promotion_window': ['19:00', '21:00'],
            'min_gross_margin': 0.2,
            'allow_staggered_pricing': False,
            'base_price': 5.99,
            'cost_price': 2.09
        },
        {
            'product_code': '8006144',
            'current_inventory': 80,
            'promotion_window': ['18:00', '20:00'],
            'min_gross_margin': 0.1,
            'allow_staggered_pricing': True,
            'base_price': 7.99,
            'cost_price': 2.19
        }
    ]
    
    # 运行批量优化
    results = hr_main.batch_discount_optimization(products_params)
    
    print(f"批量优化完成，共处理{len(results)}个商品")
    
    return results


def example_with_historical_data():
    """使用历史数据的折扣优化示例"""

    from datetime import datetime

    # 模拟历史销售数据
    historical_data = pd.DataFrame({
        '日期': pd.date_range('2025-10-01', periods=30, freq='D'),
        '商品编码': ['4834512'] * 30,
        '销售数量': np.random.randint(5, 20, 30),
        '销售金额': np.random.uniform(30, 100, 30),
        '售价': [7.99] * 30,
        '折扣类型': ['n-无折扣促销'] * 30
    })

    hr_main = HRMain(params)

    input_params = {
        'product_code': '4834512',
        'current_inventory': 100,
        'promotion_window': ['20:00', '22:00'],
        'min_gross_margin': 0.1,
        'allow_staggered_pricing': True,
        'current_date': datetime(2025, 11, 1),  # 使用datetime对象
        'historical_sales_data': historical_data,
        'base_price': 7.99,
        'cost_price': 5.59
    }

    result = hr_main.run_discount_optimization(input_params)

    return result

def run_api_example():
    """API接口使用示例"""
    import requests
    
    # API端点
    api_url = "http://localhost:5000/api/discount-plan"
    
    # 请求数据
    request_data = {
        "product_code": "8006144",
        "current_inventory": 10,
        "promotion_window": ["20:00", "22:00"],
        "min_gross_margin": 0.15,
        "allow_staggered_pricing": True,
        "current_date": "2025-11-01",
        "base_price": 7.99,
        "cost_price": 2.59
    }
    
    # 发送请求
    response = requests.post(api_url, json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("API调用成功:")
            print(json.dumps(result['data'], indent=2, ensure_ascii=False))
        else:
            print(f"API调用失败: {result.get('error')}")
    else:
        print(f"HTTP错误: {response.status_code}")
    
    return response

if __name__ == '__main__':
    print("=== 单个商品折扣优化示例 ===")
    result1 = example_single_product()
    print(result1)
    print("++" * 20)
    print("\n=== 批量商品折扣优化示例 ===")
    result2 = example_batch_products()
    print(result2)
    print("++" * 20)
    print("\n=== 使用历史数据的折扣优化示例 ===")
    result3 = example_with_historical_data()
    print(result3)
    print("++" * 20)
    print("\n示例运行完成！")