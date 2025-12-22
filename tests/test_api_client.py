# test_api_client.py
import requests
import json

# API基础URL
BASE_URL = "http://localhost:8000"

def test_api():
    """测试API接口"""
    
    print("=== 测试智能定价API ===\n")
    
    # 1. 健康检查
    print("1. 健康检查:")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   状态码: {response.status_code}")
    print(f"   响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}\n")
    
    # 2. 获取商品列表
    print("2. 获取商品列表:")
    response = requests.get(f"{BASE_URL}/api/pricing/products")
    print(f"   状态码: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   商品数量: {len(data.get('products', []))}")
        for product in data.get('products', [])[:3]:  # 显示前3个
            print(f"   - {product['product_code']}: {product['product_name']}")
    print()
    
    # 3. 生成定价策略
    print("3. 生成定价策略:")
    request_data = {
        "product_code": "4834512",  # 示例商品编码
        "initial_stock": 50,
        "original_price": 7.99,
        "cost_price": 5.50,
        "promotion_start": "20:00",
        "promotion_end": "22:00",
        "min_discount": 0.4,
        "max_discount": 0.8,
        "time_segments": 4,
        "store_code": "205625",
        "use_weather": True,
        "use_calendar": True
    }
    
    response = requests.post(
        f"{BASE_URL}/api/pricing/generate",
        json=request_data
    )
    
    print(f"   状态码: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        strategy_id = data.get('strategy_id')
        print(f"   策略ID: {strategy_id}")
        print(f"   商品: {data.get('strategy', {}).get('product_name')}")
        print(f"   置信度: {data.get('strategy', {}).get('confidence_score'):.1%}")
        
        # 4. 获取策略详情
        print("\n4. 获取策略详情:")
        response = requests.get(f"{BASE_URL}/api/pricing/strategy/{strategy_id}")
        if response.status_code == 200:
            data = response.json()
            schedule = data.get('strategy', {}).get('pricing_schedule', [])
            print(f"   阶梯定价方案:")
            for stage in schedule:
                print(f"     {stage['start_time']}-{stage['end_time']}: "
                      f"折扣{stage['discount_percentage']:.1%}, "
                      f"价格¥{stage['price']:.2f}, "
                      f"预期销量{stage['expected_sales']}件")
    else:
        print(f"   错误: {response.text}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_api()