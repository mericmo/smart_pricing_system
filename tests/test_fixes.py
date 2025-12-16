# test_fixes.py
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.demand_predictor import EnhancedDemandPredictor
from models.pricing_optimizer import PricingOptimizer
from data.feature_engineer import PricingFeatureEngineer

def create_test_transaction_data():
    """创建测试交易数据"""
    np.random.seed(42)
    
    dates = pd.date_range('2024-12-01', '2024-12-10', freq='D')
    product_code = '4834512'
    
    records = []
    for date in dates:
        for _ in range(np.random.randint(5, 15)):
            hour = np.random.randint(8, 22)
            minute = np.random.randint(0, 60)
            
            if hour >= 20:
                discount = np.random.choice([0.7, 0.8, 0.9])
                discount_type = f"促销{discount*10:.0f}折"
            else:
                discount = 1.0
                discount_type = "n-无折扣促销"
            
            quantity = np.random.randint(1, 3)
            price = 7.99
            
            records.append({
                '日期': date.strftime('%Y/%m/%d'),
                '门店编码': '205625',
                '交易时间': f"{date.strftime('%Y/%m/%d')} {hour:02d}:{minute:02d}",
                '商品编码': product_code,
                '商品名称': '福荫川式豆花380g',
                '售价': price,
                '折扣类型': discount_type,
                '销售数量': quantity,
                '销售金额': price * discount * quantity,
                '销售净额': price * discount * quantity * 0.87,
                '折扣金额': price * quantity - price * discount * quantity if discount < 1.0 else 0
            })
    
    df = pd.DataFrame(records)
    
    # 添加实际折扣率列
    df['实际折扣率'] = df.apply(
        lambda row: row['销售金额'] / (row['售价'] * row['销售数量']) 
        if row['售价'] * row['销售数量'] > 0 else 1.0,
        axis=1
    )
    
    # 添加是否折扣列
    df['是否折扣'] = df['实际折扣率'].apply(lambda x: 0 if x >= 0.99 else 1)
    
    return df

def test_demand_predictor():
    """测试需求预测器"""
    print("测试需求预测器...")
    
    # 创建测试数据
    transaction_data = create_test_transaction_data()
    
    # 创建需求预测器
    predictor = EnhancedDemandPredictor(model_type='xgboost')
    
    try:
        # 准备训练数据
        X, y = predictor.prepare_training_data_from_transactions(
            transaction_data=transaction_data,
            product_code='4834512',
            promotion_hours=(20, 22)
        )
        
        print(f"训练数据准备成功: X.shape={X.shape}, y.shape={y.shape}")
        print(f"特征列: {list(X.columns)}")
        
        # 训练模型
        predictor.train(X, y)
        print("模型训练成功")
        
        # 测试预测
        features = {
            'hist_avg_sales': 10.0,
            'price_elasticity': 1.2,
            'hist_promo_sales_ratio': 1.5,
            'sales_trend': 0.1
        }
        
        prediction = predictor.predict_demand(
            features=features,
            discount_rate=0.7,
            time_to_close=0.5,
            current_stock=100
        )
        
        print(f"需求预测结果: {prediction}")
        print("需求预测器测试通过!")
        
    except Exception as e:
        print(f"需求预测器测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_pricing_optimizer():
    """测试定价优化器"""
    print("\n测试定价优化器...")
    
    # 创建需求预测器
    predictor = EnhancedDemandPredictor(model_type='xgboost')
    
    # 创建定价优化器
    optimizer = PricingOptimizer(
        demand_predictor=predictor,
        cost_price=4.8,
        original_price=7.99
    )
    
    # 测试特征
    features = {
        'hist_avg_sales': 10.0,
        'price_elasticity': 1.2,
        'hist_promo_sales_ratio': 1.5,
        'sales_trend': 0.1
    }
    
    try:
        # 测试阶梯定价
        schedule = optimizer.optimize_staged_pricing(
            initial_stock=50,
            promotion_start="20:00",
            promotion_end="22:00",
            min_discount=0.4,
            max_discount=0.9,
            time_segments=4,
            features=features
        )
        
        print(f"生成了 {len(schedule)} 个定价时段")
        for i, segment in enumerate(schedule):
            print(f"时段{i+1}: {segment.start_time}-{segment.end_time}, "
                  f"折扣: {segment.discount:.2f}, "
                  f"预期销量: {segment.expected_sales}, "
                  f"预期利润: {segment.profit:.2f}")
        
        print("定价优化器测试通过!")
        
    except Exception as e:
        print(f"定价优化器测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_feature_engineer():
    """测试特征工程"""
    print("\n测试特征工程...")
    
    # 创建测试数据
    transaction_data = create_test_transaction_data()
    
    # 创建特征工程
    feature_engineer = PricingFeatureEngineer()
    
    try:
        # 测试特征提取
        features = feature_engineer.create_features(
            transaction_data=transaction_data,
            product_code='4834512',
            promotion_hours=(20, 22),
            current_time=pd.Timestamp('2024-12-10 19:30:00')
        )
        
        print(f"提取了 {len(features)} 个特征")
        print("主要特征:")
        for key, value in list(features.items())[:20]:  # 显示前20个特征
            print(f"  {key}: {value}")
        
        print("特征工程测试通过!")
        
    except Exception as e:
        print(f"特征工程测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_integration():
    """测试完整流程"""
    print("\n测试完整流程...")
    
    # 创建测试数据
    transaction_data = create_test_transaction_data()
    
    # 创建特征工程
    feature_engineer = PricingFeatureEngineer()
    
    # 创建需求预测器
    demand_predictor = EnhancedDemandPredictor(model_type='xgboost')
    
    # 创建定价优化器
    pricing_optimizer = PricingOptimizer(
        demand_predictor=demand_predictor,
        cost_price=4.8,
        original_price=7.99
    )
    
    try:
        # 1. 提取特征
        features = feature_engineer.create_features(
            transaction_data=transaction_data,
            product_code='4834512',
            promotion_hours=(20, 22),
            current_time=pd.Timestamp.now()
        )
        
        print("1. 特征提取成功")
        
        # 2. 训练需求预测器（简化，实际需要准备训练数据）
        print("2. 跳过需求预测器训练（使用启发式模型）")
        
        # 3. 生成定价策略
        schedule = pricing_optimizer.optimize_staged_pricing(
            initial_stock=50,
            promotion_start="20:00",
            promotion_end="22:00",
            min_discount=0.4,
            max_discount=0.9,
            time_segments=4,
            features=features
        )
        
        print(f"3. 生成了 {len(schedule)} 个定价时段")
        
        # 计算总指标
        total_sales = sum(s.expected_sales for s in schedule)
        total_profit = sum(s.profit for s in schedule)
        total_revenue = sum(s.revenue for s in schedule)
        
        print(f"总预期销量: {total_sales}")
        print(f"总预期收入: {total_revenue:.2f}")
        print(f"总预期利润: {total_profit:.2f}")
        
        print("完整流程测试通过!")
        
    except Exception as e:
        print(f"完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("测试修复的问题")
    print("=" * 60)
    
    test_demand_predictor()
    test_pricing_optimizer()
    test_feature_engineer()
    test_integration()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)