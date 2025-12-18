# api/pricing_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import uvicorn
import pandas as pd
import os
import sys
from core.config import *
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pricing_strategy_generator import EnhancedPricingStrategyGenerator
from models.demand_predictor import EnhancedDemandPredictor as DemandPredictor

app = FastAPI(
    title="智能定价API",
    description="日清品阶梯定价优化系统",
    version="1.0.0"
)

# 全局变量
strategy_generator = None
strategies_db = {}  # 存储策略的简单内存数据库

class PricingRequest(BaseModel):
    """定价请求"""
    product_code: str = Field(..., description="商品编码")
    initial_stock: int = Field(..., description="初始库存")
    original_price: Optional[float] = Field(None, description="商品原价")
    cost_price: Optional[float] = Field(None, description="商品成本价")
    promotion_start: str = Field("20:00", description="促销开始时间 (HH:MM)")
    promotion_end: str = Field("22:00", description="促销结束时间 (HH:MM)")
    min_discount: float = Field(0.4, description="最低折扣 (0-1)")
    max_discount: float = Field(0.9, description="最高折扣 (0-1)")
    time_segments: int = Field(4, description="时间段数量")
    store_code: Optional[str] = Field(None, description="门店编码")
    use_weather: bool = Field(True, description="是否使用天气数据")
    use_calendar: bool = Field(True, description="是否使用日历数据")

class SalesUpdateRequest(BaseModel):
    """销售更新请求"""
    strategy_id: str = Field(..., description="策略ID")
    product_code: str = Field(..., description="商品编码")
    quantity_sold: int = Field(..., description="销售数量")
    actual_price: float = Field(..., description="实际售价")
    discount_applied: float = Field(..., description="应用折扣")
    remaining_stock: int = Field(..., description="剩余库存")
    timestamp: Optional[str] = Field(None, description="时间戳 (YYYY-MM-DD HH:MM:SS)")

class StrategyAdjustmentRequest(BaseModel):
    """策略调整请求"""
    strategy_id: str = Field(..., description="策略ID")
    current_time: Optional[str] = Field(None, description="当前时间 (HH:MM)")
    current_stock: Optional[int] = Field(None, description="当前库存")

def load_or_create_sample_data():
    """加载或创建示例数据"""
    print("正在初始化数据...")
    
    # 尝试加载实际数据文件
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    transaction_file = os.path.join(data_dir, "historical_transactions.csv")
    weather_file = os.path.join(data_dir, "weather_info.csv")
    calendar_file = os.path.join(data_dir, "calendar_info.csv")
    
    # 导入main.py中的函数
    from main import create_sample_data, load_data_files
    
    if os.path.exists(transaction_file):
        print("找到数据文件，加载真实数据...")
        transaction_data, weather_data, calendar_data = load_data_files(
            transaction_file, weather_file, calendar_file
        )
    else:
        print("未找到数据文件，使用示例数据...")
        transaction_data, weather_data, calendar_data = create_sample_data()
    
    return transaction_data, weather_data, calendar_data

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global strategy_generator, strategies_db
    
    try:
        # 加载数据
        transaction_data, weather_data, calendar_data = load_or_create_sample_data()
        
        if transaction_data is None:
            print("警告: 无法加载数据，系统将使用空数据初始化")
            # 创建空数据框
            transaction_data = pd.DataFrame()
            weather_data = pd.DataFrame()
            calendar_data = pd.DataFrame()
        config_manager = ConfigManager()
        config = config_manager.config
        config['product_code'] = "8006144"
        # 初始化策略生成器
        strategy_generator = EnhancedPricingStrategyGenerator(
            transaction_data=transaction_data,
            weather_data=weather_data,
            calendar_data=calendar_data,
            config=config
        )
        
        strategies_db = {}
        
        print("系统初始化完成")
        print(f"加载的交易数据记录数: {len(transaction_data)}")
        
    except Exception as e:
        print(f"系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "智能定价API",
        "version": "1.0.0",
        "status": "运行正常",
        "endpoints": {
            "generate_strategy": "POST /api/pricing/generate",
            "get_strategy": "GET /api/pricing/strategy/{strategy_id}",
            "update_sales": "POST /api/pricing/sales",
            "adjust_strategy": "POST /api/pricing/adjust",
            "feasibility_check": "GET /api/pricing/feasibility/{product_code}",
            "get_products": "GET /api/pricing/products",
            "health_check": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "initialized": strategy_generator is not None,
        "strategies_count": len(strategies_db)
    }

@app.get("/api/pricing/products")
async def get_available_products(limit: int = 10):
    """获取可用的商品列表"""
    if strategy_generator is None:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        transaction_data = strategy_generator.transaction_data
        if transaction_data.empty:
            return {
                "success": True,
                "products": [],
                "message": "没有可用的商品数据"
            }
        
        # 获取唯一的商品信息
        product_info = transaction_data[['商品编码', '商品名称']].drop_duplicates()
        
        # 计算每个商品的平均价格
        avg_prices = {}
        for code in product_info['商品编码'].unique():
            prices = transaction_data[transaction_data['商品编码'] == code]['售价']
            if len(prices) > 0:
                avg_prices[code] = prices.mean()
        
        # 构建响应
        products = []
        for _, row in product_info.head(limit).iterrows():
            product_code = row['商品编码']
            products.append({
                "product_code": product_code,
                "product_name": row['商品名称'],
                "average_price": avg_prices.get(product_code),
                "sales_count": len(transaction_data[transaction_data['商品编码'] == product_code])
            })
        
        return {
            "success": True,
            "products": products,
            "total_count": len(product_info)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取商品列表失败: {str(e)}")

@app.post("/api/pricing/generate")
async def generate_pricing_strategy(request: PricingRequest):
    """生成定价策略"""
    global strategies_db
    
    if strategy_generator is None:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        # 获取原价和成本价（如果未提供，则尝试从数据中获取）
        transaction_data = strategy_generator.transaction_data
        
        if request.original_price is None and not transaction_data.empty:
            # 尝试从历史数据中获取平均价格作为原价
            product_data = transaction_data[transaction_data['商品编码'] == request.product_code]
            if not product_data.empty:
                request.original_price = product_data['售价'].mean()
            else:
                request.original_price = 10.0  # 默认值
        
        if request.cost_price is None:
            # 假设成本价为原价的70%
            request.cost_price = request.original_price * 0.7 if request.original_price else 7.0
        
        # 生成策略
        strategy = strategy_generator.generate_pricing_strategy(
            product_code=request.product_code,
            initial_stock=request.initial_stock,
            promotion_start=request.promotion_start,
            promotion_end=request.promotion_end,
            min_discount=request.min_discount,
            max_discount=request.max_discount,
            time_segments=request.time_segments,
            current_time=pd.to_datetime('2025-10-31 10:21:25'),
            store_code=request.store_code,
            use_weather=request.use_weather,
            use_calendar=request.use_calendar
        )
        
        # 保存策略
        strategies_db[strategy.strategy_id] = strategy
        
        # 转换策略为字典以便序列化
        strategy_dict = {
            "strategy_id": strategy.strategy_id,
            "generated_time": strategy.generated_time,
            "product_code": strategy.product_code,
            "product_name": strategy.product_name,
            "original_price": strategy.original_price,
            "cost_price": strategy.cost_price,
            "initial_stock": strategy.initial_stock,
            "promotion_start": strategy.promotion_start,
            "promotion_end": strategy.promotion_end,
            "min_discount": strategy.min_discount,
            "max_discount": strategy.max_discount,
            "time_segments": strategy.time_segments,
            "weather_consideration": strategy.weather_consideration,
            "calendar_consideration": strategy.calendar_consideration,
            "pricing_schedule": strategy.pricing_schedule,
            "evaluation": strategy.evaluation,
            "confidence_score": strategy.confidence_score
        }
        
        return {
            "success": True,
            "strategy_id": strategy.strategy_id,
            "strategy": strategy_dict,
            "message": "定价策略生成成功"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成策略失败: {str(e)}")

@app.get("/api/pricing/strategy/{strategy_id}")
async def get_pricing_strategy(strategy_id: str):
    """获取定价策略"""
    if strategy_id not in strategies_db:
        raise HTTPException(status_code=404, detail="策略未找到")
    
    strategy = strategies_db[strategy_id]
    
    # 转换策略为字典
    strategy_dict = {
        "strategy_id": strategy.strategy_id,
        "generated_time": strategy.generated_time,
        "product_code": strategy.product_code,
        "product_name": strategy.product_name,
        "original_price": strategy.original_price,
        "cost_price": strategy.cost_price,
        "initial_stock": strategy.initial_stock,
        "promotion_start": strategy.promotion_start,
        "promotion_end": strategy.promotion_end,
        "min_discount": strategy.min_discount,
        "max_discount": strategy.max_discount,
        "time_segments": strategy.time_segments,
        "weather_consideration": strategy.weather_consideration,
        "calendar_consideration": strategy.calendar_consideration,
        "pricing_schedule": strategy.pricing_schedule,
        "evaluation": strategy.evaluation,
        "confidence_score": strategy.confidence_score
    }
    
    return {
        "success": True,
        "strategy": strategy_dict
    }

@app.get("/api/pricing/feasibility/{product_code}")
async def check_feasibility(
    product_code: str,
    initial_stock: int,
    promotion_start: str = "20:00",
    promotion_end: str = "22:00"
):
    """检查可行性"""
    if strategy_generator is None:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        # 使用策略生成器的验证方法
        feasibility_result = strategy_generator.validate_strategy_feasibility(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end
        )
        
        return {
            "success": True,
            "feasibility": feasibility_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查可行性失败: {str(e)}")

@app.post("/api/pricing/sales")
async def update_sales_data(request: SalesUpdateRequest):
    """更新销售数据"""
    if request.strategy_id not in strategies_db:
        raise HTTPException(status_code=404, detail="策略未找到")
    
    try:
        # 这里可以添加销售数据记录逻辑
        # 例如，保存到数据库或更新策略状态
        
        return {
            "success": True,
            "message": "销售数据更新成功",
            "strategy_id": request.strategy_id,
            "remaining_stock": request.remaining_stock
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新销售数据失败: {str(e)}")

@app.post("/api/pricing/adjust")
async def adjust_pricing_strategy(request: StrategyAdjustmentRequest):
    """调整定价策略"""
    if request.strategy_id not in strategies_db:
        raise HTTPException(status_code=404, detail="策略未找到")
    
    try:
        original_strategy = strategies_db[request.strategy_id]
        
        # 这里可以实现实时调整逻辑
        # 例如，根据当前时间和库存调整折扣
        
        # 简化的调整逻辑：如果剩余库存多，可以增加折扣
        current_stock = request.current_stock if request.current_stock else original_strategy.initial_stock
        
        # 创建调整后的策略（这里简单复制原策略，实际应用需要更复杂的逻辑）
        adjusted_strategy = original_strategy
        
        # 如果实现了调整器类，可以这样调用：
        # if strategy_generator.real_time_adjuster:
        #     adjusted_strategy = strategy_generator.real_time_adjuster.check_and_adjust(
        #         original_strategy, current_time, current_stock
        #     )
        
        return {
            "success": True,
            "adjusted": False,  # 暂时不实现调整逻辑
            "message": "策略调整功能待实现",
            "current_stock": current_stock,
            "original_strategy_id": request.strategy_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调整策略失败: {str(e)}")

@app.get("/api/pricing/alternatives/{product_code}")
async def get_alternative_strategies(
    product_code: str,
    initial_stock: int,
    promotion_start: str = "20:00",
    promotion_end: str = "22:00",
    min_discount: float = 0.4,
    max_discount: float = 0.9
):
    """获取备选策略"""
    if strategy_generator is None:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        # 生成主要策略
        main_strategy = strategy_generator.generate_pricing_strategy(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount,
            time_segments=4  # 默认4个时间段
        )
        
        # 生成备选策略（不同时间段数）
        alternatives = {}
        
        # 备选1：更多时间段（更精细）
        alt1 = strategy_generator.generate_pricing_strategy(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount,
            time_segments=6  # 6个时间段
        )
        
        # 备选2：更少时间段（更简单）
        alt2 = strategy_generator.generate_pricing_strategy(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount,
            time_segments=2  # 2个时间段
        )
        
        # 保存备选策略
        strategies_db[main_strategy.strategy_id] = main_strategy
        strategies_db[alt1.strategy_id] = alt1
        strategies_db[alt2.strategy_id] = alt2
        
        # 构建响应
        alternatives = {
            "main": {
                "strategy_id": main_strategy.strategy_id,
                "time_segments": 4,
                "description": "标准策略（4个时间段）",
                "expected_profit": main_strategy.evaluation['total_profit']
            },
            "detailed": {
                "strategy_id": alt1.strategy_id,
                "time_segments": 6,
                "description": "精细策略（6个时间段）",
                "expected_profit": alt1.evaluation['total_profit']
            },
            "simple": {
                "strategy_id": alt2.strategy_id,
                "time_segments": 2,
                "description": "简单策略（2个时间段）",
                "expected_profit": alt2.evaluation['total_profit']
            }
        }
        
        return {
            "success": True,
            "alternatives": alternatives,
            "recommendation": "标准策略"  # 简单的推荐逻辑
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取备选策略失败: {str(e)}")

if __name__ == "__main__":
    # 运行API服务器
    print("正在启动智能定价API服务器...")
    print(f"访问地址: http://localhost:8000")
    print(f"API文档: http://localhost:8000/")
    uvicorn.run(
        "pricing_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )