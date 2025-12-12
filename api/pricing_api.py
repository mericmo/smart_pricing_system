# api/pricing_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn

from core.pricing_strategy_generator import PricingStrategyGenerator, PricingStrategy
from core.real_time_adjuster import RealTimeAdjuster, SalesUpdate

app = FastAPI(title="智能定价API", description="日清品阶梯定价优化系统")

# 全局变量（实际应用中应该使用数据库）
transaction_data = None
strategy_generator = None
real_time_adjuster = None

class PricingRequest(BaseModel):
    """定价请求"""
    product_code: str
    initial_stock: int
    promotion_start: str = "20:00"
    promotion_end: str = "22:00"
    min_discount: float = 0.4
    max_discount: float = 0.9
    time_segments: int = 4
    store_code: Optional[str] = None

class SalesUpdateRequest(BaseModel):
    """销售更新请求"""
    strategy_id: str
    product_code: str
    quantity_sold: int
    actual_price: float
    discount_applied: float
    remaining_stock: int
    timestamp: Optional[str] = None

class StrategyAdjustmentRequest(BaseModel):
    """策略调整请求"""
    strategy_id: str
    current_time: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global transaction_data, strategy_generator, real_time_adjuster
    
    # 这里应该从数据库加载数据
    # 为了演示，我们创建示例数据
    import pandas as pd
    from main import create_sample_data
    
    transaction_data = create_sample_data()
    strategy_generator = PricingStrategyGenerator(transaction_data)
    real_time_adjuster = RealTimeAdjuster(strategy_generator)
    
    print("系统初始化完成")

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "智能定价API",
        "version": "1.0.0",
        "endpoints": {
            "generate_strategy": "POST /api/pricing/generate",
            "get_strategy": "GET /api/pricing/strategy/{strategy_id}",
            "update_sales": "POST /api/pricing/sales",
            "adjust_strategy": "POST /api/pricing/adjust",
            "feasibility_check": "GET /api/pricing/feasibility/{product_code}"
        }
    }

@app.post("/api/pricing/generate", response_model=Dict[str, Any])
async def generate_pricing_strategy(request: PricingRequest):
    """生成定价策略"""
    
    try:
        strategy = strategy_generator.generate_pricing_strategy(
            product_code=request.product_code,
            initial_stock=request.initial_stock,
            promotion_start=request.promotion_start,
            promotion_end=request.promotion_end,
            min_discount=request.min_discount,
            max_discount=request.max_discount,
            time_segments=request.time_segments,
            store_code=request.store_code
        )
        
        return {
            "success": True,
            "strategy_id": strategy.strategy_id,
            "strategy": strategy.__dict__,
            "message": "定价策略生成成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成策略失败: {str(e)}")

@app.get("/api/pricing/strategy/{strategy_id}")
async def get_pricing_strategy(strategy_id: str):
    """获取定价策略"""
    
    strategy = strategy_generator.get_strategy_by_id(strategy_id)
    
    if not strategy:
        raise HTTPException(status_code=404, detail="策略未找到")
    
    return {
        "success": True,
        "strategy": strategy.__dict__
    }

@app.get("/api/pricing/feasibility/{product_code}")
async def check_feasibility(
    product_code: str,
    initial_stock: int,
    promotion_start: str = "20:00",
    promotion_end: str = "22:00"
):
    """检查可行性"""
    
    try:
        feasibility = strategy_generator.validate_strategy_feasibility(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end
        )
        
        return {
            "success": True,
            "feasibility": feasibility
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查可行性失败: {str(e)}")

@app.post("/api/pricing/sales")
async def update_sales_data(request: SalesUpdateRequest):
    """更新销售数据"""
    
    try:
        # 解析时间戳
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp)
        else:
            timestamp = datetime.now()
        
        # 创建销售更新对象
        sales_update = SalesUpdate(
            timestamp=timestamp,
            product_code=request.product_code,
            quantity_sold=request.quantity_sold,
            actual_price=request.actual_price,
            discount_applied=request.discount_applied,
            remaining_stock=request.remaining_stock
        )
        
        # 记录销售
        real_time_adjuster.record_sales(request.strategy_id, sales_update)
        
        return {
            "success": True,
            "message": "销售数据更新成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新销售数据失败: {str(e)}")

@app.post("/api/pricing/adjust")
async def adjust_pricing_strategy(request: StrategyAdjustmentRequest):
    """调整定价策略"""
    
    try:
        # 获取原策略
        strategy = strategy_generator.get_strategy_by_id(request.strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail="策略未找到")
        
        # 解析当前时间
        if request.current_time:
            current_time = datetime.fromisoformat(request.current_time)
        else:
            current_time = datetime.now()
        
        # 检查并调整
        adjusted_strategy = real_time_adjuster.check_and_adjust(strategy, current_time)
        
        if adjusted_strategy:
            return {
                "success": True,
                "adjusted": True,
                "new_strategy_id": adjusted_strategy.strategy_id,
                "strategy": adjusted_strategy.__dict__,
                "message": "策略已调整"
            }
        else:
            return {
                "success": True,
                "adjusted": False,
                "message": "策略无需调整"
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
    
    try:
        alternatives = strategy_generator.generate_alternative_strategies(
            product_code=product_code,
            initial_stock=initial_stock,
            promotion_start=promotion_start,
            promotion_end=promotion_end,
            min_discount=min_discount,
            max_discount=max_discount
        )
        
        # 转换格式以便JSON序列化
        result = {}
        for name, alt in alternatives.items():
            if name == 'single_price':
                result[name] = alt
            else:
                result[name] = {
                    'description': alt['description'],
                    'suitable_for': alt['suitable_for'],
                    'strategy': alt['strategy'].__dict__
                }
        
        return {
            "success": True,
            "alternatives": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取备选策略失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("pricing_api:app", host="0.0.0.0", port=8000, reload=True)