"""
pricing_api.py

FastAPI-based HTTP API for the smart pricing system MVP.

Endpoints:
- POST /generate_strategy
  Request JSON:
    {
      "product_code": "8006148",
      "initial_stock": 60,
      "original_price": 35.0,       # optional, generator will use history if missing
      "cost_price": 12.0,           # optional
      "promotion_start": "20:00",
      "promotion_end": "22:00",
      "min_discount": 0.0,
      "max_discount": 0.6,
      "time_segments": 4,
      "min_margin": 0.1
    }
  Response: generated strategy as JSON.

- POST /report_sales
  Request JSON:
    {
      "strategy_id": "<id>",
      "timestamp": "2025-12-12T20:15:00",
      "product_code": "8006148",
      "quantity_sold": 5,
      "actual_price": 28.0,
      "discount_applied": 0.2,
      "remaining_stock": 55
    }
  Response: { "adjusted": bool, "strategy": <new_strategy_or_null>, "summary": {...} }

- GET /strategy/{strategy_id}
  Response: stored strategy JSON or 404.

Run:
  uvicorn pricing_api:app --reload --port 8000

Note: this file depends on core/*.py modules in the repository (pricing_strategy_generator, real_time_adjuster).
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import os
import json

import pandas as pd

from core.pricing_strategy_generator import PricingStrategyGenerator
from core.real_time_adjuster import RealTimeAdjuster, SalesUpdate

# --- Load historical data (if exists) ---
DATA_PATH = "data/historical_transactions.csv"
if os.path.exists(DATA_PATH):
    try:
        historical_df = pd.read_csv(DATA_PATH, encoding="utf-8", parse_dates=["日期", "交易时间"], dtype={"商品编码": str, "门店编码": str})
    except Exception:
        # graceful fallback
        historical_df = pd.DataFrame()
else:
    historical_df = pd.DataFrame()

# --- Instantiate core components ---
strategy_generator = PricingStrategyGenerator(historical_df)
real_time_adjuster = RealTimeAdjuster(strategy_generator)

# In-memory strategy store: strategy_id -> strategy dict (as generated)
STRATEGY_STORE: Dict[str, Dict[str, Any]] = {}

# Directory to persist strategies
STRATEGY_DIR = "strategies"
os.makedirs(STRATEGY_DIR, exist_ok=True)

# --- FastAPI app ---
app = FastAPI(title="Smart Pricing System API", version="0.1")

# Allow common CORS for testing/demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --- Request/response models ---
class GenerateStrategyRequest(BaseModel):
    product_code: str
    initial_stock: int = Field(..., ge=0)
    original_price: Optional[float] = None
    cost_price: Optional[float] = None
    promotion_start: str = Field("20:00", regex=r"^\d{2}:\d{2}$")
    promotion_end: str = Field("22:00", regex=r"^\d{2}:\d{2}$")
    min_discount: float = Field(0.0, ge=0.0, le=1.0)
    max_discount: float = Field(0.6, ge=0.0, le=1.0)
    time_segments: int = Field(4, ge=1, le=12)
    min_margin: float = Field(0.1, ge=0.0, lt=1.0)


class ReportSalesRequest(BaseModel):
    strategy_id: str
    timestamp: datetime
    product_code: str
    quantity_sold: int = Field(..., ge=0)
    actual_price: float = Field(..., ge=0.0)
    discount_applied: float = Field(..., ge=0.0, le=1.0)
    remaining_stock: Optional[int] = None


# --- Helpers ---
def strategy_to_dict(strategy) -> Dict[str, Any]:
    """
    Convert PricingStrategy dataclass-like object to a JSON-serializable dict.
    The PricingStrategyGenerator.save_strategy writes a dataclass via asdict; here we ensure types are native.
    """
    # It should already be dataclass; try to convert using json module fallback.
    try:
        # attempt dataclass -> dict
        sd = strategy.__dict__.copy()
    except Exception:
        sd = dict(strategy)
    # ensure no non-serializable objects inside (datetime etc.)
    # pricing_schedule and evaluation are simple types per generator implementation
    return sd


def persist_strategy(strategy_dict: Dict[str, Any]):
    sid = strategy_dict.get("strategy_id") or strategy_dict.get("strategy_id")
    if not sid:
        return
    filepath = os.path.join(STRATEGY_DIR, f"./output/strategy_{sid}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(strategy_dict, f, ensure_ascii=False, indent=2)


# --- API endpoints ---
@app.post("/generate_strategy")
def generate_strategy(payload: GenerateStrategyRequest):
    """
    Generate a pricing strategy for a single SKU in one store (MVP).
    """
    # Basic input validation
    if payload.min_discount > payload.max_discount:
        raise HTTPException(status_code=400, detail="min_discount must be <= max_discount")

    # Delegate to generator
    strat = strategy_generator.generate_pricing_strategy(
        product_code=payload.product_code,
        initial_stock=payload.initial_stock,
        promotion_start=payload.promotion_start,
        promotion_end=payload.promotion_end,
        min_discount=payload.min_discount,
        max_discount=payload.max_discount,
        time_segments=payload.time_segments,
        min_margin=payload.min_margin
    )

    sdict = strategy_to_dict(strat)
    # persist
    STRATEGY_STORE[sdict["strategy_id"]] = sdict
    persist_strategy(sdict)

    return {"ok": True, "strategy_id": sdict["strategy_id"], "strategy": sdict}


@app.post("/report_sales")
def report_sales(payload: ReportSalesRequest):
    """
    Report sales updates for a strategy and optionally trigger adjustment.
    """
    sid = payload.strategy_id
    if sid not in STRATEGY_STORE:
        # try to load from disk
        filepath = os.path.join(STRATEGY_DIR, f"strategy_{sid}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                STRATEGY_STORE[sid] = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="strategy_id not found")

    # Build SalesUpdate
    su = SalesUpdate(
        timestamp=payload.timestamp,
        product_code=payload.product_code,
        quantity_sold=payload.quantity_sold,
        actual_price=payload.actual_price,
        discount_applied=payload.discount_applied,
        remaining_stock=payload.remaining_stock if payload.remaining_stock is not None else None
    )
    # record
    real_time_adjuster.record_sales(sid, su)

    # attempt to adjust: need a strategy object. The generator returns a dataclass; here we operate on dict.
    # For simplicity, we reconstruct a minimal object expected by RealTimeAdjuster.check_and_adjust: it expects
    # strategy with attributes strategy_id, promotion_start, promotion_end, time_segments, pricing_schedule, evaluation, original_price, cost_price, max_discount, initial_stock
    sdict = STRATEGY_STORE[sid]
    # Create a lightweight namespace object
    class _S:
        pass
    strat_obj = _S()
    for k, v in sdict.items():
        setattr(strat_obj, k, v)

    # check & adjust
    adjusted = real_time_adjuster.check_and_adjust(strat_obj, payload.timestamp)

    if adjusted:
        # adjusted is a PricingStrategy dataclass-like object; convert and persist
        adj_dict = strategy_to_dict(adjusted)
        STRATEGY_STORE[adj_dict["strategy_id"]] = adj_dict
        persist_strategy(adj_dict)
        return {"ok": True, "adjusted": True, "strategy": adj_dict}
    else:
        summary = real_time_adjuster.get_adjustment_summary(sid)
        return {"ok": True, "adjusted": False, "summary": summary}


@app.get("/strategy/{strategy_id}")
def get_strategy(strategy_id: str):
    if strategy_id in STRATEGY_STORE:
        return {"ok": True, "strategy": STRATEGY_STORE[strategy_id]}
    filepath = os.path.join(STRATEGY_DIR, f"strategy_{strategy_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            s = json.load(f)
        STRATEGY_STORE[strategy_id] = s
        return {"ok": True, "strategy": s}
    raise HTTPException(status_code=404, detail="strategy_id not found")


@app.get("/health")
def health():
    return {"ok": True, "status": "ready", "have_history": not historical_df.empty}


# --- If run directly, start uvicorn (development) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pricing_api:app", host="0.0.0.0", port=8000, reload=True)