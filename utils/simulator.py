# utils/simulator.py
import numpy as np
from typing import List, Tuple


def simulate_schedule_random(schedule: List[dict], stock: int, demand_noise: float = 0.3, seed: int = 42):
    """
    基于 schedule 中 expected_sales 进行泊松抽样（lambda = expected_sales）
    返回 sold_per_stage list 和 remaining_stock
    """
    rng = np.random.default_rng(seed)
    remaining = stock
    sold = []
    for s in schedule:
        lam = max(0.0, s.get("expected_sales", 0.0))
        # 添加噪声到 lambda（log-normal multiplicative noise）
        jitter = rng.normal(1.0, demand_noise)
        lam_j = max(0.0, lam * jitter)
        qty = int(rng.poisson(lam_j))
        qty = min(qty, remaining)
        sold.append(qty)
        remaining -= qty
        if remaining <= 0:
            break
    return sold, remaining
