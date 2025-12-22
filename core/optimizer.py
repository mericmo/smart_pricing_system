# core/optimizer.py
from typing import List
import math

def minimal_price_from_margin(cost: float, min_margin: float) -> float:
    if min_margin >= 1.0:
        raise ValueError("min_margin must be < 1.0")
    return cost / (1.0 - min_margin)

def generate_discount_candidates(min_discount: float, max_discount: float, step: float = 0.1):
    # discounts are fractions, 0.0 ~ 1.0
    candidates = []
    d = min_discount
    while d <= max_discount + 1e-9:
        candidates.append(round(d, 4))
        d += step
    return sorted(set(candidates))

def greedy_schedule_solver(stock: int, cost: float, p0: float, stages_hours: List[float], forecast, discount_candidates: List[float], min_price: float):
    """
    简单贪心：在每阶段选择尽可能高的价格（即尽可能小的折扣），
    若在最后阶段累计预测销量仍不足以售罄，则在最后阶段上使用最大折扣。
    返回 list of dicts per stage with keys: discount, price, expected_sales
    """
    schedule = []
    remaining = stock
    K = len(stages_hours)

    # 先计算每个阶段在每个折扣下的预测销量
    pred_table = []
    for k in range(K):
        row = []
        for d in discount_candidates:
            price = max(p0 * (1 - d), min_price)
            expected = forecast.predict_stage_sales(k, price, stages_hours[k])
            row.append((d, price, expected))
        pred_table.append(row)

    cum_expected = 0.0
    for k in range(K):
        # 从小折扣（高价）到大折扣（低价）选择第一可行使得
        chosen = None
        for d, price, expected in pred_table[k]:
            # 选择最小折扣（最大价格），但如果在未来阶段累加也许可以完成售罄，保守选最高价
            chosen = {"stage": k, "discount": d, "price": price, "expected_sales": expected}
            break
        if chosen is None:
            d, price, expected = pred_table[k][-1]
            chosen = {"stage": k, "discount": d, "price": price, "expected_sales": expected}
        schedule.append(chosen)
        cum_expected += chosen["expected_sales"]
    # 如果最后 cum_expected < stock，尝试在后段逐步增大折扣直到满足
    if cum_expected < stock:
        shortage = stock - cum_expected
        # 从最后一阶段开始放宽折扣
        for k in reversed(range(K)):
            options = pred_table[k]
            # current index
            cur_d = schedule[k]["discount"]
            # find index
            idx = next((i for i, (d, p, e) in enumerate(options) if abs(d - cur_d) < 1e-6), len(options)-1)
            # try larger discounts
            for j in range(idx+1, len(options)):
                d, price, expected = options[j]
                # apply change and recompute cum_expected
                new_cum = cum_expected - schedule[k]["expected_sales"] + expected
                if new_cum >= stock or j == len(options)-1:
                    schedule[k] = {"stage": k, "discount": d, "price": price, "expected_sales": expected}
                    cum_expected = new_cum
                    break
            if cum_expected >= stock:
                break
    return schedule