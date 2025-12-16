# core/forecast.py
from typing import Dict
import numpy as np

class ElasticityForecast:
    """
    简单的基线销量 + 价格弹性预测模型。
    base_rates: dict mapping stage_index -> baseline sales rate per hour (在参考价 p0 下)
    price_elasticity: epsilon (正值)，q ~ base * (p/p0)^(-epsilon)
    """
    def __init__(self, base_rates: Dict[int, float], price_elasticity: float = 1.0, p0: float = 1.0):
        self.base_rates = base_rates.copy()
        self.epsilon = float(price_elasticity)
        self.p0 = float(p0)

    def predict_stage_sales(self, stage_index: int, price: float, duration_hours: float) -> float:
        base = self.base_rates.get(stage_index, 0.0)
        if price <= 0:
            return 0.0
        factor = (price / self.p0) ** (-self.epsilon)
        return max(0.0, base * factor * duration_hours)

    def update_base_rate(self, stage_index: int, observed_rate_per_hour: float, alpha: float = 0.4):
        old = self.base_rates.get(stage_index, 0.0)
        self.base_rates[stage_index] = alpha * observed_rate_per_hour + (1 - alpha) * old

    @classmethod
    def from_history(cls, tx_df, product_code, promotion_start, promotion_end, time_segments, p0):
        """
        根据历史交易数据粗略估计每个阶段的基线速率（件/小时）。
        tx_df: 包含 '商品编码', '交易时间', '销售数量', '折扣' 或 '单价' 的 DataFrame
        """
        # 建立每条交易对应的小时（0-23）
        # 将时间窗口分段并统计历史平均销售速率
        import pandas as pd
        from math import ceil
        if tx_df is None or tx_df.empty:
            # 经验默认值：若无历史，给小的常数基线
            return cls({i: 1.0 for i in range(time_segments)}, price_elasticity=1.0, p0=p0)

        start = int(promotion_start.split(":")[0])
        end = int(promotion_end.split(":")[0])
        total_hours = (end - start) if end > start else (24 - start + end)
        seg_hours = total_hours / time_segments
        base_rates = {}
        # 以交易时间提取小时（本例假设交易时间为 pd.Timestamp）
        tx = tx_df.copy()
        if "交易时间" in tx.columns:
            tx["hour"] = pd.to_datetime(tx["交易时间"]).dt.hour
        else:
            tx["hour"] = pd.to_datetime(tx.get("日期")).dt.hour
        # 过滤商品
        tx = tx[tx["商品编码"] == product_code]
        # 统计每个段的平均每小时销量
        for i in range(time_segments):
            seg_start_h = (start + i * seg_hours)
            seg_end_h = (start + (i + 1) * seg_hours)
            # 包含小时值落在 seg_start_h 到 seg_end_h 的交易（近似）
            hours = list(range(int(seg_start_h), int(seg_end_h) + 1))
            seg_tx = tx[tx["hour"].isin(hours)]
            total_qty = seg_tx["销售数量"].sum() if not seg_tx.empty else 0.0
            # 估计为 每小时销量（平均）
            hours_len = max(1, len(hours))
            base_rates[i] = total_qty / max(1, len(pd.date_range(start=tx_df["日期"].min(), end=tx_df["日期"].max(), freq="D"))) / hours_len
            # 若数据过稀疏，给最小值
            if base_rates[i] <= 0:
                base_rates[i] = 0.5
        return cls(base_rates, price_elasticity=1.0, p0=p0)