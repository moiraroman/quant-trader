# ============================================================
# ai/support_resistance.py — 支撑与阻力位识别
# 综合 Pivot Point / 成交量轮廓 / 均线 / 历史高低点
# 数据来源：yfinance 历史K线（免费真实数据）
# ============================================================
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SRLevel:
    """单一支撑/阻力位"""
    price: float
    level_type: str           # "支撑" / "阻力" / "强支撑" / "强阻力" / "心理关口"
    source: str                # "pivot_s", "pivot_r", "volume", "ma50", "ma200", "swing_high", "swing_low", "psychological"
    confidence: float          # 0~1，多个指标共振时更高
    width_pct: float           # 区域宽度（%），不是精确价格
    label: str                  # 描述标签
    breakout_threshold: float  # 突破阈值（%）


@dataclass
class SRAnalysisResult:
    """完整支撑阻力分析"""
    ticker: str
    latest_price: float
    current_price_distance_pct: float  # 价格距离各主要阻力/支撑的百分比
    support_levels: list[SRLevel]
    resistance_levels: list[SRLevel]
    # 关键区域（宽幅区域，而非精确价格）
    demand_zone: tuple[float, float]   # (下限, 上限) 需求区
    supply_zone: tuple[float, float]  # (下限,上限) 供给区
    # 距离最近的关键价位
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    # 综合评估
    outlook: str     # "看多" / "看空" / "震荡" / "接近支撑" / "接近阻力"
    risk_reward_zones: list[str]  # R:R 较好的区域描述


# ============================================================
# 辅助函数
# ============================================================

def _is_psychological(price: float) -> bool:
    """判断是否为心理关口（整数位或xx.00, xx.50等）"""
    rounded = round(price, 0)
    return abs(price - rounded) < price * 0.002 or abs(price - rounded) < 0.5


def _round_sr(price: float, precision: int = 2) -> float:
    """支撑阻力位取整（保留合适精度）"""
    if price > 1000:
        return round(price, 0)
    elif price > 100:
        return round(price, 1)
    else:
        return round(price, precision)


# ============================================================
# 1. Pivot Point（经典/ Camarilla / Fibonacci）
# ============================================================

def _calc_pivot_points(df: pd.DataFrame) -> dict:
    """计算标准 Pivot Point（基于前日HLC）"""
    if len(df) < 3:
        return {}

    h = float(df["High"].iloc[-2])
    l = float(df["Low"].iloc[-2])
    c = float(df["Close"].iloc[-2])
    o = float(df["Open"].iloc[-2]) if "Open" in df.columns else (h + l + c) / 3

    pivot = (h + l + c) / 3

    # 标准Pivot
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    r3 = h + 2 * (pivot - l)
    s3 = l - 2 * (h - pivot)

    # Camarilla
    r4_cam = c + (h - l) * 1.1
    s4_cam = c - (h - l) * 1.1

    # Fibonacci
    r1_fib = pivot + (h - l) * 0.382
    s1_fib = pivot - (h - l) * 0.382
    r2_fib = pivot + (h - l) * 0.618
    s2_fib = pivot - (h - l) * 0.618

    return {
        "pivot": pivot,
        "r1": r1, "r2": r2, "r3": r3,
        "s1": s1, "s2": s2, "s3": s3,
        "camarilla_r": r4_cam, "camarilla_s": s4_cam,
        "fib_r1": r1_fib, "fib_r2": r2_fib,
        "fib_s1": s1_fib, "fib_s2": s2_fib,
    }


# ============================================================
# 2. 摆动高低点（近N日内）
# ============================================================

def _find_swing_levels(df: pd.DataFrame, lookback: int = 60) -> tuple[list[float], list[float]]:
    """识别近 lookback 日内的摆动高点和低点"""
    if len(df) < 5:
        return [], []

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    # 简化：取近 lookback 内的局部高低点
    # 用窗口扫描
    window = 5
    swing_highs = []
    swing_lows = []

    for i in range(window, n - window):
        is_high = True
        is_low = True
        for j in range(i - window, i + window + 1):
            if j == i:
                continue
            if highs[j] >= highs[i]:
                is_high = False
            if lows[j] <= lows[i]:
                is_low = False
        if is_high:
            swing_highs.append(float(highs[i]))
        if is_low:
            swing_lows.append(float(lows[i]))

    # 取最近的3个
    recent_highs = sorted(swing_highs)[-3:] if len(swing_highs) >= 3 else sorted(swing_highs)
    recent_lows = sorted(swing_lows)[:3] if len(swing_lows) >= 3 else sorted(swing_lows)

    return recent_highs, recent_lows


# ============================================================
# 3. 均线支撑/阻力
# ============================================================

def _find_ma_levels(df: pd.DataFrame) -> dict[str, float]:
    """计算各均线当前值（作为隐性支撑/阻力）"""
    closes = df["Close"].dropna()
    result = {}
    for window in [20, 50, 100, 200]:
        if len(closes) >= window:
            ma = closes.rolling(window).mean().iloc[-1]
            if not pd.isna(ma):
                result[f"MA{window}"] = float(ma)
    return result


# ============================================================
# 4. 成交量轮廓（VP）
# ============================================================

def _calc_volume_profile(df: pd.DataFrame, bins: int = 50) -> dict:
    """
    计算成交量加权价格分布（Volume Profile）
    - POC: Point of Control（最大成交量价格）
    - VAH: Value Area High（70%成交量区域上限）
    - VAL: Value Area Low（70%成交量区域下限）
    - LVN: Low Volume Node（薄弱区域）
    """
    if len(df) < 20:
        return {}

    closes = df["Close"].values
    volumes = df["Volume"].values
    highs = df["High"].values
    lows = df["Low"].values

    # 合并所有价格到高低点之间
    all_prices = []
    all_volumes = []
    for i in range(len(df)):
        prices_in_bar = np.linspace(lows[i], highs[i], max(2, int(volumes[i] / 10000) + 1))
        all_prices.extend(prices_in_bar)
        all_volumes.extend([volumes[i] / len(prices_in_bar)] * len(prices_in_bar))

    if not all_prices:
        return {}

    all_prices = np.array(all_prices)
    all_volumes = np.array(all_volumes)

    # 统计各价格区间的成交量
    price_min, price_max = all_prices.min(), all_prices.max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.histogram(all_prices, bins=bin_edges, weights=all_volumes)[0]

    # POC
    poc_idx = np.argmax(bin_volumes)
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    # Value Area (70%)
    total_vol = bin_volumes.sum()
    cumsum = np.cumsum(bin_volumes)
    target = total_vol * 0.70

    vah_price = None
    val_price = None
    for i in range(len(bin_volumes)):
        if cumsum[i] >= target * 0.15 and vah_price is None:
            val_price = bin_edges[i]
        if cumsum[i] >= target * 0.85 and vah_price is None:
            vah_price = bin_edges[i]

    if vah_price is None:
        vah_price = price_max
    if val_price is None:
        val_price = price_min

    # LVN（低成交量区域）
    avg_vol_per_bin = bin_volumes.mean()
    low_vol_bins = np.where(bin_volumes < avg_vol_per_bin * 0.5)[0]
    lvn_prices = []
    for idx in low_vol_bins:
        lvn_prices.append((bin_edges[idx] + bin_edges[idx + 1]) / 2)

    return {
        "poc": float(poc_price),
        "vah": float(vah_price),
        "val": float(val_price),
        "lvn": [float(p) for p in lvn_prices[:5]],  # 最多5个
    }


# ============================================================
# 5. 历史高点/低点
# ============================================================

def _find_historical_levels(df: pd.DataFrame, price: float) -> dict:
    """识别历史重要价位（52周高低、近一年高低、年初至今高低）"""
    if len(df) < 5:
        return {}

    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]

    result = {}

    # 52周（252个交易日）
    if len(df) >= 252:
        result["52w_high"] = float(highs.iloc[-252:].max())
        result["52w_low"] = float(lows.iloc[-252:].min())
        # 是否接近52周高低
        dist_52w_high = (result["52w_high"] - price) / price * 100
        result["dist_52w_high_pct"] = dist_52w_high

    # 近一年高低
    if len(df) >= 252:
        result["1y_high"] = float(highs.iloc[-252:].max())
        result["1y_low"] = float(lows.iloc[-252:].min())
    elif len(df) >= 60:
        result["6m_high"] = float(highs.iloc[-126:].max())
        result["6m_low"] = float(lows.iloc[-126:].min())

    # 年初至今（YTD）
    ytd_start_idx = 0
    for i, idx in enumerate(df.index):
        if str(idx.year) == str(df.index[-1].year):
            ytd_start_idx = i
            break
    if ytd_start_idx < len(df) - 1:
        ytd_df = df.iloc[ytd_start_idx:]
        result["ytd_high"] = float(ytd_df["High"].max())
        result["ytd_low"] = float(ytd_df["Low"].min())

    return result


# ============================================================
# 6. 整合所有价位
# ============================================================

def _build_sr_levels(
    df: pd.DataFrame,
    price: float,
    price_min: float,
    price_max: float,
) -> SRAnalysisResult:
    """整合所有方法得出的支撑/阻力位，过滤噪音"""

    all_levels: list[SRLevel] = []

    # --- Pivot Point ---
    pivots = _calc_pivot_points(df)
    if pivots:
        for label, val in [
            ("R1", pivots.get("r1")), ("R2", pivots.get("r2")), ("R3", pivots.get("r3")),
            ("S1", pivots.get("s1")), ("S2", pivots.get("s2")), ("S3", pivots.get("s3")),
        ]:
            if val and price_min * 0.5 < val < price_max * 1.5:
                lt = "阻力" if label.startswith("R") else "支撑"
                conf = 0.6 if label in ("R1", "S1") else (0.5 if label in ("R2", "S2") else 0.4)
                all_levels.append(SRLevel(
                    price=_round_sr(val),
                    level_type=lt,
                    source="pivot_" + label.lower(),
                    confidence=conf,
                    width_pct=0.3,
                    label=f"Pivot {label} ({_round_sr(val)})",
                    breakout_threshold=0.3,
                ))

    # --- 摆动高低点 ---
    swing_highs, swing_lows = _find_swing_levels(df, lookback=60)
    for sh in swing_highs[-3:]:
        if price_min * 0.8 < sh < price_max * 1.2:
            all_levels.append(SRLevel(
                price=_round_sr(sh),
                level_type="阻力",
                source="swing_high",
                confidence=0.65,
                width_pct=0.5,
                label=f"摆动高点 ({_round_sr(sh)})",
                breakout_threshold=0.5,
            ))
    for sl in swing_lows[:3]:
        if price_min * 0.8 < sl < price_max * 1.2:
            all_levels.append(SRLevel(
                price=_round_sr(sl),
                level_type="支撑",
                source="swing_low",
                confidence=0.65,
                width_pct=0.5,
                label=f"摆动低点 ({_round_sr(sl)})",
                breakout_threshold=0.5,
            ))

    # --- 均线 ---
    ma_levels = _find_ma_levels(df)
    for ma_name, ma_val in ma_levels.items():
        if price_min * 0.8 < ma_val < price_max * 1.2:
            lt = "支撑" if price > ma_val else "阻力"
            all_levels.append(SRLevel(
                price=_round_sr(ma_val),
                level_type=lt,
                source=ma_name.lower(),
                confidence=0.55,
                width_pct=0.2,
                label=f"{ma_name} ({_round_sr(ma_val)})",
                breakout_threshold=0.2,
            ))

    # --- 成交量轮廓 ---
    vp = _calc_volume_profile(df)
    if vp:
        for label, val in [("POC", vp.get("poc")), ("VAH", vp.get("vah")), ("VAL", vp.get("val"))]:
            if val and price_min * 0.8 < val < price_max * 1.2:
                lt = "阻力" if label == "VAH" else ("支撑" if label == "VAL" else "中性")
                if label == "POC":
                    lt = "强支撑" if price > val else "强阻力"
                all_levels.append(SRLevel(
                    price=_round_sr(val),
                    level_type=lt,
                    source="volume_" + label.lower(),
                    confidence=0.7 if label == "POC" else 0.6,
                    width_pct=0.3 if label == "POC" else 0.5,
                    label=f"VP {label} ({_round_sr(val)})",
                    breakout_threshold=0.3,
                ))

    # --- 历史高低点 ---
    hist = _find_historical_levels(df, price)
    for label, val in hist.items():
        if isinstance(val, float) and price_min * 0.5 < val < price_max * 1.5:
            lt = "阻力" if val > price else "支撑"
            conf = 0.75
            if "high" in label:
                lt = "强阻力"
                conf = 0.8
            elif "low" in label:
                lt = "强支撑"
                conf = 0.8
            all_levels.append(SRLevel(
                price=_round_sr(val),
                level_type=lt,
                source=label,
                confidence=conf,
                width_pct=1.0,
                label=f"{label} ({_round_sr(val)})",
                breakout_threshold=1.0,
            ))

    # --- 心理关口 ---
    for thresh in [50, 100, 200, 300, 400, 500]:
        if abs(price - thresh) < price * 0.05:
            lt = "支撑" if price > thresh else "阻力"
            all_levels.append(SRLevel(
                price=float(thresh),
                level_type="心理关口",
                source="psychological",
                confidence=0.5,
                width_pct=0.5,
                label=f"心理关口 ({thresh})",
                breakout_threshold=1.0,
            ))

    # ---- 分类 ----
    support_levels = sorted(
        [l for l in all_levels if l.level_type in ("支撑", "强支撑")],
        key=lambda x: x.price, reverse=True,
    )
    resistance_levels = sorted(
        [l for l in all_levels if l.level_type in ("阻力", "强阻力")],
        key=lambda x: x.price,
    )

    # ---- 合并附近价位（去重） ----
    def merge_nearby(levels: list[SRLevel], threshold_pct: float = 1.0) -> list[SRLevel]:
        """合并距离过近的价位"""
        if not levels:
            return []
        merged = [levels[0]]
        for lvl in levels[1:]:
            last = merged[-1]
            diff = abs(lvl.price - last.price) / last.price * 100
            if diff < threshold_pct:
                # 合并：保留置信度更高的
                if lvl.confidence > last.confidence:
                    merged[-1] = lvl
                else:
                    merged[-1].confidence = max(lvl.confidence, last.confidence)
            else:
                merged.append(lvl)
        return merged

    support_levels = merge_nearby(support_levels)
    resistance_levels = merge_nearby(resistance_levels)

    # 取前3个最近的
    nearest_s = None
    for s in reversed(support_levels):
        if s.price < price:
            nearest_s = s.price
            break

    nearest_r = None
    for r in resistance_levels:
        if r.price > price:
            nearest_r = r.price
            break

    # 供给/需求区
    demand_zone = (
        nearest_s * 0.995 if nearest_s else price * 0.92,
        nearest_s * 1.005 if nearest_s else price * 0.95,
    ) if nearest_s else (price * 0.92, price * 0.95)

    supply_zone = (
        nearest_r * 0.995 if nearest_r else price * 1.05,
        nearest_r * 1.005 if nearest_r else price * 1.08,
    ) if nearest_r else (price * 1.05, price * 1.08)

    # 评估
    dist_to_s = ((price - nearest_s) / price * 100) if nearest_s else None
    dist_to_r = ((nearest_r - price) / price * 100) if nearest_r else None

    if dist_to_s is not None and dist_to_s < 1.0:
        outlook = "接近支撑"
    elif dist_to_r is not None and dist_to_r < 1.0:
        outlook = "接近阻力"
    elif dist_to_s is not None and dist_to_r is not None and dist_to_s < dist_to_r:
        outlook = "震荡偏多"
    elif dist_to_s is not None and dist_to_r is not None and dist_to_r < dist_to_s:
        outlook = "震荡偏空"
    else:
        outlook = "中性震荡"

    return SRAnalysisResult(
        ticker=df.get("ticker", "UNKNOWN"),
        latest_price=price,
        current_price_distance_pct=0,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        demand_zone=demand_zone,
        supply_zone=supply_zone,
        nearest_support=nearest_s,
        nearest_resistance=nearest_r,
        outlook=outlook,
        risk_reward_zones=[],
    )


# ============================================================
# 主函数
# ============================================================

def analyze_support_resistance(
    ticker: str,
    latest_price: float,
    fetcher,
) -> SRAnalysisResult:
    """
    完整支撑阻力分析。

    数据策略：
        - 日线数据取1年（252根），用于Pivot/均线/摆动/历史高低
        - 成交量轮廓使用近3个月数据（波动区更清晰）
        - 所有数据来自 yfinance（免费真实数据）

    数据降级策略：
        - 如 yfinance 数据不足，标注"数据不足"
        - 绝不估算/假设价格
    """
    from datetime import datetime

    # 获取日线数据
    df = fetcher.download_history(ticker, period="1y", interval="1d")
    if df.empty:
        logger.warning(f"[SR] {ticker} 无历史数据")
        return SRAnalysisResult(
            ticker=ticker,
            latest_price=latest_price,
            current_price_distance_pct=0,
            support_levels=[],
            resistance_levels=[],
            demand_zone=(latest_price * 0.95, latest_price * 0.97),
            supply_zone=(latest_price * 1.03, latest_price * 1.05),
            nearest_support=None,
            nearest_resistance=None,
            outlook="数据不足",
            risk_reward_zones=[],
        )

    df["ticker"] = ticker
    price_min = latest_price * 0.7
    price_max = latest_price * 1.3

    result = _build_sr_levels(df, latest_price, price_min, price_max)

    # 计算距离百分比
    if result.nearest_support:
        result.current_price_distance_pct = round(
            (latest_price - result.nearest_support) / latest_price * 100, 2
        )
    if result.nearest_resistance:
        dist_r = round((result.nearest_resistance - latest_price) / latest_price * 100, 2)
        # R:R 区域描述
        if result.nearest_support:
            rr = round(result.nearest_resistance - latest_price, 2)
            sl = round(latest_price - result.nearest_support, 2)
            if sl > 0:
                rr_ratio = round(rr / sl, 2) if sl != 0 else 0
                result.risk_reward_zones.append(
                    f"当前→阻力区间 R:R ≈ 1:{rr_ratio}（潜在涨幅{_round_sr(rr)}，潜在跌幅{_round_sr(sl)}）"
                )

    return result


# ============================================================
# 格式化输出
# ============================================================

def format_sr_result(result: SRAnalysisResult) -> dict:
    """格式化输出供WebUI展示"""
    def _level_to_dict(lvl: SRLevel) -> dict:
        return {
            "价格": _round_sr(lvl.price),
            "类型": lvl.level_type,
            "来源": lvl.source,
            "置信度": f"{lvl.confidence:.0%}",
            "宽度%": f"{lvl.width_pct:.1f}%",
            "突破阈值": f"{lvl.breakout_threshold:.1f}%",
            "标签": lvl.label,
        }

    return {
        "标的": result.ticker,
        "最新价格": _round_sr(result.latest_price),
        "最近支撑": _round_sr(result.nearest_support) if result.nearest_support else "无数据",
        "最近阻力": _round_sr(result.nearest_resistance) if result.nearest_resistance else "无数据",
        "距支撑": f"{result.current_price_distance_pct:.2f}%" if result.current_price_distance_pct else "N/A",
        "需求区": f"{_round_sr(result.demand_zone[0])} ~ {_round_sr(result.demand_zone[1])}",
        "供给区": f"{_round_sr(result.supply_zone[0])} ~ {_round_sr(result.supply_zone[1])}",
        "综合判断": result.outlook,
        "R:R机会": result.risk_reward_zones,
        "支撑位列表": [_level_to_dict(s) for s in result.support_levels[:5]],
        "阻力位列表": [_level_to_dict(r) for r in result.resistance_levels[:5]],
    }
