# ============================================================
# ai/volume_profile.py — 成交量分布分析模块
# VPVR(Volume Profile Visible Range)、VWAP、LVN、POC、VAH/VAL
# 数据来源：yfinance 历史K线（免费真实数据）
# ============================================================
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class VolumeNode:
    """单一成交量节点"""
    price: float
    volume: float
    volume_pct: float          # 占总成交量%
    is_poc: bool               # 是否为POC
    is_vah: bool               # 是否为VAH
    is_val: bool               # 是否为VAL
    is_lvn: bool               # 是否为LVN（低成交量节点）


@dataclass
class VolumeProfile:
    """成交量分布结果"""
    ticker: str
    price_min: float
    price_max: float
    poc: float                 # Point of Control（最大成交量价格）
    vah: float                 # Value Area High（70%区域上限）
    val: float                 # Value Area Low（70%区域下限）
    vwap: float                # Volume Weighted Average Price
    vwap_std: float            # VWAP标准差（波动带）
    nodes: list[VolumeNode]    # 所有节点
    lvn_zones: list[tuple[float, float]]  # 低成交量区域
    high_volume_zones: list[tuple[float, float]]  # 高成交量区域


@dataclass
class VWAP_Analysis:
    """VWAP分析"""
    vwap: float
    price_vs_vwap: str         # "above"/"below"/"at"
    distance_pct: float        # 价格偏离VWAP%
    std_bands: dict            # 1σ/2σ/3σ价格带
    interpretation: str        # 解读


@dataclass
class VPResult:
    """完整成交量分析结果"""
    ticker: str
    current_price: float
    profile: VolumeProfile
    vwap_analysis: VWAP_Analysis
    # 交易信号
    near_poc: bool             # 是否接近POC
    near_vah: bool             # 是否接近VAH
    near_val: bool             # 是否接近VAL
    in_value_area: bool        # 是否在价值区内
    # 解读
    outlook: str
    key_levels: list[str]
    missing_data: list[str] = field(default_factory=list)


# ============================================================
# 成交量分布计算（VPVR）
# ============================================================

def calculate_volume_profile(
    df: pd.DataFrame,
    ticker: str,
    bins: int = 50,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    """
    计算成交量分布（Volume Profile）。

    算法：
        1. 将价格区间分为N个bin
        2. 统计每个bin内的成交量（按K线高低点加权分配）
        3. 找出POC（最大成交量bin）
        4. 从POC向两侧扩展，直到累计成交量达到70% → VAH/VAL
        5. 识别LVN（成交量显著低于平均的bin）

    参数:
        df: OHLCV DataFrame
        ticker: 标的代码
        bins: 价格分档数
        value_area_pct: 价值区成交量占比（默认70%）
    """
    if df.empty or len(df) < 10:
        raise ValueError("数据不足，无法计算成交量分布")

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values

    price_min = lows.min()
    price_max = highs.max()

    if price_max <= price_min:
        raise ValueError("价格范围无效")

    # 创建价格bin
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_volumes = np.zeros(bins)

    # 将每根K线的成交量分配到价格bin
    for i in range(len(df)):
        bar_low = lows[i]
        bar_high = highs[i]
        bar_vol = volumes[i] if not np.isnan(volumes[i]) else 0

        if bar_high <= bar_low or bar_vol <= 0:
            continue

        # 找出该K线覆盖的bin
        low_idx = max(0, np.searchsorted(bin_edges, bar_low, side="left") - 1)
        high_idx = min(bins - 1, np.searchsorted(bin_edges, bar_high, side="right"))

        if low_idx > high_idx:
            continue

        # 均匀分配成交量（简化：实际应按时间在bin内的占比分配）
        n_bins_covered = high_idx - low_idx + 1
        vol_per_bin = bar_vol / n_bins_covered

        for j in range(low_idx, high_idx + 1):
            bin_volumes[j] += vol_per_bin

    total_volume = bin_volumes.sum()
    if total_volume == 0:
        raise ValueError("总成交量为0")

    # POC（最大成交量节点）
    poc_idx = np.argmax(bin_volumes)
    poc_price = bin_centers[poc_idx]

    # 价值区（从POC向两侧扩展，累计达到70%）
    target_volume = total_volume * value_area_pct
    cumsum_left = 0
    cumsum_right = 0
    vah_idx = poc_idx
    val_idx = poc_idx

    left_idx = poc_idx - 1
    right_idx = poc_idx + 1
    current_vol = bin_volumes[poc_idx]

    while current_vol < target_volume and (left_idx >= 0 or right_idx < bins):
        left_vol = bin_volumes[left_idx] if left_idx >= 0 else 0
        right_vol = bin_volumes[right_idx] if right_idx < bins else 0

        # 优先扩展成交量大的一侧
        if left_vol >= right_vol and left_idx >= 0:
            current_vol += left_vol
            val_idx = left_idx
            left_idx -= 1
        elif right_idx < bins:
            current_vol += right_vol
            vah_idx = right_idx
            right_idx += 1
        else:
            break

    vah_price = bin_centers[vah_idx]
    val_price = bin_centers[val_idx]

    # LVN（低成交量节点：低于平均50%）
    avg_vol = bin_volumes.mean()
    lvn_threshold = avg_vol * 0.5
    lvn_indices = np.where(bin_volumes < lvn_threshold)[0]

    # 合并连续的LVN
    lvn_zones = []
    if len(lvn_indices) > 0:
        start = lvn_indices[0]
        prev = lvn_indices[0]
        for idx in lvn_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                lvn_zones.append((bin_centers[start], bin_centers[prev]))
                start = idx
                prev = idx
        lvn_zones.append((bin_centers[start], bin_centers[prev]))

    # 高成交量区域（高于平均150%）
    hv_threshold = avg_vol * 1.5
    hv_indices = np.where(bin_volumes > hv_threshold)[0]
    hv_zones = []
    if len(hv_indices) > 0:
        start = hv_indices[0]
        prev = hv_indices[0]
        for idx in hv_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                hv_zones.append((bin_centers[start], bin_centers[prev]))
                start = idx
                prev = idx
        hv_zones.append((bin_centers[start], bin_centers[prev]))

    # 构建节点列表
    nodes = []
    for i in range(bins):
        nodes.append(VolumeNode(
            price=round(bin_centers[i], 2),
            volume=round(bin_volumes[i], 0),
            volume_pct=round(bin_volumes[i] / total_volume * 100, 2),
            is_poc=(i == poc_idx),
            is_vah=(i == vah_idx),
            is_val=(i == val_idx),
            is_lvn=(bin_volumes[i] < lvn_threshold),
        ))

    # VWAP
    vwap = np.average(closes, weights=volumes)
    # VWAP标准差（简化）
    vwap_var = np.average((closes - vwap) ** 2, weights=volumes)
    vwap_std = np.sqrt(vwap_var)

    return VolumeProfile(
        ticker=ticker,
        price_min=round(price_min, 2),
        price_max=round(price_max, 2),
        poc=round(poc_price, 2),
        vah=round(vah_price, 2),
        val=round(val_price, 2),
        vwap=round(vwap, 2),
        vwap_std=round(vwap_std, 2),
        nodes=nodes,
        lvn_zones=[(round(z[0], 2), round(z[1], 2)) for z in lvn_zones],
        high_volume_zones=[(round(z[0], 2), round(z[1], 2)) for z in hv_zones],
    )


# ============================================================
# VWAP 分析
# ============================================================

def analyze_vwap(
    df: pd.DataFrame,
    current_price: float,
    ticker: str,
) -> VWAP_Analysis:
    """
    VWAP（成交量加权平均价）分析。

    VWAP 解读：
        - 价格在VWAP上方：多头控制
        - 价格在VWAP下方：空头控制
        - 价格回归VWAP：均值回归信号
        - 远离VWAP 2σ以上：超买/超卖
    """
    closes = df["Close"].values
    volumes = df["Volume"].values

    vwap = np.average(closes, weights=volumes)
    vwap_var = np.average((closes - vwap) ** 2, weights=volumes)
    vwap_std = np.sqrt(vwap_var)

    distance_pct = (current_price - vwap) / vwap * 100

    if current_price > vwap * 1.01:
        vs = "above"
        interp = "价格在VWAP上方，多头控制"
    elif current_price < vwap * 0.99:
        vs = "below"
        interp = "价格在VWAP下方，空头控制"
    else:
        vs = "at"
        interp = "价格在VWAP附近，均衡状态"

    # 标准差带
    std_bands = {
        "+3σ": round(vwap + 3 * vwap_std, 2),
        "+2σ": round(vwap + 2 * vwap_std, 2),
        "+1σ": round(vwap + 1 * vwap_std, 2),
        "VWAP": round(vwap, 2),
        "-1σ": round(vwap - 1 * vwap_std, 2),
        "-2σ": round(vwap - 2 * vwap_std, 2),
        "-3σ": round(vwap - 3 * vwap_std, 2),
    }

    # 超买超卖判断
    if distance_pct > 5:
        interp += "，远离VWAP（超买）"
    elif distance_pct < -5:
        interp += "，远离VWAP（超卖）"

    return VWAP_Analysis(
        vwap=round(vwap, 2),
        price_vs_vwap=vs,
        distance_pct=round(distance_pct, 2),
        std_bands=std_bands,
        interpretation=interp,
    )


# ============================================================
# 主函数
# ============================================================

def analyze_volume_profile(
    ticker: str,
    current_price: float,
    fetcher,
    period: str = "3mo",
    interval: str = "1d",
) -> VPResult:
    """
    完整成交量分布分析。

    参数:
        ticker: 标的代码
        current_price: 当前价格
        fetcher: 数据获取器
        period: 数据周期（默认3个月，VPVR通常用可见范围）
        interval: K线周期

    数据策略：
        - 使用yfinance历史数据
        - 日线数据计算VPVR
        - 缺失标注"缺少"
    """
    missing = []

    try:
        df = fetcher.download_history(ticker, period=period, interval=interval)
        if df.empty:
            missing.append(f"{ticker}历史数据")
            return _empty_result(ticker, current_price, missing)
    except Exception as e:
        logger.warning(f"[VolumeProfile] {ticker} 数据获取失败: {e}")
        missing.append(f"{ticker}历史数据")
        return _empty_result(ticker, current_price, missing)

    # 计算成交量分布
    try:
        profile = calculate_volume_profile(df, ticker, bins=50)
    except Exception as e:
        logger.warning(f"[VolumeProfile] {ticker} VP计算失败: {e}")
        missing.append("成交量分布计算失败")
        return _empty_result(ticker, current_price, missing)

    # VWAP分析
    vwap = analyze_vwap(df, current_price, ticker)

    # 判断位置
    near_poc = abs(current_price - profile.poc) / current_price < 0.01
    near_vah = abs(current_price - profile.vah) / current_price < 0.01
    near_val = abs(current_price - profile.val) / current_price < 0.01
    in_va = profile.val <= current_price <= profile.vah

    # 解读
    if near_poc:
        outlook = "价格接近POC（最大成交量区），支撑/阻力强"
    elif near_vah:
        outlook = "价格接近VAH（价值区上限），面临阻力"
    elif near_val:
        outlook = "价格接近VAL（价值区下限），面临支撑"
    elif in_va:
        outlook = "价格在价值区内，正常波动"
    elif current_price > profile.vah:
        outlook = "价格突破VAH上方，强势（关注是否可持续）"
    else:
        outlook = "价格跌破VAL下方，弱势（关注是否可回归）"

    # 关键价位
    key_levels = [
        f"POC {profile.poc:.2f}（最大成交量）",
        f"VAH {profile.vah:.2f}（价值区上限）",
        f"VAL {profile.val:.2f}（价值区下限）",
        f"VWAP {vwap.vwap:.2f}（成交量加权均价）",
    ]
    if profile.lvn_zones:
        key_levels.append(f"LVN区域: {', '.join([f'{z[0]:.2f}-{z[1]:.2f}' for z in profile.lvn_zones[:2]])}")

    return VPResult(
        ticker=ticker,
        current_price=current_price,
        profile=profile,
        vwap_analysis=vwap,
        near_poc=near_poc,
        near_vah=near_vah,
        near_val=near_val,
        in_value_area=in_va,
        outlook=outlook,
        key_levels=key_levels,
        missing_data=missing,
    )


def _empty_result(ticker: str, price: float, missing: list) -> VPResult:
    """返回空结果"""
    return VPResult(
        ticker=ticker,
        current_price=price,
        profile=VolumeProfile(
            ticker=ticker, price_min=0, price_max=0,
            poc=0, vah=0, val=0, vwap=0, vwap_std=0,
            nodes=[], lvn_zones=[], high_volume_zones=[],
        ),
        vwap_analysis=VWAP_Analysis(
            vwap=0, price_vs_vwap="unknown", distance_pct=0,
            std_bands={}, interpretation="数据不足",
        ),
        near_poc=False, near_vah=False, near_val=False,
        in_value_area=False, outlook="数据不足",
        key_levels=[], missing_data=missing,
    )


# ============================================================
# 格式化输出
# ============================================================

def format_vp_result(result: VPResult) -> dict:
    """格式化成交量分析结果供WebUI展示"""
    p = result.profile
    v = result.vwap_analysis

    return {
        "标的": result.ticker,
        "当前价格": result.current_price,
        "成交量分布": {
            "价格范围": f"{p.price_min:.2f} ~ {p.price_max:.2f}",
            "POC": f"{p.poc:.2f}（最大成交量价格）",
            "VAH": f"{p.vah:.2f}（价值区上限70%）",
            "VAL": f"{p.val:.2f}（价值区下限70%）",
            "价值区宽度": f"{((p.vah - p.val) / result.current_price * 100):.1f}%",
        },
        "VWAP": {
            "VWAP": f"{v.vwap:.2f}",
            "价格位置": "上方" if v.price_vs_vwap == "above" else ("下方" if v.price_vs_vwap == "below" else "附近"),
            "偏离%": f"{v.distance_pct:+.2f}%",
            "标准差带": v.std_bands,
            "解读": v.interpretation,
        },
        "位置判断": {
            "接近POC": "是" if result.near_poc else "否",
            "接近VAH": "是" if result.near_vah else "否",
            "接近VAL": "是" if result.near_val else "否",
            "在价值区内": "是" if result.in_value_area else "否",
        },
        "综合判断": result.outlook,
        "关键价位": result.key_levels,
        "LVN区域": [f"{z[0]:.2f} ~ {z[1]:.2f}" for z in p.lvn_zones[:3]],
        "缺少数据": result.missing_data,
    }
