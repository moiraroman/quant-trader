# ============================================================
# ai/dynamic_weights.py — 动态权重调整
# 功能：根据VIX/波动率/趋势强度自动调整技术/宏观权重
# ============================================================
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DynamicWeightResult:
    """动态权重分析结果"""
    ticker: str
    analysis_time: str

    # 市场状态
    vix_level: float = 0.0
    vix_regime: str = "normal"  # "low"/"normal"/"high"/"extreme"
    atr_percent: float = 0.0
    trend_strength: float = 0.0  # ADX或自定义趋势强度

    # 默认权重
    base_tech_weight: float = 0.70
    base_macro_weight: float = 0.30

    # 调整后权重
    adjusted_tech_weight: float = 0.70
    adjusted_macro_weight: float = 0.30
    adjusted_sentiment_weight: float = 0.0  # 从宏观中拆分

    # 调整因子
    vix_adjustment: float = 0.0   # VIX导致的调整
    trend_adjustment: float = 0.0  # 趋势强度导致的调整
    vol_adjustment: float = 0.0    # 波动率导致的调整

    # 调整说明
    adjustment_reason: str = ""

    # 各模块最终权重（供策略使用）
    module_weights: dict = field(default_factory=dict)

    missing_data: list = field(default_factory=list)


# ============================================================
# 权重调整逻辑
# ============================================================

def _get_vix_regime(vix: float) -> tuple[str, float]:
    """
    VIX分级及对应的权重调整。
    调整逻辑：
      - VIX极低(<12)：趋势行情，技术面权重↑
      - VIX正常(12-20)：默认权重
      - VIX高(20-30)：不确定性增加，宏观权重↑
      - VIX极端(>30)：恐慌/危机模式，宏观+情绪权重↑↑，技术面↓
    """
    if vix < 12:
        return "low", 0.05      # 技术面+5%
    elif vix < 20:
        return "normal", 0.0    # 不变
    elif vix < 30:
        return "high", -0.08    # 技术面-8%，宏观+8%
    elif vix < 40:
        return "extreme", -0.15  # 技术面-15%，宏观+15%
    else:
        return "crisis", -0.20   # 技术面-20%，宏观+20%


def _get_trend_adjustment(adx: float, trend_direction: str) -> float:
    """
    根据趋势强度调整权重。
    ADX > 25 强趋势 → 技术面权重↑
    ADX < 20 弱趋势 → 技术面权重↓，宏观权重↑
    """
    if adx > 35:
        return 0.08   # 强趋势，技术+8%
    elif adx > 25:
        return 0.04   # 中等趋势，技术+4%
    elif adx < 15:
        return -0.05  # 无趋势，技术-5%
    else:
        return 0.0


def _get_vol_adjustment(atr_pct: float, hist_atr_pct: float) -> float:
    """
    根据当前波动率相对历史水平调整。
    当前ATR% > 历史均值1.5倍 → 高波动，降低技术权重
    """
    if hist_atr_pct <= 0:
        return 0.0
    ratio = atr_pct / hist_atr_pct
    if ratio > 2.0:
        return -0.10
    elif ratio > 1.5:
        return -0.05
    elif ratio < 0.5:
        return 0.05   # 低波动，技术+5%
    return 0.0


# ============================================================
# 主分析入口
# ============================================================

def calculate_dynamic_weights(
    ticker: str,
    latest_price: float,
    fetcher,
    vix_ticker: str = "^VIX",
) -> DynamicWeightResult:
    """
    计算动态权重。

    默认权重：技术70% / 宏观30%
    调整后权重范围：技术40%-85% / 宏观15%-60%

    数据源：
      - yfinance VIX数据（^VIX，免费）
      - yfinance 标的ATR数据（免费）
    """
    from datetime import datetime

    result = DynamicWeightResult(
        ticker=ticker,
        analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        base_tech_weight=0.70,
        base_macro_weight=0.30,
    )

    # 1. 获取VIX
    try:
        vix_df = fetcher.download_history(vix_ticker, period="30d", interval="1d")
        if vix_df is not None and not vix_df.empty:
            result.vix_level = round(float(vix_df["Close"].iloc[-1]), 2)
        else:
            result.missing_data.append("VIX数据获取失败")
    except Exception as e:
        logger.warning(f"[DynamicWeights] VIX获取失败: {e}")
        result.missing_data.append("VIX数据异常")

    # 2. 获取标的ATR和ADX
    try:
        df = fetcher.download_history(ticker, period="90d", interval="1d")
        if df is not None and not df.empty and len(df) >= 20:
            # ATR计算（14日）
            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df["Close"].shift())
            low_close = np.abs(df["Low"] - df["Close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            result.atr_percent = round(atr.iloc[-1] / latest_price * 100, 2) if latest_price > 0 else 0.0

            # 历史ATR%均值（60日）
            hist_atr_pct = (atr / df["Close"] * 100).rolling(60).mean().iloc[-1]

            # ADX近似（简化版）
            # 使用价格变动幅度作为趋势强度代理
            returns = df["Close"].pct_change().dropna()
            result.trend_strength = round(abs(returns.iloc[-20:].mean()) / (returns.iloc[-20:].std() + 1e-6) * np.sqrt(20), 2)
        else:
            result.missing_data.append("标的ATR数据不足")
    except Exception as e:
        logger.warning(f"[DynamicWeights] ATR计算失败: {e}")
        result.missing_data.append("ATR计算异常")

    # 3. 计算调整
    vix_regime, vix_adj = _get_vix_regime(result.vix_level)
    result.vix_regime = vix_regime
    result.vix_adjustment = vix_adj

    trend_adj = _get_trend_adjustment(result.trend_strength, "")
    result.trend_adjustment = trend_adj

    vol_adj = 0.0
    if result.atr_percent > 0:
        try:
            vol_adj = _get_vol_adjustment(result.atr_percent, hist_atr_pct)
            result.vol_adjustment = vol_adj
        except:
            pass

    # 4. 应用调整（限制范围）
    total_adj = vix_adj + trend_adj + vol_adj
    new_tech = result.base_tech_weight + total_adj

    # 限制在合理范围
    new_tech = max(0.40, min(0.85, new_tech))
    new_macro = 1.0 - new_tech

    result.adjusted_tech_weight = round(new_tech, 2)
    result.adjusted_macro_weight = round(new_macro, 2)

    # 5. 构建调整说明
    reasons = []
    if vix_adj != 0:
        direction = "降低" if vix_adj < 0 else "提升"
        reasons.append(f"VIX={result.vix_level}({vix_regime})→技术面{direction}{abs(vix_adj):.0%}")
    if trend_adj != 0:
        direction = "降低" if trend_adj < 0 else "提升"
        reasons.append(f"趋势强度={result.trend_strength:.1f}→技术面{direction}{abs(trend_adj):.0%}")
    if vol_adj != 0:
        direction = "降低" if vol_adj < 0 else "提升"
        reasons.append(f"ATR%={result.atr_percent:.2f}%→技术面{direction}{abs(vol_adj):.0%}")

    if reasons:
        result.adjustment_reason = "; ".join(reasons)
    else:
        result.adjustment_reason = "市场环境正常，使用默认权重"

    # 6. 模块级权重分配
    # 技术内部分配
    result.module_weights = {
        "multi_timeframe": round(new_tech * 0.25, 3),      # 多时间框架
        "support_resistance": round(new_tech * 0.15, 3),   # 支撑阻力
        "pattern_recognition": round(new_tech * 0.15, 3),  # 形态识别
        "volume_profile": round(new_tech * 0.15, 3),       # 成交量分布
        "derivatives": round(new_tech * 0.15, 3),          # 衍生品
        "classical_theory": round(new_tech * 0.10, 3),     # 经典理论
        "pattern_match": round(new_tech * 0.05, 3),        # 模式匹配
        # 宏观内部分配
        "macro_scanner": round(new_macro * 0.35, 3),       # 宏观扫描
        "sentiment": round(new_macro * 0.25, 3),           # 情绪
        "institutional_flows": round(new_macro * 0.20, 3), # 机构资金流
        "macro_policy": round(new_macro * 0.20, 3),        # 宏观政策
    }

    return result


# ============================================================
# 格式化输出
# ============================================================

def format_dynamic_weights_result(result: DynamicWeightResult) -> dict:
    """格式化动态权重结果为字典（供WebUI）"""
    return {
        "标的": result.ticker,
        "市场环境": {
            "VIX": result.vix_level,
            "VIX分级": result.vix_regime,
            "ATR%": f"{result.atr_percent:.2f}%",
            "趋势强度": result.trend_strength,
        },
        "权重调整": {
            "默认技术": f"{result.base_tech_weight:.0%}",
            "默认宏观": f"{result.base_macro_weight:.0%}",
            "调整后技术": f"{result.adjusted_tech_weight:.0%}",
            "调整后宏观": f"{result.adjusted_macro_weight:.0%}",
            "调整原因": result.adjustment_reason,
        },
        "调整因子": {
            "VIX调整": f"{result.vix_adjustment:+.0%}",
            "趋势调整": f"{result.trend_adjustment:+.0%}",
            "波动率调整": f"{result.vol_adjustment:+.0%}",
        },
        "模块权重": {k: f"{v:.1%}" for k, v in result.module_weights.items()},
        "缺少数据": result.missing_data,
    }
