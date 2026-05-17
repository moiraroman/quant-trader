# ============================================================
# ai/classical_theory.py — 经典理论分析模块
# 道氏理论(Dow Theory) + 艾略特波浪(Elliott Wave)估算
# 数据策略：基于yfinance历史价格，标注主观性，绝不伪装精确
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
class DowTheorySignal:
    """道氏理论信号"""
    primary_trend: str           # "上升"/"下降"/"不确定"
    trend_stage: str             # "积累"/"上涨"/"派发"/"下跌"
    confirmation: str            # "确认"/"未确认"/"背离"
    volume_confirmation: str     # "确认"/"未确认"
    key_evidence: list[str]      # 关键证据
    reliability: str             # "高"/"中"/"低"


@dataclass
class ElliottWaveEstimate:
    """艾略特波浪估算（高度主观，明确标注）"""
    current_wave: str            # 当前可能所处的浪
    wave_count: str              # 波浪计数描述
    confidence: str              # "高"/"中"/"低"/"猜测"
    next_wave_prediction: str    # 下一浪预测
    key_fib_levels: list[str]    # 关键斐波那契位
    disclaimer: str              # 免责声明


@dataclass
class ClassicalTheoryResult:
    """完整经典理论分析结果"""
    ticker: str
    current_price: float

    # 道氏理论
    dow_theory: DowTheorySignal

    # 艾略特波浪
    elliott_wave: ElliottWaveEstimate

    # 综合
    composite_outlook: str
    composite_confidence: str

    # 缺少的数据
    missing_data: list[str] = field(default_factory=list)


# ============================================================
# 道氏理论分析
# ============================================================

def analyze_dow_theory(df: pd.DataFrame, ticker: str) -> DowTheorySignal:
    """
    基于价格数据应用道氏理论。

    道氏理论核心原则：
        1. 价格反映一切
        2. 市场有三种趋势：主要（数月-数年）、次要（数周-数月）、日常（数日）
        3. 主要趋势分三阶段：积累→上涨→派发（牛市）；派发→下跌→恐慌（熊市）
        4. 两种指数必须相互确认（工业+运输）
        5. 成交量确认趋势
        6. 趋势持续直到明确反转信号

    注意：
        - 道氏理论是定性分析，非精确量化
        - 需要DJIA和DJTA两个指数确认（我们只有单个标的）
        - 明确标注局限性
    """
    if df.empty or len(df) < 50:
        return DowTheorySignal(
            primary_trend="数据不足",
            trend_stage="未知",
            confirmation="无法判断",
            volume_confirmation="无法判断",
            key_evidence=["历史数据不足（需至少50根K线）"],
            reliability="低",
        )

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values if "Volume" in df.columns else None

    evidence = []

    # 1. 判断主要趋势（用200日均线）
    ma50 = pd.Series(closes).rolling(50).mean().values
    ma200 = pd.Series(closes).rolling(200).mean().values

    current = closes[-1]
    ma50_current = ma50[-1] if not np.isnan(ma50[-1]) else current
    ma200_current = ma200[-1] if not np.isnan(ma200[-1]) else current

    if current > ma50_current > ma200_current:
        primary = "上升"
        evidence.append(f"价格({current:.2f})>MA50({ma50_current:.2f})>MA200({ma200_current:.2f})，多头排列")
    elif current < ma50_current < ma200_current:
        primary = "下降"
        evidence.append(f"价格({current:.2f})<MA50({ma50_current:.2f})<MA200({ma200_current:.2f})，空头排列")
    else:
        primary = "不确定/震荡"
        evidence.append(f"均线排列混乱，趋势不明确")

    # 2. 趋势阶段判断
    # 用最近20根K线的HH/HL或LH/LL判断
    recent_highs = highs[-20:]
    recent_lows = lows[-20:]

    if len(recent_highs) >= 5:
        # 检查是否形成更高的高点和更高的低点
        hh = all(recent_highs[i] <= recent_highs[i+1] for i in range(len(recent_highs)-5, len(recent_highs)-1))
        hl = all(recent_lows[i] <= recent_lows[i+1] for i in range(len(recent_lows)-5, len(recent_lows)-1))

        if hh and hl:
            stage = "上涨阶段（Higher Highs + Higher Lows）"
            evidence.append("近期形成更高高点和更高低点，上升趋势确认")
        elif not hh and not hl:
            stage = "下跌阶段（Lower Highs + Lower Lows）"
            evidence.append("近期形成更低高点和更低低点，下降趋势确认")
        else:
            stage = "震荡/转折阶段"
            evidence.append("高低点形态不一致，可能处于震荡或转折")
    else:
        stage = "数据不足"

    # 3. 成交量确认
    vol_confirm = "无法判断"
    if volumes is not None and len(volumes) >= 20:
        recent_vol = np.mean(volumes[-5:])
        prev_vol = np.mean(volumes[-20:-5])
        price_change = (closes[-1] - closes[-5]) / closes[-5] * 100

        if price_change > 0 and recent_vol > prev_vol * 1.2:
            vol_confirm = "确认（上涨放量）"
            evidence.append("近期上涨伴随放量，成交量确认趋势")
        elif price_change > 0 and recent_vol < prev_vol * 0.8:
            vol_confirm = "未确认（上涨缩量）"
            evidence.append("近期上涨但缩量，成交量未确认")
        elif price_change < 0 and recent_vol > prev_vol * 1.2:
            vol_confirm = "确认（下跌放量）"
            evidence.append("近期下跌伴随放量，成交量确认趋势")
        else:
            vol_confirm = "中性"
    else:
        evidence.append("成交量数据不足，无法确认")

    # 4. 确认状态（单标的局限性）
    confirmation = "未确认（单标的分析，缺乏指数间确认）"
    evidence.append("⚠️ 道氏理论要求工业指数和运输指数相互确认，单标的分析有局限性")

    reliability = "中" if len(evidence) >= 3 else "低"

    return DowTheorySignal(
        primary_trend=primary,
        trend_stage=stage,
        confirmation=confirmation,
        volume_confirmation=vol_confirm,
        key_evidence=evidence,
        reliability=reliability,
    )


# ============================================================
# 艾略特波浪估算
# ============================================================

def estimate_elliott_wave(df: pd.DataFrame, ticker: str) -> ElliottWaveEstimate:
    """
    艾略特波浪理论估算。

    ⚠️ 重要声明：
        - 波浪计数高度主观，不同分析师结论可能完全不同
        - 此处仅提供基于价格结构的粗略估算
        - 绝不伪装精确，明确标注置信度

    艾略特波浪基础：
        - 完整周期：5浪上升（1-2-3-4-5）+ 3浪调整（A-B-C）
        - 浪3通常最长，浪4不进入浪1区域
        - 斐波那契比率：浪2≈0.618浪1，浪3≈1.618浪1，浪5≈浪1
    """
    if df.empty or len(df) < 100:
        return ElliottWaveEstimate(
            current_wave="数据不足",
            wave_count="无法判断",
            confidence="无",
            next_wave_prediction="无",
            key_fib_levels=[],
            disclaimer="数据不足（需至少100根K线）",
        )

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    # 找主要趋势方向（用50日均线）
    ma50 = pd.Series(closes).rolling(50).mean().values
    current = closes[-1]
    trend = "up" if current > ma50[-1] else "down"

    # 找显著的波段高低点（简化版）
    # 使用局部极值
    def find_swing_points(prices, window=10):
        """找摆动点"""
        highs_idx = []
        lows_idx = []
        for i in range(window, len(prices) - window):
            # 局部高点
            if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                highs_idx.append(i)
            # 局部低点
            if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                lows_idx.append(i)
        return highs_idx, lows_idx

    swing_highs, swing_lows = find_swing_points(closes)

    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return ElliottWaveEstimate(
            current_wave="无法识别",
            wave_count="摆动点不足",
            confidence="低",
            next_wave_prediction="无",
            key_fib_levels=[],
            disclaimer="价格结构不清晰，无法可靠计数",
        )

    # 粗略判断当前位置
    # 如果最近是上升趋势中的回调
    recent_highs = [closes[i] for i in swing_highs[-3:]]
    recent_lows = [closes[i] for i in swing_lows[-3:]]

    if trend == "up":
        if len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[-2]:
            # 可能处于浪3或浪5
            if len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[-2]:
                current_wave = "可能处于浪3或浪5（上升趋势中）"
                confidence = "低"
                next_pred = "若当前为浪3，后续有浪4回调和浪5上涨；若为浪5，后续可能进入ABC调整"
            else:
                current_wave = "可能处于浪2回调或浪4回调"
                confidence = "低"
                next_pred = "回调结束后可能继续原趋势"
        else:
            current_wave = "趋势不明确"
            confidence = "低"
            next_pred = "无法可靠预测"
    else:
        current_wave = "可能处于下跌浪或调整浪"
        confidence = "低"
        next_pred = "下跌趋势中，谨慎操作"

    # 斐波那契关键位（基于近期波段）
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        last_swing_high = recent_highs[-1]
        last_swing_low = recent_lows[-1]
        range_ = last_swing_high - last_swing_low

        fib_levels = [
            f"0.0% (高点): {last_swing_high:.2f}",
            f"23.6%: {last_swing_high - range_ * 0.236:.2f}",
            f"38.2%: {last_swing_high - range_ * 0.382:.2f}",
            f"50.0%: {last_swing_high - range_ * 0.5:.2f}",
            f"61.8%: {last_swing_high - range_ * 0.618:.2f}",
            f"78.6%: {last_swing_high - range_ * 0.786:.2f}",
            f"100.0% (低点): {last_swing_low:.2f}",
        ]
    else:
        fib_levels = ["摆动点不足，无法计算斐波那契位"]

    return ElliottWaveEstimate(
        current_wave=current_wave,
        wave_count=f"识别到 {len(swing_highs)} 个高点, {len(swing_lows)} 个低点",
        confidence=confidence,
        next_wave_prediction=next_pred,
        key_fib_levels=fib_levels,
        disclaimer="⚠️ 艾略特波浪计数高度主观，不同分析师结论可能完全不同。此处仅为基于价格结构的粗略估算，不构成交易建议。",
    )


# ============================================================
# 主函数
# ============================================================

def analyze_classical_theory(
    ticker: str,
    current_price: float,
    fetcher,
    period: str = "1y",
    interval: str = "1d",
) -> ClassicalTheoryResult:
    """
    完整经典理论分析。

    数据策略：
        - 道氏理论：基于价格+成交量数据
        - 艾略特波浪：基于价格结构，明确标注主观性
        - 缺失数据标注"缺少"
    """
    missing = []

    try:
        df = fetcher.download_history(ticker, period=period, interval=interval)
        if df.empty:
            missing.append(f"{ticker}历史数据")
            return _empty_classical_result(ticker, current_price, missing)
    except Exception as e:
        logger.warning(f"[ClassicalTheory] {ticker} 数据获取失败: {e}")
        missing.append(f"{ticker}历史数据")
        return _empty_classical_result(ticker, current_price, missing)

    # 道氏理论
    dow = analyze_dow_theory(df, ticker)

    # 艾略特波浪
    elliott = estimate_elliott_wave(df, ticker)

    # 综合
    if dow.primary_trend == "上升" and "浪3" in elliott.current_wave:
        outlook = "道氏理论看多 + 波浪理论可能处于主升浪，趋势较强"
        conf = "中"
    elif dow.primary_trend == "下降":
        outlook = "道氏理论看空，建议谨慎"
        conf = "中"
    else:
        outlook = "经典理论信号不一致或不确定，建议结合其他分析"
        conf = "低"

    return ClassicalTheoryResult(
        ticker=ticker,
        current_price=current_price,
        dow_theory=dow,
        elliott_wave=elliott,
        composite_outlook=outlook,
        composite_confidence=conf,
        missing_data=missing,
    )


def _empty_classical_result(ticker: str, price: float, missing: list) -> ClassicalTheoryResult:
    return ClassicalTheoryResult(
        ticker=ticker,
        current_price=price,
        dow_theory=DowTheorySignal(
            primary_trend="数据不足", trend_stage="未知", confirmation="无法判断",
            volume_confirmation="无法判断", key_evidence=[], reliability="低",
        ),
        elliott_wave=ElliottWaveEstimate(
            current_wave="数据不足", wave_count="无法判断", confidence="无",
            next_wave_prediction="无", key_fib_levels=[],
            disclaimer="数据不足",
        ),
        composite_outlook="数据不足",
        composite_confidence="无",
        missing_data=missing,
    )


# ============================================================
# 格式化输出
# ============================================================

def format_classical_result(result: ClassicalTheoryResult) -> dict:
    """格式化经典理论分析结果供WebUI展示"""
    d = result.dow_theory
    e = result.elliott_wave

    return {
        "标的": result.ticker,
        "当前价格": result.current_price,
        "道氏理论": {
            "主要趋势": d.primary_trend,
            "趋势阶段": d.trend_stage,
            "指数确认": d.confirmation,
            "成交量确认": d.volume_confirmation,
            "关键证据": d.key_evidence,
            "可靠性": d.reliability,
        },
        "艾略特波浪": {
            "当前浪": e.current_wave,
            "波浪计数": e.wave_count,
            "置信度": e.confidence,
            "下一浪预测": e.next_wave_prediction,
            "斐波那契关键位": e.key_fib_levels,
            "⚠️声明": e.disclaimer,
        },
        "综合判断": {
            " outlook": result.composite_outlook,
            "置信度": result.composite_confidence,
        },
        "缺少数据": result.missing_data,
    }
