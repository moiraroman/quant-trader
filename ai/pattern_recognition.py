# ============================================================
# ai/pattern_recognition.py — K线形态识别 + 背离检测
# 独立模块，支持多时间框架复用
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
class CandlePattern:
    """检测到的K线形态"""
    tf_label: str
    pattern_name: str
    pattern_type: str          # "反转" / "持续" / "中继" / "震荡"
    direction: str             # "bullish" / "bearish" / "neutral"
    strength: str              # "strong" / "moderate" / "weak"
    reliability: float         # 历史统计胜率 0~1
    bars_involved: list[int]   # 涉及的K线索引
    description: str           # 价格行为描述
    signal: str               # 短期信号（如"可能止跌"）


@dataclass
class Divergence:
    """背离信号"""
    tf_label: str
    indicator: str             # "RSI" / "MACD" / "CCI" / "Stochastic"
    direction: str             # "bullish" / "bearish"
    type_: str                 # "regular" / "hidden" / "exaggerated"
    price_action: str          # 价格行为描述
    indicator_action: str      # 指标行为描述
    description: str           # 完整描述
    strength: str              # "strong" / "moderate" / "weak"


@dataclass
class PatternResult:
    """完整形态识别结果"""
    ticker: str
    tf_label: str
    patterns: list[CandlePattern]
    divergences: list[Divergence]
    # 汇总
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    composite_bias: str = "neutral"  # "bullish" / "bearish" / "neutral"
    composite_strength: float = 0.0  # 0~1


# ============================================================
# K线数据准备
# ============================================================

def _prep_candles(df: pd.DataFrame) -> dict:
    """准备K线数据"""
    if df.empty or len(df) < 3:
        return {}

    closes = df["Close"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values if "Volume" in df.columns else None

    n = len(df)

    def body(i): return abs(closes[i] - opens[i])
    def upper_shadow(i): return highs[i] - max(closes[i], opens[i])
    def lower_shadow(i): return min(closes[i], opens[i]) - lows[i]
    def range_(i): return highs[i] - lows[i]
    def is_bull(i): return closes[i] > opens[i]
    def is_bear(i): return closes[i] < opens[i]
    def bullish_pct(i): return body(i) / range_(i) * 100 if range_(i) > 0 else 0
    def upper_pct(i): return upper_shadow(i) / range_(i) * 100 if range_(i) > 0 else 0
    def lower_pct(i): return lower_shadow(i) / range_(i) * 100 if range_(i) > 0 else 0

    return {
        "closes": closes, "opens": opens, "highs": highs, "lows": lows,
        "volumes": volumes, "n": n,
        "body": body, "upper_shadow": upper_shadow, "lower_shadow": lower_shadow,
        "range_": range_, "is_bull": is_bull, "is_bear": is_bear,
        "bullish_pct": bullish_pct, "upper_pct": upper_pct, "lower_pct": lower_pct,
    }


# ============================================================
# 单根K线形态
# ============================================================

def _detect_single_patterns(c: dict, i: int, tf_label: str) -> list[CandlePattern]:
    """检测单根K线形态"""
    patterns = []
    n = c["n"]

    rng = c["range_"](i)
    bod = c["body"](i)
    up = c["upper_shadow"](i)
    dn = c["lower_shadow"](i)
    bull_pct = c["bullish_pct"](i)
    up_pct = c["upper_pct"](i)
    dn_pct = c["lower_pct"](i)

    if rng == 0:
        return patterns

    # 锤头（Hammer）
    if dn_pct > 60 and up_pct < 15 and bod / rng < 0.35:
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="锤头（Hammer）",
            pattern_type="反转",
            direction="bullish",
            strength="moderate",
            reliability=0.58,
            bars_involved=[i],
            description="下影线超60%，实体小，价格触底反弹",
            signal="可能止跌回升（需确认）",
        ))

    # 倒锤头（Inverted Hammer）
    if up_pct > 60 and dn_pct < 15 and bod / rng < 0.35:
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="倒锤头（Inverted Hammer）",
            pattern_type="反转",
            direction="bullish",
            strength="weak",
            reliability=0.52,
            bars_involved=[i],
            description="上影线超60%，实体小，多头上攻受阻",
            signal="需下一根阳线确认",
        ))

    # 射击之星（Shooting Star）
    if up_pct > 60 and dn_pct < 15 and bod / rng < 0.35:
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="射击之星（Shooting Star）",
            pattern_type="反转",
            direction="bearish",
            strength="moderate",
            reliability=0.57,
            bars_involved=[i],
            description="上影线超60%，实体小，空头反扑",
            signal="可能滞涨回落（需确认）",
        ))

    # 吊颈线（Hanging Man）
    if dn_pct > 60 and up_pct < 15 and bod / rng < 0.35:
        # 和锤头类似，但需要出现在上升趋势中
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="吊颈线（Hanging Man）",
            pattern_type="反转",
            direction="bearish",
            strength="moderate",
            reliability=0.55,
            bars_involved=[i],
            description="形态同锤头但出现在高位，是看跌信号",
            signal="高位出现需警惕（需趋势确认）",
        ))

    # 十字星（Doji）
    if bod / rng < 0.1 and up > 0 and dn > 0:
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="十字星（Doji）",
            pattern_type="震荡",
            direction="neutral",
            strength="moderate",
            reliability=0.50,
            bars_involved=[i],
            description="开盘=收盘，多空均衡，等待方向",
            signal="多空分歧，等待确认K线",
        ))

    # 长脚十字（Long-legged Doji）
    if bod / rng < 0.15 and (up > bod * 2) and (dn > bod * 2):
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="长脚十字（Long-legged Doji）",
            pattern_type="震荡",
            direction="neutral",
            strength="moderate",
            reliability=0.52,
            bars_involved=[i],
            description="上下影线均长，多空激烈博弈",
            signal="变盘信号，关注下一根",
        ))

    # 纺锤线（Spinning Top）
    if 0.1 <= bod / rng < 0.25 and up_pct < 40 and dn_pct < 40:
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="纺锤线（Spinning Top）",
            pattern_type="震荡",
            direction="neutral",
            strength="weak",
            reliability=0.48,
            bars_involved=[i],
            description="上下影线适中，实体小，犹豫信号",
            signal="趋势暂停，观望",
        ))

    # 大阳线/大阴线（Marubozu）
    if bod / rng > 0.9:
        direction = "bullish" if c["is_bull"](i) else "bearish"
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="大阳线（Marubozu）" if direction == "bullish" else "大阴线（Marubozu）",
            pattern_type="持续",
            direction=direction,
            strength="strong",
            reliability=0.62,
            bars_involved=[i],
            description="实体超90%，光头光脚，趋势强烈延续",
            signal="多头/空头主导，趋势延续信号",
        ))

    return patterns


# ============================================================
# 双根K线形态
# ============================================================

def _detect_double_patterns(c: dict, i: int, tf_label: str) -> list[CandlePattern]:
    """检测双根K线形态（i为第二根）"""
    patterns = []
    if i < 1:
        return patterns

    # 吞没形态（Engulfing）
    body1 = c["body"](i - 1)
    body2 = c["body"](i)
    rng2 = c["range_"](i)

    if body1 > 0 and body2 > 0 and rng2 > 0:
        # 看跌吞没
        if c["is_bull"](i - 1) and c["is_bear"](i):
            if (c["opens"][i] < c["closes"][i - 1] and
                c["closes"][i] < c["opens"][i - 1] and
                c["highs"][i] > c["highs"][i - 1]):
                patterns.append(CandlePattern(
                    tf_label=tf_label,
                    pattern_name="看跌吞没（Bearish Engulfing）",
                    pattern_type="反转",
                    direction="bearish",
                    strength="strong",
                    reliability=0.63,
                    bars_involved=[i - 1, i],
                    description="阴线完全吞没前一根阳线，空头反扑",
                    signal="趋势转弱信号（高位更有效）",
                ))

        # 看涨吞没
        if c["is_bear"](i - 1) and c["is_bull"](i):
            if (c["opens"][i] > c["closes"][i - 1] and
                c["closes"][i] > c["opens"][i - 1] and
                c["lows"][i] < c["lows"][i - 1]):
                patterns.append(CandlePattern(
                    tf_label=tf_label,
                    pattern_name="看涨吞没（Bullish Engulfing）",
                    pattern_type="反转",
                    direction="bullish",
                    strength="strong",
                    reliability=0.63,
                    bars_involved=[i - 1, i],
                    description="阳线完全吞没前一根阴线，多头反扑",
                    signal="趋势转强信号（低位更有效）",
                ))

    # 孕线（Harami）
    if body1 > body2:
        if (c["is_bull"](i - 1) and c["is_bear"](i) and
            c["opens"][i] > c["closes"][i - 1] and
            c["closes"][i] < c["opens"][i - 1]):
            patterns.append(CandlePattern(
                tf_label=tf_label,
                pattern_name="看跌孕线（Bearish Harami）",
                pattern_type="反转",
                direction="bearish",
                strength="moderate",
                reliability=0.55,
                bars_involved=[i - 1, i],
                description="大阳线包含小阴线，趋势暂停",
                signal="高位出现需警惕",
            ))
        elif (c["is_bear"](i - 1) and c["is_bull"](i) and
              c["opens"][i] < c["closes"][i - 1] and
              c["closes"][i] > c["opens"][i - 1]):
            patterns.append(CandlePattern(
                tf_label=tf_label,
                pattern_name="看涨孕线（Bullish Harami）",
                pattern_type="反转",
                direction="bullish",
                strength="moderate",
                reliability=0.55,
                bars_involved=[i - 1, i],
                description="大阴线包含小阳线，跌势暂停",
                signal="低位出现可关注",
            ))

    # 贯穿线（Piercing）
    if c["is_bear"](i - 1) and c["is_bull"](i):
        mid = (c["opens"][i - 1] + c["closes"][i - 1]) / 2
        if c["closes"][i] > mid and c["opens"][i] < c["closes"][i - 1]:
            patterns.append(CandlePattern(
                tf_label=tf_label,
                pattern_name="贯穿线（Piercing Line）",
                pattern_type="反转",
                direction="bullish",
                strength="moderate",
                reliability=0.58,
                bars_involved=[i - 1, i],
                description="阳线从阴线中点以上切入，多方反攻",
                signal="低位反转信号",
            ))

    # 乌云盖顶（Dark Cloud Cover）
    if c["is_bull"](i - 1) and c["is_bear"](i):
        mid = (c["opens"][i - 1] + c["closes"][i - 1]) / 2
        if c["closes"][i] < mid and c["opens"][i] > c["closes"][i - 1]:
            patterns.append(CandlePattern(
                tf_label=tf_label,
                pattern_name="乌云盖顶（Dark Cloud Cover）",
                pattern_type="反转",
                direction="bearish",
                strength="moderate",
                reliability=0.57,
                bars_involved=[i - 1, i],
                description="阴线插入阳线实体中点以上，空方反攻",
                signal="高位反转信号",
            ))

    return patterns


# ============================================================
# 三根K线形态
# ============================================================

def _detect_triple_patterns(c: dict, i: int, tf_label: str) -> list[CandlePattern]:
    """检测三根K线形态（i为第三根）"""
    patterns = []
    if i < 2:
        return patterns

    closes = c["closes"]
    opens = c["opens"]
    highs = c["highs"]
    lows = c["lows"]

    # 三乌鸦（Three Black Crows）— 3根连续下跌的阴线
    if all(c["is_bear"](i - j) for j in range(3)):
        # 每根都创新低
        if (closes[i] < closes[i-1] < closes[i-2] and
            closes[i] < opens[i] and opens[i] < closes[i-1]):
            patterns.append(CandlePattern(
                tf_label=tf_label,
                pattern_name="三乌鸦（Three Black Crows）",
                pattern_type="反转",
                direction="bearish",
                strength="strong",
                reliability=0.65,
                bars_involved=[i - 2, i - 1, i],
                description="3根连续阴线，每根都创新低",
                signal="下跌趋势强烈延续（高位更有效）",
            ))

    # 三兵前进（Three White Soldiers）— 3根连续上涨的阳线
    if all(c["is_bull"](i - j) for j in range(3)):
        if (closes[i] > closes[i-1] > closes[i-2] and
            closes[i] > opens[i] > closes[i-1]):
            patterns.append(CandlePattern(
                tf_label=tf_label,
                pattern_name="三兵前进（Three White Soldiers）",
                pattern_type="持续",
                direction="bullish",
                strength="strong",
                reliability=0.65,
                bars_involved=[i - 2, i - 1, i],
                description="3根连续阳线，每根都创新高",
                signal="上升趋势强烈延续",
            ))

    # 黄昏星（Evening Star）— 顶部反转
    if (c["is_bull"](i - 2) and
        (abs(c["closes"][i - 1] - c["opens"][i - 1]) / c["range_"](i - 1)) < 0.3 and
        c["is_bear"](i) and
        c["closes"][i] < (c["closes"][i - 2] + c["opens"][i - 2]) / 2):
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="黄昏星（Evening Star）",
            pattern_type="反转",
            direction="bearish",
            strength="strong",
            reliability=0.62,
            bars_involved=[i - 2, i - 1, i],
            description="大阳→十字→大阴，顶部反转经典形态",
            signal="高位出现强烈看跌",
        ))

    # 晨星（Morning Star）— 底部反转
    if (c["is_bear"](i - 2) and
        (abs(c["closes"][i - 1] - c["opens"][i - 1]) / c["range_"](i - 1)) < 0.3 and
        c["is_bull"](i) and
        c["closes"][i] > (c["closes"][i - 2] + c["opens"][i - 2]) / 2):
        patterns.append(CandlePattern(
            tf_label=tf_label,
            pattern_name="晨星（Morning Star）",
            pattern_type="反转",
            direction="bullish",
            strength="strong",
            reliability=0.62,
            bars_involved=[i - 2, i - 1, i],
            description="大阴→十字→大阳，底部反转经典形态",
            signal="低位出现强烈看涨",
        ))

    # 收敛形态（Tightening Range / Coiling）
    if i >= 4:
        ranges = [c["range_"](i - j) for j in range(5)]
        avg_range = sum(ranges) / 5
        if ranges[0] < avg_range * 0.5:
            # 价格区间持续收缩，突破在即
            direction = "bullish" if c["is_bull"](i) else "bearish"
            patterns.append(CandlePattern(
                tf_label=tf_label,
                pattern_name="收敛整理（Coiling/Tightening）",
                pattern_type="震荡",
                direction="neutral",
                strength="moderate",
                reliability=0.55,
                bars_involved=[i - 4, i],
                description="价格区间持续收缩，突破在即",
                signal="关注突破方向，突破后顺势操作",
            ))

    # 吞没三明治（Bearish Belt Hold / Bullish Belt Hold）
    if i >= 1:
        # 乌云盖顶（已在双根中处理）
        pass

    return patterns


# ============================================================
# 背离检测
# ============================================================

def _calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)
    # EMA方式
    alpha = 2.0 / (period + 1)
    for i in range(1, len(closes)):
        avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    def ema(arr, span):
        alpha = 2.0 / (span + 1)
        out = np.zeros_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _detect_divergences(df: pd.DataFrame, tf_label: str) -> list[Divergence]:
    """检测RSI和MACD背离"""
    divergences = []
    if len(df) < 30:
        return divergences

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    rsi = _calc_rsi(closes, 14)
    macd_line, signal_line, macd_hist = _calc_macd(closes)

    lookback = min(25, len(closes) - 1)

    # ---- RSI背离 ----
    # 找价格高点（局部）
    def _find_local_highs(arr, window=5):
        highs_idx = []
        for i in range(window, len(arr) - window):
            is_high = True
            for j in range(i - window, i + window + 1):
                if j != i and arr[j] >= arr[i]:
                    is_high = False
                    break
            if is_high:
                highs_idx.append(i)
        return highs_idx

    def _find_local_lows(arr, window=5):
        lows_idx = []
        for i in range(window, len(arr) - window):
            is_low = True
            for j in range(i - window, i + window + 1):
                if j != i and arr[j] <= arr[i]:
                    is_low = False
                    break
            if is_low:
                lows_idx.append(i)
        return lows_idx

    recent_closes = closes[-lookback:]
    recent_rsi = rsi[-lookback:]

    price_highs = _find_local_highs(recent_closes, window=4)
    price_lows = _find_local_lows(recent_closes, window=4)

    # RSI顶背离：价格创新高，RSI未跟随
    if len(price_highs) >= 2:
        highest_price_idx = max(price_highs)
        highest_rsi_idx = max(price_highs, key=lambda idx: recent_rsi[idx])
        if (recent_closes[-1] > recent_closes[highest_price_idx] and
            recent_rsi[-1] < recent_rsi[highest_price_idx] and
            highest_price_idx != len(recent_closes) - 1):
            divergences.append(Divergence(
                tf_label=tf_label,
                indicator="RSI(14)",
                direction="bearish",
                type_="regular",
                price_action=f"价格创{lookback}日新高（{recent_closes[highest_price_idx]:.2f}→{recent_closes[-1]:.2f}）",
                indicator_action=f"RSI未跟随（{recent_rsi[highest_price_idx]:.1f}→{recent_rsi[-1]:.1f}）",
                description="价格创新高但RSI下降，顶背离，看空信号",
                strength="moderate",
            ))

    # RSI底背离：价格创新低，RSI未跟随
    if len(price_lows) >= 2:
        lowest_price_idx = min(price_lows)
        lowest_rsi_idx = min(price_lows, key=lambda idx: recent_rsi[idx])
        if (recent_closes[-1] < recent_closes[lowest_price_idx] and
            recent_rsi[-1] > recent_rsi[lowest_price_idx] and
            lowest_price_idx != len(recent_closes) - 1):
            divergences.append(Divergence(
                tf_label=tf_label,
                indicator="RSI(14)",
                direction="bullish",
                type_="regular",
                price_action=f"价格创{lookback}日新低（{recent_closes[lowest_price_idx]:.2f}→{recent_closes[-1]:.2f}）",
                indicator_action=f"RSI未跟随（{recent_rsi[lowest_price_idx]:.1f}→{recent_rsi[-1]:.1f}）",
                description="价格创新低但RSI上升，底背离，看多信号",
                strength="moderate",
            ))

    # ---- MACD背离 ----
    recent_macdh = macd_hist[-lookback:]
    macd_highs = _find_local_highs(recent_macdh, window=4)
    macd_lows = _find_local_lows(recent_macdh, window=4)

    # MACD顶背离
    if macd_highs and len(price_highs):
        mh_idx = max(macd_highs)
        highest_price_in_macdh = max([p for p in price_highs if p <= mh_idx + 4], default=None)
        if (highest_price_in_macdh is not None and
            recent_closes[-1] > recent_closes[highest_price_in_macdh] and
            recent_macdh[-1] < recent_macdh[mh_idx]):
            divergences.append(Divergence(
                tf_label=tf_label,
                indicator="MACD Histogram",
                direction="bearish",
                type_="regular",
                price_action="价格新高",
                indicator_action="MACD柱未跟随创新高",
                description="MACD顶背离，动能衰竭",
                strength="moderate",
            ))

    # MACD底背离
    if macd_lows and len(price_lows):
        ml_idx = min(macd_lows)
        lowest_price_in_macd = min([p for p in price_lows if p >= ml_idx - 4], default=None)
        if (lowest_price_in_macd is not None and
            recent_closes[-1] < recent_closes[lowest_price_in_macd] and
            recent_macdh[-1] > recent_macdh[ml_idx]):
            divergences.append(Divergence(
                tf_label=tf_label,
                indicator="MACD Histogram",
                direction="bullish",
                type_="regular",
                price_action="价格新低",
                indicator_action="MACD柱未跟随创新低",
                description="MACD底背离，底部积累动能",
                strength="moderate",
            ))

    return divergences


# ============================================================
# 主函数
# ============================================================

def recognize_patterns(
    ticker: str,
    df_daily: pd.DataFrame,
    df_weekly: Optional[pd.DataFrame] = None,
    df_monthly: Optional[pd.DataFrame] = None,
    df_h4: Optional[pd.DataFrame] = None,
) -> list[PatternResult]:
    """
    多时间框架形态识别。

    数据策略：
        - 所有数据来自 yfinance（免费真实数据）
        - 如某时间框架数据不足，该框架返回空列表
        - 绝不估算/假设形态
    """
    results = []

    def _analyze_tf(df: pd.DataFrame, tf_label: str) -> PatternResult:
        result = PatternResult(
            ticker=ticker, tf_label=tf_label,
            patterns=[], divergences=[]
        )
        if df.empty or len(df) < 5:
            return result

        c = _prep_candles(df)
        all_patterns = []

        # 扫描最近5根K线
        for offset in range(min(5, c["n"])):
            i = c["n"] - 1 - offset
            all_patterns.extend(_detect_single_patterns(c, i, tf_label))
            all_patterns.extend(_detect_double_patterns(c, i, tf_label))
            all_patterns.extend(_detect_triple_patterns(c, i, tf_label))

        # 去重（同一位置同形态保留最强）
        seen = {}
        for p in all_patterns:
            key = (tuple(p.bars_involved), p.pattern_name)
            if key not in seen or p.reliability > seen[key].reliability:
                seen[key] = p
        result.patterns = list(seen.values())

        # 背离（主要在日线级别）
        if tf_label == "日线":
            result.divergences = _detect_divergences(df, tf_label)

        # 统计
        for p in result.patterns:
            if p.direction == "bullish":
                result.bullish_count += 1
            elif p.direction == "bearish":
                result.bearish_count += 1
            else:
                result.neutral_count += 1

        # 综合偏多/偏空
        total = result.bullish_count + result.bearish_count + result.neutral_count
        if total == 0:
            result.composite_bias = "neutral"
            result.composite_strength = 0.0
        else:
            bull_w = result.bullish_count * 0.6
            bear_w = result.bearish_count * 0.6
            neu_w = result.neutral_count * 0.2
            net = (bull_w - bear_w) / total
            result.composite_strength = min(1.0, abs(net))
            result.composite_bias = "bullish" if net > 0.1 else ("bearish" if net < -0.1 else "neutral")

        return result

    # 各时间框架
    if not df_daily.empty:
        results.append(_analyze_tf(df_daily, "日线"))
    if df_weekly is not None and not df_weekly.empty:
        results.append(_analyze_tf(df_weekly, "周线"))
    if df_monthly is not None and not df_monthly.empty:
        results.append(_analyze_tf(df_monthly, "月线"))
    if df_h4 is not None and not df_h4.empty:
        results.append(_analyze_tf(df_h4, "4H"))

    return results


# ============================================================
# 格式化
# ============================================================

def format_pattern_result(results: list[PatternResult]) -> dict:
    """格式化形态识别结果"""
    all_patterns = []
    all_divergences = []

    composite_bull = 0
    composite_bear = 0
    composite_neutral = 0

    for r in results:
        for p in r.patterns:
            all_patterns.append({
                "时间框架": p.tf_label,
                "形态": p.pattern_name,
                "类型": p.pattern_type,
                "方向": "🐂" if p.direction == "bullish" else ("🐻" if p.direction == "bearish" else "⚖️"),
                "强度": p.strength,
                "胜率": f"{p.reliability:.0%}",
                "信号": p.signal,
                "描述": p.description,
            })
        for d in r.divergences:
            all_divergences.append({
                "时间框架": d.tf_label,
                "指标": d.indicator,
                "类型": d.type_,
                "方向": "🐂" if d.direction == "bullish" else "🐻",
                "价格行为": d.price_action,
                "指标行为": d.indicator_action,
                "描述": d.description,
                "强度": d.strength,
            })
        composite_bull += r.bullish_count
        composite_bear += r.bearish_count
        composite_neutral += r.neutral_count

    return {
        "K线形态": all_patterns,
        "背离信号": all_divergences,
        "汇总": {
            "看多形态数": composite_bull,
            "看空形态数": composite_bear,
            "中性形态数": composite_neutral,
            "综合偏向": "🐂 看多" if composite_bull > composite_bear else ("🐻 看空" if composite_bear > composite_bull else "⚖️ 中性"),
        },
    }
