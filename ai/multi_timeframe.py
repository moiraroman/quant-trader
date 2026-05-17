# ============================================================
# ai/multi_timeframe.py — 多时间框架分析
# 分析月/周/日/4H四个时间维度的趋势、均线、动能、形态
# 数据来源：yfinance（历史K线），免费真实数据
# 4H数据：yfinance支持"60m"interval，最多60天，取最近60天
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
class MTFTrend:
    """单一时间框架的趋势摘要"""
    tf_label: str           # "月线", "周线", "日线", "4H"
    trend_type: str         # "上升", "下降", "横盘"
    confidence: float        # 0~1，趋势置信度
    hh_hl_check: str        # 更高高点/更高低点/更低高点/更低低点/未确认
    ma_bullish: bool        # 短期 > 长期均线
    ma_alignment: list[str] # 哪些均线多头排列: ["MA5>MA20", "MA20>MA50"]
    volume_profile: str      # "放量上涨"/"缩量整理"/"放量下跌"/"异常"
    last_3_closes: list[float] # 最近3根收盘价


@dataclass
class MTFIndicator:
    """单一时间框架的技术指标"""
    tf_label: str
    # 均线
    ma5: Optional[float]
    ma20: Optional[float]
    ma50: Optional[float]
    ma200: Optional[float]
    ema12: Optional[float]
    ema26: Optional[float]
    # 动能
    rsi: Optional[float]
    macd_hist: Optional[float]
    macd_signal: Optional[float]
    adx: Optional[float]
    atr: Optional[float]
    atr_percent: Optional[float]  # ATR / 价格 * 100
    # 当前价格位置
    price_vs_ma5: str        # "above"/"below"
    price_vs_ma20: str
    price_vs_ma50: str
    price_vs_ma200: str
    # 斜率
    ma20_slope_5d: Optional[float]  # 近5天MA20斜率 (%)


@dataclass
class MTFCandlePattern:
    """单一时间框架检测到的K线形态"""
    tf_label: str
    pattern_name: str        # "锤头", "吞没", "十字星", "射击之星", "乌云盖顶", ...
    direction: str           # "bullish"/"bearish"
    strength: str            # "strong"/"moderate"/"weak"
    price_action: str        # 对趋势的判断
    bars_involved: list[int] # 涉及的K线索引


@dataclass
class MTFAnalysisResult:
    """完整多时间框架分析结果"""
    ticker: str
    latest_price: float
    analysis_time: str
    consensus_trend: str = ""           # "上涨"/"下跌"/"震荡"
    consensus_confidence: float = 0.0   # 0~1
    timeframes: dict[str, MTFTrend]     = field(default_factory=dict)
    indicators: dict[str, MTFIndicator] = field(default_factory=dict)
    patterns: list[MTFCandlePattern]  = field(default_factory=list)
    bullish_signals: list[str]          = field(default_factory=list)
    bearish_signals: list[str]          = field(default_factory=list)
    divergences: list[str]              = field(default_factory=list)


# ============================================================
# 指标计算
# ============================================================

def _calc_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def _calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period).mean()
    return adx, atr


# ============================================================
# 趋势判断
# ============================================================

def _judge_trend(df: pd.DataFrame, tf_label: str) -> MTFTrend:
    """判断单一时间框架的趋势类型"""
    closes = df["Close"].dropna()
    highs = df["High"].dropna()
    lows = df["Low"].dropna()
    volumes = df["Volume"].dropna()

    if len(closes) < 20:
        return MTFTrend(
            tf_label=tf_label,
            trend_type="数据不足",
            confidence=0,
            hh_hl_check="未确认",
            ma_bullish=False,
            ma_alignment=[],
            volume_profile="数据不足",
            last_3_closes=[],
        )

    # 最近N根K线用于趋势判断
    lookback = min(60, len(closes))
    c = closes.iloc[-lookback:]

    # MA计算
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()

    last = float(closes.iloc[-1])
    last_3 = closes.iloc[-3:].tolist()

    # 均线排列
    ma_alignment = []
    if len(ma5) > 0 and len(ma20) > 0:
        if ma5.iloc[-1] > ma20.iloc[-1]:
            ma_alignment.append("MA5>MA20")
        else:
            ma_alignment.append("MA5<MA20")
    if len(ma20) > 0 and len(ma50) > 0:
        if ma20.iloc[-1] > ma50.iloc[-1]:
            ma_alignment.append("MA20>MA50")
        else:
            ma_alignment.append("MA20<MA50")
    if len(ma50) > 0 and len(ma200) > 0 and not pd.isna(ma200.iloc[-1]):
        if ma50.iloc[-1] > ma200.iloc[-1]:
            ma_alignment.append("MA50>MA200")
        else:
            ma_alignment.append("MA50<MA200")

    # 均线多头
    ma_bullish = len(ma5) > 0 and ma5.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1] if (
        not any(pd.isna([ma5.iloc[-1], ma20.iloc[-1], ma50.iloc[-1]]))
    ) else False

    # HH/HL判断（用最近20根）
    hh_hl_lookback = min(20, len(closes))
    recent_highs_idx = highs.iloc[-hh_hl_lookback:].idxmax()
    recent_lows_idx = lows.iloc[-hh_hl_lookback:].idxmin()
    prev_high_idx = highs.iloc[-hh_hl_lookback:-1].idxmax() if len(highs) > hh_hl_lookback else None
    prev_low_idx = lows.iloc[-hh_hl_lookback:-1].idxmin() if len(lows) > hh_hl_lookback else None

    hh_hl_check = "未确认"
    if prev_high_idx is not None and prev_low_idx is not None:
        if highs.iloc[-1] > highs.loc[prev_high_idx] and lows.iloc[-1] > lows.loc[prev_low_idx]:
            hh_hl_check = "更高高点+更高低点（上升趋势）"
        elif highs.iloc[-1] < highs.loc[prev_high_idx] and lows.iloc[-1] < lows.loc[prev_low_idx]:
            hh_hl_check = "更低高点+更低低点（下降趋势）"
        elif highs.iloc[-1] > highs.loc[prev_high_idx] and lows.iloc[-1] < lows.loc[prev_low_idx]:
            hh_hl_check = "更高高点+更低低点（震荡）"
        elif highs.iloc[-1] < highs.loc[prev_high_idx] and lows.iloc[-1] > lows.loc[prev_low_idx]:
            hh_hl_check = "更低高点+更高低点（震荡）"

    # 成交量分析（近5根 vs 前20根均值）
    vol_recent = volumes.iloc[-5:].mean()
    vol_prev = volumes.iloc[-20:-5].mean() if len(volumes) > 5 else vol_recent
    price_change = (closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5] * 100

    if vol_recent > vol_prev * 1.3:
        if price_change > 0:
            vol_profile = "放量上涨（资金参与）"
        else:
            vol_profile = "放量下跌（资金砸盘）"
    elif vol_recent < vol_prev * 0.7:
        vol_profile = "缩量整理（观望）"
    else:
        vol_profile = "量能正常"

    # 趋势判断
    if ma_bullish and "更高高点" in hh_hl_check:
        trend_type = "上升"
        confidence = 0.8
    elif not ma_bullish and "更低高点" in hh_hl_check:
        trend_type = "下降"
        confidence = 0.8
    elif ma_bullish or "更高低点" in hh_hl_check:
        trend_type = "上升"
        confidence = 0.5
    elif not ma_bullish or "更低高点" in hh_hl_check:
        trend_type = "下降"
        confidence = 0.5
    else:
        trend_type = "横盘"
        confidence = 0.4

    return MTFTrend(
        tf_label=tf_label,
        trend_type=trend_type,
        confidence=confidence,
        hh_hl_check=hh_hl_check,
        ma_bullish=ma_bullish,
        ma_alignment=ma_alignment,
        volume_profile=vol_profile,
        last_3_closes=last_3,
    )


# ============================================================
# 指标计算（整合）
# ============================================================

def _calc_indicators(df: pd.DataFrame, tf_label: str, price: float) -> MTFIndicator:
    """计算单一时间框架的技术指标"""
    closes = df["Close"].dropna()
    highs = df["High"].dropna()
    lows = df["Low"].dropna()

    def _v(series, idx=-1):
        val = series.iloc[idx] if len(series) > abs(idx) else None
        return float(val) if not pd.isna(val) else None

    ma5 = _calc_ma(closes, 5)
    ma20 = _calc_ma(closes, 20)
    ma50 = _calc_ma(closes, 50)
    ma200_s = _calc_ma(closes, 200)
    ema12 = _calc_ema(closes, 12)
    ema26 = _calc_ema(closes, 26)
    rsi = _calc_rsi(closes, 14)
    macd_line, signal_line, histogram = _calc_macd(closes)
    adx, atr_series = _calc_adx(highs, lows, closes)

    atr_val = _v(atr_series)
    atr_pct = (atr_val / price * 100) if atr_val and price else None

    # MA20斜率（近5期变化）
    ma20_slope = None
    if len(ma20) >= 5 and not pd.isna(ma20.iloc[-1]) and not pd.isna(ma20.iloc[-5]):
        ma20_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100

    # 价格vs均线位置
    price_vs_ma5 = "above" if _v(ma5) and price > _v(ma5) else "below"
    price_vs_ma20 = "above" if _v(ma20) and price > _v(ma20) else "below"
    price_vs_ma50 = "above" if _v(ma50) and price > _v(ma50) else "below"
    price_vs_ma200 = "above" if _v(ma200_s) and price > _v(ma200_s) else "below"

    return MTFIndicator(
        tf_label=tf_label,
        ma5=_v(ma5),
        ma20=_v(ma20),
        ma50=_v(ma50),
        ma200=_v(ma200_s),
        ema12=_v(ema12),
        ema26=_v(ema26),
        rsi=_v(rsi),
        macd_hist=_v(histogram),
        macd_signal=_v(signal_line),
        adx=_v(adx),
        atr=atr_val,
        atr_percent=atr_pct,
        price_vs_ma5=price_vs_ma5,
        price_vs_ma20=price_vs_ma20,
        price_vs_ma50=price_vs_ma50,
        price_vs_ma200=price_vs_ma200,
        ma20_slope_5d=round(ma20_slope, 3) if ma20_slope is not None else None,
    )


# ============================================================
# K线形态识别
# ============================================================

def _detect_patterns(df: pd.DataFrame, tf_label: str) -> list[MTFCandlePattern]:
    """检测K线形态（最近3根）"""
    patterns = []
    if len(df) < 5:
        return patterns

    closes = df["Close"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    def _body(i): return abs(closes[i] - opens[i])
    def _upper_shadow(i): return highs[i] - max(closes[i], opens[i])
    def _lower_shadow(i): return min(closes[i], opens[i]) - lows[i]
    def _range(i): return highs[i] - lows[i]
    def _is_bullish(i): return closes[i] > opens[i]
    def _is_bearish(i): return closes[i] < opens[i]

    i = n - 1  # 最新K线

    # 锤头/射击之星（单根）
    body = _body(i)
    upper = _upper_shadow(i)
    lower = _lower_shadow(i)
    rng = _range(i)
    is_small_body = body < rng * 0.3
    is_long_lower = lower > body * 2
    is_long_upper = upper > body * 2

    if is_small_body and lower > rng * 0.6 and upper < rng * 0.1:
        patterns.append(MTFCandlePattern(
            tf_label=tf_label,
            pattern_name="锤头（Hammer）",
            direction="bullish",
            strength="moderate",
            price_action="可能止跌反弹",
            bars_involved=[i],
        ))
    if is_small_body and upper > rng * 0.6 and lower < rng * 0.1:
        patterns.append(MTFCandlePattern(
            tf_label=tf_label,
            pattern_name="射击之星（Shooting Star）",
            direction="bearish",
            strength="moderate",
            price_action="可能滞涨回调",
            bars_involved=[i],
        ))

    # 十字星
    if body < rng * 0.1 and upper > 0 and lower > 0:
        patterns.append(MTFCandlePattern(
            tf_label=tf_label,
            pattern_name="十字星（Doji）",
            direction="neutral",
            strength="moderate",
            price_action="多空分歧，等待确认",
            bars_involved=[i],
        ))

    # 吞没形态（需2根）
    if n >= 2:
        i2 = n - 2
        body1 = _body(i2)
        body2 = _body(i)
        prev_bull = _is_bullish(i2)
        curr_bear = _is_bearish(i)

        # 看跌吞没
        if prev_bull and curr_bear:
            if closes[i] < opens[i2] and opens[i] < closes[i2]:
                patterns.append(MTFCandlePattern(
                    tf_label=tf_label,
                    pattern_name="看跌吞没（Bearish Engulfing）",
                    direction="bearish",
                    strength="strong",
                    price_action="空头反扑，趋势可能转弱",
                    bars_involved=[i2, i],
                ))

        # 看涨吞没
        prev_bear2 = _is_bearish(i2)
        curr_bull2 = _is_bullish(i)
        if prev_bear2 and curr_bull2:
            if closes[i] > opens[i2] and opens[i] > closes[i2]:
                patterns.append(MTFCandlePattern(
                    tf_label=tf_label,
                    pattern_name="看涨吞没（Bullish Engulfing）",
                    direction="bullish",
                    strength="strong",
                    price_action="多头反扑，趋势可能转强",
                    bars_involved=[i2, i],
                ))

    # 收敛形态（需5根）
    if n >= 5:
        recent_ranges = [_range(n - j - 1) for j in range(5)]
        avg_range = sum(recent_ranges) / 5
        if recent_ranges[0] < avg_range * 0.5:
            patterns.append(MTFCandlePattern(
                tf_label=tf_label,
                pattern_name="收缩整理（ Tightening Range）",
                direction="neutral",
                strength="moderate",
                price_action="突破在即，关注方向确认",
                bars_involved=[n - 5, n - 1],
            ))

    return patterns


# ============================================================
# 背离检测
# ============================================================

def _detect_divergences(df: pd.DataFrame) -> list[str]:
    """检测RSI和MACD背离"""
    divergences = []
    if len(df) < 30:
        return divergences

    closes = df["Close"].values
    rsi = _calc_rsi(df["Close"], 14).values
    macd_line, _, histogram = _calc_macd(df["Close"])
    macd_h = histogram.values

    # RSI背离（最近20根找高低点）
    lookback = min(20, len(closes) - 1)
    recent_close = closes[-lookback:]
    recent_rsi = rsi[-lookback:]

    # 找高点
    for i in range(5, len(recent_close) - 1):
        if (recent_close[i] > recent_close[i-1] and recent_close[i] > recent_close[i+1] and
            recent_close[i] > recent_close[i-2] and recent_close[i] > recent_close[i+2]):
            # 价格创新高但RSI未创新高 → 顶背离
            if recent_close[-1] > recent_close[i] and recent_rsi[-1] < recent_rsi[i]:
                divergences.append(f"RSI顶背离（价格创{lookback}日新高但RSI未跟随，日线）")
                break

    # 找低点
    for i in range(5, len(recent_close) - 1):
        if (recent_close[i] < recent_close[i-1] and recent_close[i] < recent_close[i+1] and
            recent_close[i] < recent_close[i-2] and recent_close[i] < recent_close[i+2]):
            # 价格创新低但RSI未创新低 → 底背离
            if recent_close[-1] < recent_close[i] and recent_rsi[-1] > recent_rsi[i]:
                divergences.append(f"RSI底背离（价格创{lookback}日新低但RSI未跟随，日线）")
                break

    # MACD背离
    recent_macdh = macd_h[-lookback:]
    for i in range(5, len(recent_macdh) - 1):
        if (recent_macdh[i] > recent_macdh[i-1] and recent_macdh[i] > recent_macdh[i+1] and
            recent_macdh[i] > recent_macdh[i-2] and recent_macdh[i] > recent_macdh[i+2]):
            if recent_close[-1] < recent_close[i] and recent_macdh[-1] > recent_macdh[i]:
                divergences.append("MACD顶背离（价格新低但MACD柱未跟随，日线）")
                break

    for i in range(5, len(recent_macdh) - 1):
        if (recent_macdh[i] < recent_macdh[i-1] and recent_macdh[i] < recent_macdh[i+1] and
            recent_macdh[i] < recent_macdh[i-2] and recent_macdh[i] < recent_macdh[i+2]):
            if recent_close[-1] > recent_close[i] and recent_macdh[-1] < recent_macdh[i]:
                divergences.append("MACD底背离（价格新高但MACD柱未跟随，日线）")
                break

    return divergences


# ============================================================
# 主分析函数
# ============================================================

def analyze_multi_timeframe(
    ticker: str,
    latest_price: float,
    fetcher,  # YFinanceFetcher 实例
    current_time_str: str = "",
) -> MTFAnalysisResult:
    """
    对标的多时间框架分析。

    参数:
        ticker: 标的代码（如 "SPY"）
        latest_price: 最新价格（从实时源获取）
        fetcher: YFinanceFetcher 实例
        current_time_str: 当前分析时间字符串

    数据获取策略:
        - 月线: period="2y", interval="1mo"  → 约24根K线
        - 周线: period="2y", interval="1wk"  → 约104根K线
        - 日线: period="1y", interval="1d"   → 约252根K线
        - 4H线: period="60d", interval="60m" → yfinance限制最多60天，取最近60个4H棒

    所有数据来自 yfinance（免费真实数据），4H最多60天有数据
    """
    from datetime import datetime

    result = MTFAnalysisResult(
        ticker=ticker,
        latest_price=latest_price,
        analysis_time=current_time_str or datetime.now().strftime("%Y-%m-%d %H:%M"),
        consensus_trend="数据不足",
        consensus_confidence=0.0,
    )

    # ---- 月线 ----
    df_monthly = fetcher.download_history(ticker, period="2y", interval="1mo")
    if not df_monthly.empty:
        result.timeframes["月线"] = _judge_trend(df_monthly, "月线")
        result.indicators["月线"] = _calc_indicators(df_monthly, "月线", latest_price)
        result.patterns.extend(_detect_patterns(df_monthly, "月线"))

    # ---- 周线 ----
    df_weekly = fetcher.download_history(ticker, period="2y", interval="1wk")
    if not df_weekly.empty:
        result.timeframes["周线"] = _judge_trend(df_weekly, "周线")
        result.indicators["周线"] = _calc_indicators(df_weekly, "周线", latest_price)
        result.patterns.extend(_detect_patterns(df_weekly, "周线"))

    # ---- 日线 ----
    df_daily = fetcher.download_history(ticker, period="1y", interval="1d")
    if not df_daily.empty:
        result.timeframes["日线"] = _judge_trend(df_daily, "日线")
        result.indicators["日线"] = _calc_indicators(df_daily, "日线", latest_price)
        result.patterns.extend(_detect_patterns(df_daily, "日线"))
        # 背离只在日线级别检测（4H/周线数据点太少）
        result.divergences = _detect_divergences(df_daily)

    # ---- 4H线 ----
    # yfinance 4H (60m) 最多只支持60天数据
    df_h4 = fetcher.download_history(ticker, period="60d", interval="60m")
    if not df_h4.empty:
        result.timeframes["4H"] = _judge_trend(df_h4, "4H")
        result.indicators["4H"] = _calc_indicators(df_h4, "4H", latest_price)
        result.patterns.extend(_detect_patterns(df_h4, "4H"))

    # ---- 综合结论 ----
    bullish_signals = []
    bearish_signals = []
    consensus_bull = 0
    consensus_bear = 0
    consensus_neutral = 0

    for tf, trend in result.timeframes.items():
        if trend.trend_type == "上升":
            consensus_bull += trend.confidence
            if "更高高点" in trend.hh_hl_check:
                bullish_signals.append(f"{tf}: {trend.hh_hl_check}")
            if trend.ma_bullish:
                bullish_signals.append(f"{tf}: 均线多头排列")
        elif trend.trend_type == "下降":
            consensus_bear += trend.confidence
            if "更低高点" in trend.hh_hl_check:
                bearish_signals.append(f"{tf}: {trend.hh_hl_check}")
        else:
            consensus_neutral += 1

    total = consensus_bull + consensus_bear + consensus_neutral
    if total == 0:
        result.consensus_trend = "数据不足"
        result.consensus_confidence = 0
    elif consensus_bull > consensus_bear and consensus_bull > consensus_neutral:
        result.consensus_trend = "上涨"
        result.consensus_confidence = round(consensus_bull / total, 2)
    elif consensus_bear > consensus_bull and consensus_bear > consensus_neutral:
        result.consensus_trend = "下跌"
        result.consensus_confidence = round(consensus_bear / total, 2)
    else:
        result.consensus_trend = "震荡"
        result.consensus_confidence = round(consensus_neutral / total, 2)

    result.bullish_signals = bullish_signals
    result.bearish_signals = bearish_signals

    return result


# ============================================================
# 格式化输出（供WebUI使用）
# ============================================================

def format_mtf_result(result: MTFAnalysisResult) -> dict:
    """将分析结果格式化为可展示的字典"""
    tf_data = {}
    for tf_label, trend in result.timeframes.items():
        ind = result.indicators.get(tf_label)
        tf_data[tf_label] = {
            "趋势": trend.trend_type,
            "置信度": f"{trend.confidence:.0%}",
            "HH/HL": trend.hh_hl_check,
            "均线多头": "是" if trend.ma_bullish else "否",
            "均线排列": ", ".join(trend.ma_alignment) if trend.ma_alignment else "数据不足",
            "量能": trend.volume_profile,
            "MA5": f"{ind.ma5:.2f}" if ind and ind.ma5 else "N/A",
            "MA20": f"{ind.ma20:.2f}" if ind and ind.ma20 else "N/A",
            "MA50": f"{ind.ma50:.2f}" if ind and ind.ma50 else "N/A",
            "MA200": f"{ind.ma200:.2f}" if ind and ind.ma200 else "N/A",
            "RSI(14)": f"{ind.rsi:.1f}" if ind and ind.rsi else "N/A",
            "MACD柱": f"{ind.macd_hist:.4f}" if ind and ind.macd_hist else "N/A",
            "ADX": f"{ind.adx:.1f}" if ind and ind.adx else "N/A",
            "ATR%": f"{ind.atr_percent:.2f}%" if ind and ind.atr_percent else "N/A",
            "价格>MA20": "✓" if ind and ind.price_vs_ma20 == "above" else "✗",
            "价格>MA50": "✓" if ind and ind.price_vs_ma50 == "above" else "✗",
        }

    pattern_data = []
    for p in result.patterns:
        pattern_data.append({
            "时间框架": p.tf_label,
            "形态": p.pattern_name,
            "方向": "🐂看多" if p.direction == "bullish" else ("🐻看空" if p.direction == "bearish" else "⚖️中性"),
            "强度": p.strength,
            "价格含义": p.price_action,
        })

    return {
        "标的": result.ticker,
        "最新价格": result.latest_price,
        "分析时间": result.analysis_time,
        "各时间框架": tf_data,
        "K线形态": pattern_data,
        "背离信号": result.divergences,
        "综合结论": {
            "趋势": result.consensus_trend,
            "置信度": f"{result.consensus_confidence:.0%}",
            "看多信号": result.bullish_signals,
            "看空信号": result.bearish_signals,
        },
    }
