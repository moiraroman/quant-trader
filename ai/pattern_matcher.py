# ============================================================
# ai/pattern_matcher.py — 历史模式匹配（DTW + 相似度搜索）
# 功能：找到历史最相似的K线序列，统计后续表现回测胜率
# 数据原则：真实历史数据，绝不估算
# ============================================================
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """单个历史匹配结果"""
    match_date: str          # 匹配到的历史日期
    similarity: float        # 相似度分数 (0~1)
    lookback_days: int       # 回溯天数
    # 匹配点之后的表现
    fwd_5d_return: float     # 5日后收益率
    fwd_20d_return: float    # 20日后收益率
    fwd_5d_max_dd: float     # 5日最大回撤
    fwd_20d_max_dd: float    # 20日最大回撤
    # 匹配点时的市场环境
    vix_level: Optional[float] = None
    trend_20d: str = ""      # "up"/"down"/"sideways"


@dataclass
class PatternMatchResult:
    """模式匹配分析结果"""
    ticker: str
    analysis_time: str
    lookback_days: int       # 当前分析使用的回溯天数
    current_regime: str      # 当前市场环境

    # 匹配结果
    matches: list = field(default_factory=list)  # PatternMatch列表
    top_matches: list = field(default_factory=list)

    # 统计回测
    win_rate_5d: float = 0.0   # 5日胜率（正收益比例）
    win_rate_20d: float = 0.0  # 20日胜率
    avg_return_5d: float = 0.0
    avg_return_20d: float = 0.0
    avg_max_dd_5d: float = 0.0
    avg_max_dd_20d: float = 0.0
    median_return_5d: float = 0.0
    median_return_20d: float = 0.0

    # 分布
    bullish_scenarios: int = 0   # 后续上涨次数
    bearish_scenarios: int = 0   # 后续下跌次数
    neutral_scenarios: int = 0   # 后续横盘次数（±2%）

    # 最相似场景描述
    best_match_date: str = ""
    best_match_similarity: float = 0.0

    # 免责声明
    disclaimer: str = ""

    missing_data: list = field(default_factory=list)


# ============================================================
# 相似度计算
# ============================================================

def _normalize_series(series: np.ndarray) -> np.ndarray:
    """Z-score标准化序列"""
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return series - mean
    return (series - mean) / std


def _dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    动态时间规整(DTW)距离。
    允许序列在时间轴上弹性对齐，适合形态匹配。
    """
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m]


def _correlation_similarity(s1: np.ndarray, s2: np.ndarray) -> float:
    """皮尔逊相关系数作为相似度 (-1~1)"""
    if len(s1) != len(s2) or len(s1) < 2:
        return 0.0
    s1n = _normalize_series(s1)
    s2n = _normalize_series(s2)
    corr = np.corrcoef(s1n, s2n)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _composite_similarity(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    综合相似度：相关系数 + 归一化DTW距离。
    返回 0~1 分数，1=完全相同。
    """
    corr = _correlation_similarity(s1, s2)
    if corr < 0.3:
        # 低相关直接淘汰
        return 0.0

    # DTW距离（需要相同长度）
    dtw = _dtw_distance(s1, s2)
    # 归一化：假设最大合理DTW距离为序列长度×2×标准差
    max_dtw = len(s1) * 2.0 * max(np.std(s1), 0.001)
    dtw_score = max(0, 1 - dtw / max_dtw) if max_dtw > 0 else 0.0

    # 加权综合
    return 0.6 * max(0, corr) + 0.4 * dtw_score


# ============================================================
# 核心匹配逻辑
# ============================================================

def _find_similar_patterns(
    df: pd.DataFrame,
    lookback: int = 20,
    top_n: int = 10,
    min_history: int = 252 * 3,  # 至少3年数据
) -> list[PatternMatch]:
    """
    在历史数据中找到与最近N天最相似的K线序列。

    匹配特征：
      - 收盘价标准化序列的相关系数
      - 成交量趋势（放量/缩量）
      - 波动率环境
    """
    if len(df) < min_history + lookback:
        logger.warning(f"[PatternMatcher] 历史数据不足: {len(df)} < {min_history + lookback}")
        return []

    # 当前序列（最近lookback天）
    current_close = df["Close"].iloc[-lookback:].values
    current_vol = df["Volume"].iloc[-lookback:].values if "Volume" in df.columns else None

    current_close_norm = _normalize_series(current_close)
    current_vol_norm = _normalize_series(current_vol) if current_vol is not None else None

    # 历史滑动窗口搜索
    matches = []
    # 跳过最近30天（避免自相关）
    max_idx = len(df) - lookback - 30
    min_idx = lookback

    for i in range(min_idx, max_idx):
        hist_close = df["Close"].iloc[i - lookback:i].values
        hist_fwd_close = df["Close"].iloc[i:i + 20].values if i + 20 <= len(df) else None

        if hist_fwd_close is None or len(hist_fwd_close) < 5:
            continue

        # 相似度计算
        sim = _composite_similarity(current_close_norm, _normalize_series(hist_close))
        if sim < 0.5:
            continue

        # 成交量相似度（如有）
        if current_vol_norm is not None and "Volume" in df.columns:
            hist_vol = df["Volume"].iloc[i - lookback:i].values
            vol_sim = _correlation_similarity(current_vol_norm, _normalize_series(hist_vol))
            sim = 0.7 * sim + 0.3 * max(0, vol_sim)

        if sim < 0.45:
            continue

        # 计算后续表现
        entry_price = hist_close[-1]
        fwd_5 = hist_fwd_close[min(4, len(hist_fwd_close) - 1)]
        fwd_20 = hist_fwd_close[-1]

        fwd_5_ret = (fwd_5 - entry_price) / entry_price * 100
        fwd_20_ret = (fwd_20 - entry_price) / entry_price * 100

        # 最大回撤
        fwd_prices = hist_fwd_close[:5]
        cummax = np.maximum.accumulate(fwd_prices)
        dd_5 = np.min((fwd_prices - cummax) / cummax) * 100 if len(fwd_prices) > 1 else 0.0

        fwd_prices_20 = hist_fwd_close
        cummax_20 = np.maximum.accumulate(fwd_prices_20)
        dd_20 = np.min((fwd_prices_20 - cummax_20) / cummax_20) * 100 if len(fwd_prices_20) > 1 else 0.0

        # 趋势环境
        ma20 = df["Close"].iloc[i - 20:i].mean() if i >= 20 else entry_price
        trend = "up" if entry_price > ma20 * 1.02 else "down" if entry_price < ma20 * 0.98 else "sideways"

        # VIX（如有）
        vix = None
        if "VIX" in df.columns:
            vix = float(df["VIX"].iloc[i])

        match = PatternMatch(
            match_date=str(df.index[i]),
            similarity=round(sim, 3),
            lookback_days=lookback,
            fwd_5d_return=round(fwd_5_ret, 2),
            fwd_20d_return=round(fwd_20_ret, 2),
            fwd_5d_max_dd=round(dd_5, 2),
            fwd_20d_max_dd=round(dd_20, 2),
            vix_level=vix,
            trend_20d=trend,
        )
        matches.append(match)

    # 按相似度排序取前N
    matches.sort(key=lambda x: x.similarity, reverse=True)
    return matches[:top_n]


# ============================================================
# 主分析入口
# ============================================================

def analyze_pattern_match(
    ticker: str,
    latest_price: float,
    fetcher,
    lookback_days: int = 20,
    top_n: int = 10,
) -> PatternMatchResult:
    """
    历史模式匹配分析。

    参数:
        lookback_days: 回溯天数（默认20日）
        top_n: 返回最相似的历史场景数

    数据源:
        - yfinance历史日线（免费，至少3年）
        - 缺失：VIX历史数据（需单独获取）
    """
    from datetime import datetime

    result = PatternMatchResult(
        ticker=ticker,
        analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        lookback_days=lookback_days,
        current_regime="unknown",
    )

    # 获取历史数据（5年）
    try:
        df = fetcher.download_history(ticker, period="5y", interval="1d")
        if df is None or df.empty or len(df) < 252 * 3:
            result.missing_data.append(f"历史数据不足（仅{len(df) if df is not None else 0}天，需≥756天）")
            return result
    except Exception as e:
        logger.warning(f"[PatternMatcher] 获取历史数据失败: {e}")
        result.missing_data.append("历史数据获取失败")
        return result

    # 判断当前市场环境
    current_ma20 = df["Close"].iloc[-20:].mean() if len(df) >= 20 else latest_price
    if latest_price > current_ma20 * 1.03:
        result.current_regime = "上升趋势"
    elif latest_price < current_ma20 * 0.97:
        result.current_regime = "下降趋势"
    else:
        result.current_regime = "区间震荡"

    # 执行匹配
    matches = _find_similar_patterns(df, lookback=lookback_days, top_n=top_n)
    result.matches = matches
    result.top_matches = matches[:5]

    if not matches:
        result.missing_data.append("未找到足够相似的历史模式（相似度<0.45）")
        return result

    # 统计回测
    result.best_match_date = matches[0].match_date
    result.best_match_similarity = matches[0].similarity

    # 胜率统计
    wins_5d = sum(1 for m in matches if m.fwd_5d_return > 0)
    wins_20d = sum(1 for m in matches if m.fwd_20d_return > 0)
    result.win_rate_5d = round(wins_5d / len(matches) * 100, 1)
    result.win_rate_20d = round(wins_20d / len(matches) * 100, 1)

    # 平均收益
    result.avg_return_5d = round(np.mean([m.fwd_5d_return for m in matches]), 2)
    result.avg_return_20d = round(np.mean([m.fwd_20d_return for m in matches]), 2)
    result.avg_max_dd_5d = round(np.mean([m.fwd_5d_max_dd for m in matches]), 2)
    result.avg_max_dd_20d = round(np.mean([m.fwd_20d_max_dd for m in matches]), 2)
    result.median_return_5d = round(np.median([m.fwd_5d_return for m in matches]), 2)
    result.median_return_20d = round(np.median([m.fwd_20d_return for m in matches]), 2)

    # 场景分布
    result.bullish_scenarios = sum(1 for m in matches if m.fwd_20d_return > 2)
    result.bearish_scenarios = sum(1 for m in matches if m.fwd_20d_return < -2)
    result.neutral_scenarios = len(matches) - result.bullish_scenarios - result.bearish_scenarios

    # 免责声明
    result.disclaimer = (
        "历史模式匹配仅反映过去相似形态的表现，不构成未来收益保证。"
        "市场结构可能变化，过往模式可能失效。"
    )

    return result


# ============================================================
# 格式化输出
# ============================================================

def format_pattern_match_result(result: PatternMatchResult) -> dict:
    """格式化模式匹配结果为字典（供WebUI）"""
    return {
        "标的": result.ticker,
        "当前环境": result.current_regime,
        "回溯天数": result.lookback_days,
        "最相似历史": {
            "日期": result.best_match_date,
            "相似度": f"{result.best_match_similarity:.1%}",
        },
        "回测统计": {
            "样本数": len(result.matches),
            "5日胜率": f"{result.win_rate_5d:.0f}%",
            "20日胜率": f"{result.win_rate_20d:.0f}%",
            "5日平均收益": f"{result.avg_return_5d:+.2f}%",
            "20日平均收益": f"{result.avg_return_20d:+.2f}%",
            "5日平均最大回撤": f"{result.avg_max_dd_5d:.2f}%",
            "20日平均最大回撤": f"{result.avg_max_dd_20d:.2f}%",
            "5日中位数收益": f"{result.median_return_5d:+.2f}%",
            "20日中位数收益": f"{result.median_return_20d:+.2f}%",
        },
        "场景分布": {
            "上涨(>2%)": result.bullish_scenarios,
            "下跌(<-2%)": result.bearish_scenarios,
            "横盘(±2%)": result.neutral_scenarios,
        },
        "历史匹配TOP3": [
            {
                "日期": m.match_date,
                "相似度": f"{m.similarity:.1%}",
                "5日后收益": f"{m.fwd_5d_return:+.2f}%",
                "20日后收益": f"{m.fwd_20d_return:+.2f}%",
                "当时趋势": m.trend_20d,
            }
            for m in result.top_matches[:3]
        ],
        "免责声明": result.disclaimer,
        "缺少数据": result.missing_data,
    }
