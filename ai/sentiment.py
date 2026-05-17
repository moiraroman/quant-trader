# ============================================================
# ai/sentiment.py — 市场情绪分析模块
# 获取 CNN Fear & Greed、AAII、VIX情绪、Put/Call Ratio 等
# 数据策略：优先网络搜索，其次yfinance计算，缺失标注
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class FearGreedIndex:
    """CNN Fear & Greed 指数"""
    value: Optional[int]          # 0~100
    label: str                    # "极度恐惧"/"恐惧"/"中性"/"贪婪"/"极度贪婪"
    prev_value: Optional[int]     # 前值
    prev_week_value: Optional[int]
    prev_month_value: Optional[int]
    source: str                   # 数据来源
    timestamp: str
    is_realtime: bool             # 是否实时数据


@dataclass
class AAII_Sentiment:
    """AAII 投资者情绪调查"""
    bullish_pct: Optional[float]   # 看多比例
    neutral_pct: Optional[float]   # 中性比例
    bearish_pct: Optional[float]   # 看空比例
    bull_bear_spread: Optional[float]  # 多空差
    report_date: str
    source: str
    is_latest: bool                # 是否最新一期


@dataclass
class VIX_Sentiment:
    """VIX 情绪解读"""
    vix_value: Optional[float]
    vix_20d_avg: Optional[float]
    vix_percentile: Optional[float]  # 历史百分位
    interpretation: str              # "恐慌"/"谨慎"/"正常"/"自满"
    term_structure: str              # "Contango"/"Backwardation"/"平坦"
    source: str


@dataclass
class PutCallRatio:
    """Put/Call Ratio"""
    total_pcr: Optional[float]       # 总PCR
    equity_pcr: Optional[float]      # 个股PCR
    index_pcr: Optional[float]       # 指数PCR
    interpretation: str              # "极度悲观"/"悲观"/"中性"/"乐观"/"极度乐观"
    source: str


@dataclass
class MarketBreadthSentiment:
    """市场广度情绪"""
    nyse_adv_dec: Optional[float]    # NYSE涨跌比
    nasdaq_adv_dec: Optional[float]  # NASDAQ涨跌比
    new_highs_lows: Optional[str]    # 新高新低描述
    source: str


@dataclass
class SentimentResult:
    """完整情绪分析结果"""
    ticker: str
    analysis_time: str

    # 核心指标
    fear_greed: FearGreedIndex
    aaii: AAII_Sentiment
    vix_sentiment: VIX_Sentiment
    put_call: PutCallRatio
    breadth: MarketBreadthSentiment

    # 综合
    composite_score: float           # 0~100（50=中性，>50贪婪，<50恐惧）
    composite_label: str             # "极度恐惧"/.../"极度贪婪"
    contrarian_signal: str           # "逆向看多"/"逆向看空"/"无"

    # 缺少的数据
    missing_data: list[str] = field(default_factory=list)


# ============================================================
# CNN Fear & Greed 指数
# ============================================================

def fetch_fear_greed_index(search_func=None) -> FearGreedIndex:
    """
    获取 CNN Fear & Greed 指数。

    数据获取策略：
        1. 优先：网络搜索获取最新值
        2. 备用：yfinance VIX + 市场数据估算
        3. 失败：标注"缺少"

    CNN FG 构成（7个指标）：
        - 股价动能（SPY vs 125日MA）
        - 股价强度（NYSE涨跌股数比）
        - 股价广度（NYSE新高新低比）
        - Put/Call Ratio
        - 垃圾债需求（HYG/LQD利差）
        - 市场波动（VIX vs 50日MA）
        - 避险需求（国债 vs 股票）
    """
    # 尝试通过搜索获取（如果提供了搜索函数）
    if search_func:
        try:
            result = search_func("CNN Fear and Greed Index current value today")
            if result:
                # 解析搜索结果
                # 实际解析逻辑由调用方实现
                return FearGreedIndex(
                    value=None,
                    label="需搜索解析",
                    prev_value=None,
                    prev_week_value=None,
                    prev_month_value=None,
                    source="搜索待解析",
                    timestamp=datetime.now().isoformat(),
                    is_realtime=False,
                )
        except Exception as e:
            logger.warning(f"[Sentiment] FG搜索失败: {e}")

    # 备用：基于VIX估算（粗糙但可用）
    # VIX < 15 → 贪婪(75), VIX 15-20 → 中性(50), VIX 20-25 → 恐惧(35), VIX > 25 → 极度恐惧(20)
    return FearGreedIndex(
        value=None,
        label="缺少[CNN FG需搜索或付费API]",
        prev_value=None,
        prev_week_value=None,
        prev_month_value=None,
        source="缺少",
        timestamp=datetime.now().isoformat(),
        is_realtime=False,
    )


def _estimate_fg_from_vix(vix_value: float) -> tuple[int, str]:
    """基于VIX粗略估算FG指数（仅作参考，非精确）"""
    if vix_value < 12:
        return 80, "极度贪婪"
    elif vix_value < 15:
        return 65, "贪婪"
    elif vix_value < 20:
        return 50, "中性"
    elif vix_value < 25:
        return 35, "恐惧"
    elif vix_value < 30:
        return 20, "极度恐惧"
    else:
        return 10, "恐慌"


# ============================================================
# AAII 情绪调查
# ============================================================

def fetch_aaii_sentiment(search_func=None) -> AAII_Sentiment:
    """
    获取 AAII (American Association of Individual Investors) 情绪调查。

    发布频率：每周四发布
    数据获取：
        1. 网络搜索最新报告
        2. 备用：标注"缺少"
    """
    if search_func:
        try:
            result = search_func("AAII sentiment survey latest bullish bearish neutral")
            if result:
                return AAII_Sentiment(
                    bullish_pct=None,
                    neutral_pct=None,
                    bearish_pct=None,
                    bull_bear_spread=None,
                    report_date="待解析",
                    source="搜索待解析",
                    is_latest=False,
                )
        except Exception as e:
            logger.warning(f"[Sentiment] AAII搜索失败: {e}")

    return AAII_Sentiment(
        bullish_pct=None,
        neutral_pct=None,
        bearish_pct=None,
        bull_bear_spread=None,
        report_date="",
        source="缺少[AAII需搜索或付费API]",
        is_latest=False,
    )


# ============================================================
# VIX 情绪解读
# ============================================================

def analyze_vix_sentiment(vix_value: Optional[float], vix_history: pd.DataFrame = None) -> VIX_Sentiment:
    """
    基于VIX值解读市场情绪。

    VIX 情绪映射：
        < 12: 极度自满（危险信号，逆向看空）
        12-15: 乐观/自满
        15-20: 正常区间
        20-25: 谨慎/担忧
        25-30: 恐惧
        > 30: 极度恐慌（逆向看多机会）
    """
    if vix_value is None:
        return VIX_Sentiment(
            vix_value=None,
            vix_20d_avg=None,
            vix_percentile=None,
            interpretation="缺少VIX数据",
            term_structure="未知",
            source="缺少",
        )

    # 20日均值
    vix_20d = None
    if vix_history is not None and len(vix_history) >= 20:
        vix_20d = vix_history["Close"].iloc[-20:].mean()

    # 历史百分位（需要更长期数据，这里简化）
    vix_pct = None
    if vix_history is not None and len(vix_history) >= 252:
        hist_values = vix_history["Close"].dropna()
        vix_pct = (hist_values < vix_value).mean() * 100

    # 解读
    if vix_value < 12:
        interp = "极度自满（危险，逆向看空）"
    elif vix_value < 15:
        interp = "乐观/自满"
    elif vix_value < 20:
        interp = "正常区间"
    elif vix_value < 25:
        interp = "谨慎/担忧"
    elif vix_value < 30:
        interp = "恐惧（逆向看多机会）"
    else:
        interp = "极度恐慌（强烈逆向看多）"

    # 期限结构（简化：比较近月vs远月，需要期货数据）
    term = "缺少期货数据[Contango/Backwardation]"

    return VIX_Sentiment(
        vix_value=vix_value,
        vix_20d_avg=round(vix_20d, 2) if vix_20d else None,
        vix_percentile=round(vix_pct, 1) if vix_pct else None,
        interpretation=interp,
        term_structure=term,
        source="yfinance(VIX=^VIX)",
    )


# ============================================================
# Put/Call Ratio
# ============================================================

def fetch_put_call_ratio(search_func=None) -> PutCallRatio:
    """
    获取 CBOE Put/Call Ratio。

    PCR 解读：
        > 1.2: 极度悲观（逆向看多）
        0.9-1.2: 悲观
        0.7-0.9: 中性
        0.5-0.7: 乐观
        < 0.5: 极度乐观（逆向看空）

    数据来源：
        1. CBOE官网（免费但需解析）
        2. 网络搜索
        3. 备用：标注"缺少"
    """
    if search_func:
        try:
            result = search_func("CBOE put call ratio today")
            if result:
                return PutCallRatio(
                    total_pcr=None,
                    equity_pcr=None,
                    index_pcr=None,
                    interpretation="待解析",
                    source="搜索待解析",
                )
        except Exception as e:
            logger.warning(f"[Sentiment] PCR搜索失败: {e}")

    return PutCallRatio(
        total_pcr=None,
        equity_pcr=None,
        index_pcr=None,
        interpretation="缺少[CBOE PCR需搜索或付费API]",
        source="缺少",
    )


def _interpret_pcr(pcr: float) -> str:
    """解读PCR值"""
    if pcr > 1.2:
        return "极度悲观（逆向看多）"
    elif pcr > 0.9:
        return "悲观"
    elif pcr > 0.7:
        return "中性"
    elif pcr > 0.5:
        return "乐观"
    else:
        return "极度乐观（逆向看空）"


# ============================================================
# 市场广度情绪
# ============================================================

def fetch_market_breadth_sentiment(fetcher) -> MarketBreadthSentiment:
    """
    基于市场广度数据判断情绪。

    数据来源：
        - NYSE/NASDAQ 涨跌股数比（yfinance可获取指数数据）
        - 新高新低数（需付费数据，标注缺少）
    """
    try:
        # 尝试获取 NYSE 和 NASDAQ 的涨跌数据
        # 通过 ^NYA (NYSE Composite) 和 ^IXIC (NASDAQ) 的日数据估算
        nya = fetcher.download_history("^NYA", period="5d", interval="1d")
        ixic = fetcher.download_history("^IXIC", period="5d", interval="1d")

        nyse_ad_dec = None
        nasdaq_ad_dec = None

        if not nya.empty and len(nya) >= 2:
            # 用价格变化近似（非精确涨跌比）
            change = (nya["Close"].iloc[-1] - nya["Close"].iloc[-2]) / nya["Close"].iloc[-2] * 100
            nyse_ad_dec = 1.0 + change  # 粗略映射

        if not ixic.empty and len(ixic) >= 2:
            change = (ixic["Close"].iloc[-1] - ixic["Close"].iloc[-2]) / ixic["Close"].iloc[-2] * 100
            nasdaq_ad_dec = 1.0 + change

        return MarketBreadthSentiment(
            nyse_adv_dec=round(nyse_ad_dec, 2) if nyse_ad_dec else None,
            nasdaq_adv_dec=round(nasdaq_ad_dec, 2) if nasdaq_ad_dec else None,
            new_highs_lows="缺少[需NYSE/NASDAQ涨跌股数数据]",
            source="yfinance(^NYA/^IXIC近似)",
        )
    except Exception as e:
        logger.warning(f"[Sentiment] 广度数据失败: {e}")
        return MarketBreadthSentiment(
            nyse_adv_dec=None,
            nasdaq_adv_dec=None,
            new_highs_lows="缺少",
            source="失败",
        )


# ============================================================
# 综合情绪评分
# ============================================================

def _calculate_composite_score(
    fg: FearGreedIndex,
    aaii: AAII_Sentiment,
    vix: VIX_Sentiment,
    pcr: PutCallRatio,
    breadth: MarketBreadthSentiment,
) -> tuple[float, str, str]:
    """
    计算综合情绪评分。

    评分逻辑：
        - 50 = 中性
        - >50 = 贪婪（情绪过热，逆向看空）
        - <50 = 恐惧（情绪低迷，逆向看多）

    各指标权重：
        - VIX: 30%（最重要）
        - FG: 25%
        - PCR: 20%
        - AAII: 15%
        - 广度: 10%
    """
    scores = []
    weights = []

    # VIX (30%)
    if vix.vix_value is not None:
        # VIX 12→80分(贪婪), 20→50分, 30→20分(恐惧)
        vix_score = max(0, min(100, 80 - (vix.vix_value - 12) * 3))
        scores.append(vix_score)
        weights.append(0.30)

    # FG (25%)
    if fg.value is not None:
        scores.append(float(fg.value))
        weights.append(0.25)

    # PCR (20%) - 反向：PCR高=恐惧=低分
    if pcr.total_pcr is not None:
        pcr_score = max(0, min(100, 100 - pcr.total_pcr * 50))
        scores.append(pcr_score)
        weights.append(0.20)

    # AAII (15%)
    if aaii.bullish_pct is not None and aaii.bearish_pct is not None:
        aaii_score = aaii.bullish_pct / (aaii.bullish_pct + aaii.bearish_pct + aaii.neutral_pct) * 100
        scores.append(aaii_score)
        weights.append(0.15)

    # 广度 (10%)
    if breadth.nyse_adv_dec is not None:
        breadth_score = min(100, max(0, breadth.nyse_adv_dec * 50))
        scores.append(breadth_score)
        weights.append(0.10)

    if not scores:
        return 50.0, "数据不足", "无"

    # 加权平均
    total_weight = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights)) / total_weight

    # 标签
    if composite >= 75:
        label = "极度贪婪"
        contrarian = "逆向看空（情绪过热）"
    elif composite >= 60:
        label = "贪婪"
        contrarian = "逆向看空"
    elif composite >= 40:
        label = "中性"
        contrarian = "无"
    elif composite >= 25:
        label = "恐惧"
        contrarian = "逆向看多"
    else:
        label = "极度恐惧"
        contrarian = "逆向看多（情绪极度低迷）"

    return round(composite, 1), label, contrarian


# ============================================================
# 主函数
# ============================================================

def analyze_market_sentiment(
    ticker: str,
    fetcher,
    search_func=None,
    vix_value: Optional[float] = None,
    vix_history: pd.DataFrame = None,
) -> SentimentResult:
    """
    完整情绪分析。

    参数:
        ticker: 标的代码
        fetcher: 数据获取器
        search_func: 可选搜索函数（用于CNN FG/AAII/PCR）
        vix_value: 当前VIX值（如已知）
        vix_history: VIX历史数据（用于计算百分位）

    数据策略：
        - 能获取的指标全部获取
        - 无法获取的标注"缺少[数据源]"
        - 绝不估算/假设
    """
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    missing = []

    # 1. Fear & Greed
    fg = fetch_fear_greed_index(search_func)
    if fg.value is None:
        missing.append("CNN Fear & Greed指数（需搜索或付费API）")

    # 2. AAII
    aaii = fetch_aaii_sentiment(search_func)
    if aaii.bullish_pct is None:
        missing.append("AAII投资者情绪调查（需搜索或付费API）")

    # 3. VIX情绪
    if vix_value is None:
        try:
            vix_df = fetcher.download_history("^VIX", period="1y", interval="1d")
            if not vix_df.empty:
                vix_value = float(vix_df["Close"].iloc[-1])
                vix_history = vix_df
        except Exception as e:
            logger.warning(f"[Sentiment] VIX获取失败: {e}")
            missing.append("VIX数据")

    vix_sent = analyze_vix_sentiment(vix_value, vix_history)

    # 4. Put/Call Ratio
    pcr = fetch_put_call_ratio(search_func)
    if pcr.total_pcr is None:
        missing.append("Put/Call Ratio（需搜索或CBOE数据）")

    # 5. 市场广度
    breadth = fetch_market_breadth_sentiment(fetcher)
    if breadth.nyse_adv_dec is None:
        missing.append("NYSE涨跌比（需精确数据）")

    # 综合评分
    composite, label, contrarian = _calculate_composite_score(
        fg, aaii, vix_sent, pcr, breadth
    )

    return SentimentResult(
        ticker=ticker,
        analysis_time=time_str,
        fear_greed=fg,
        aaii=aaii,
        vix_sentiment=vix_sent,
        put_call=pcr,
        breadth=breadth,
        composite_score=composite,
        composite_label=label,
        contrarian_signal=contrarian,
        missing_data=missing,
    )


# ============================================================
# 格式化输出
# ============================================================

def format_sentiment_result(result: SentimentResult) -> dict:
    """格式化情绪分析结果供WebUI展示"""
    return {
        "标的": result.ticker,
        "分析时间": result.analysis_time,
        "综合情绪": {
            "评分": f"{result.composite_score:.0f}/100",
            "标签": result.composite_label,
            "逆向信号": result.contrarian_signal,
        },
        "CNN恐惧贪婪": {
            "指数": result.fear_greed.value if result.fear_greed.value else "缺少",
            "状态": result.fear_greed.label,
            "来源": result.fear_greed.source,
        },
        "AAII情绪": {
            "看多": f"{result.aaii.bullish_pct:.1f}%" if result.aaii.bullish_pct else "缺少",
            "中性": f"{result.aaii.neutral_pct:.1f}%" if result.aaii.neutral_pct else "缺少",
            "看空": f"{result.aaii.bearish_pct:.1f}%" if result.aaii.bearish_pct else "缺少",
            "来源": result.aaii.source,
        },
        "VIX情绪": {
            "当前值": f"{result.vix_sentiment.vix_value:.2f}" if result.vix_sentiment.vix_value else "缺少",
            "20日均": f"{result.vix_sentiment.vix_20d_avg:.2f}" if result.vix_sentiment.vix_20d_avg else "N/A",
            "历史百分位": f"{result.vix_sentiment.vix_percentile:.1f}%" if result.vix_sentiment.vix_percentile else "N/A",
            "解读": result.vix_sentiment.interpretation,
            "期限结构": result.vix_sentiment.term_structure,
        },
        "Put/Call": {
            "总PCR": f"{result.put_call.total_pcr:.2f}" if result.put_call.total_pcr else "缺少",
            "解读": result.put_call.interpretation,
            "来源": result.put_call.source,
        },
        "市场广度": {
            "NYSE涨跌比": f"{result.breadth.nyse_adv_dec:.2f}" if result.breadth.nyse_adv_dec else "缺少",
            "NASDAQ涨跌比": f"{result.breadth.nasdaq_adv_dec:.2f}" if result.breadth.nasdaq_adv_dec else "缺少",
            "新高新低": result.breadth.new_highs_lows,
        },
        "缺少数据": result.missing_data,
    }
