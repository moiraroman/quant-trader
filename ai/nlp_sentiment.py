# ============================================================
# ai/nlp_sentiment.py — NLP情绪分析
# 基于新闻标题/社交媒体的情绪提取（使用简单关键词+规则引擎）
# 无需外部NLP模型，纯规则驱动
# ============================================================
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 情绪词典
# ============================================================

BULLISH_KEYWORDS = {
    "surge", "rally", "boom", "soar", "jump", "gain", "rise", "bull", "bullish",
    "breakout", "momentum", "strong", "growth", "upgrade", "outperform", "beat",
    "record", "high", "rallying", "surging", "soaring", "gaining", "rising",
    "反弹", "上涨", "突破", "强劲", "增长", "利好", "超预期", "创新高",
    "牛市", "看多", "买入", "推荐", "上调", "盈利", "增长", "扩张",
}

BEARISH_KEYWORDS = {
    "crash", "plunge", "tumble", "drop", "fall", "decline", "bear", "bearish",
    "sell-off", "correction", "recession", "loss", "miss", "downgrade", "underperform",
    "low", "crashing", "plunging", "tumbling", "dropping", "falling", "declining",
    "下跌", "暴跌", "崩盘", "抛售", "调整", "衰退", "亏损", "不及预期",
    "熊市", "看空", "卖出", "下调", "裁员", "萎缩", "风险", "警告",
}

NEUTRAL_KEYWORDS = {
    "flat", "steady", "stable", "unchanged", "hold", "neutral", "mixed",
    "持平", "稳定", "不变", "观望", "中性", "震荡",
}

# 权重放大词
AMPLIFIERS = {
    "very", "extremely", "highly", "significantly", "sharply", "massively",
    "大幅", "剧烈", "严重", "极度", "非常",
}

# 否定词
NEGATORS = {
    "not", "no", "never", "without", "lack", "fail", "unable",
    "不", "没", "无", "未", "缺乏", "失败",
}

# 实体映射（标的相关）
TICKER_ALIASES = {
    "SPY": {"spy", "s&p 500", "sp500", "标普", "标普500"},
    "GLD": {"gld", "gold", "gold etf", "黄金", "金价"},
    "QQQ": {"qqq", "nasdaq", "纳指", "纳斯达克"},
    "IWM": {"iwm", "russell", "罗素", "小盘股"},
    "TLT": {"tlt", "treasury", "bond", "国债", "美债"},
    "VIX": {"vix", "volatility", "恐慌指数", "波动率"},
    "DXY": {"dxy", "dollar", "usd", "美元", "美元指数"},
}


@dataclass
class NewsItem:
    """单条新闻/帖子"""
    source: str
    title: str
    timestamp: str
    url: str = ""
    relevance_score: float = 0.0  # 与标的的相关度


@dataclass
class SentimentScore:
    """情绪评分"""
    bullish_score: float = 0.0
    bearish_score: float = 0.0
    neutral_score: float = 0.0
    composite_score: float = 50.0  # 0-100，50=中性
    confidence: float = 0.0
    keyword_matches: list = field(default_factory=list)


@dataclass
class NLPSentimentResult:
    """NLP情绪分析总结果"""
    ticker: str
    analysis_time: str = ""
    # 新闻源分析
    news_items_analyzed: int = 0
    # 情绪分布
    overall_sentiment: str = "中性"
    overall_score: float = 50.0  # 0-100
    sentiment_trend: str = "稳定"  # 上升/下降/稳定
    # 详细分数
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    # 关键词云
    top_bullish_keywords: list = field(default_factory=list)
    top_bearish_keywords: list = field(default_factory=list)
    # 时间序列（如果有）
    sentiment_history: list = field(default_factory=list)
    # 风险提示
    sentiment_extreme: bool = False  # 是否极端情绪（可作为反向指标）
    # 缺失数据
    missing_data: list = field(default_factory=list)
    disclaimer: str = "NLP情绪分析基于关键词匹配规则引擎，非深度学习模型，结果仅供参考"


def _calculate_sentiment(text: str, ticker: str) -> SentimentScore:
    """
    对单条文本进行情绪评分。
    基于关键词匹配 + 权重规则。
    """
    score = SentimentScore()
    text_lower = text.lower()

    # 检查相关性
    aliases = TICKER_ALIASES.get(ticker, {ticker.lower()})
    is_relevant = any(alias in text_lower for alias in aliases)
    if not is_relevant:
        score.neutral_score = 1.0
        return score

    # 分词（简单空格分割）
    words = re.findall(r'\b\w+\b', text_lower)

    bull_count = 0
    bear_count = 0
    neut_count = 0
    matched_keywords = []

    i = 0
    while i < len(words):
        word = words[i]
        weight = 1.0

        # 检查放大词
        if i > 0 and words[i - 1] in AMPLIFIERS:
            weight = 2.0
        # 检查否定词
        if i > 0 and words[i - 1] in NEGATORS:
            weight = -1.0  # 反转

        if word in BULLISH_KEYWORDS or any(bw in text_lower for bw in BULLISH_KEYWORDS if len(bw) > 4 and bw in text_lower):
            bull_count += weight
            matched_keywords.append((word, "bullish", weight))
        elif word in BEARISH_KEYWORDS:
            bear_count += weight
            matched_keywords.append((word, "bearish", weight))
        elif word in NEUTRAL_KEYWORDS:
            neut_count += weight

        i += 1

    # 长文本中的隐含情绪（句子级别）
    sentences = re.split(r'[.!?。！？]', text_lower)
    for sent in sentences:
        if any(b in sent for b in ["rise", "gain", "up", "high", "增长", "上涨"]):
            bull_count += 0.3
        if any(b in sent for b in ["fall", "drop", "down", "low", "下跌", "下降"]):
            bear_count += 0.3

    score.bullish_score = max(0, bull_count)
    score.bearish_score = max(0, bear_count)
    score.neutral_score = max(0, neut_count)
    score.keyword_matches = matched_keywords

    # 综合评分
    total = score.bullish_score + score.bearish_score + score.neutral_score
    if total > 0:
        score.composite_score = 50 + (score.bullish_score - score.bearish_score) / total * 50
        score.composite_score = max(0, min(100, score.composite_score))
        score.confidence = min(100, total * 10)
    else:
        score.composite_score = 50.0
        score.confidence = 0.0

    return score


def analyze_news_sentiment(
    ticker: str,
    news_items: list[dict],
) -> NLPSentimentResult:
    """
    分析新闻列表的情绪。

    参数:
        ticker: 标的代码
        news_items: 新闻列表，每项 {"source": str, "title": str, "timestamp": str}

    返回:
        NLPSentimentResult
    """
    result = NLPSentimentResult(ticker=ticker)
    result.analysis_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if not news_items:
        result.missing_data.append("无新闻数据")
        return result

    scores = []
    bull_keywords = {}
    bear_keywords = {}

    for item in news_items:
        text = item.get("title", "") + " " + item.get("summary", "")
        if not text.strip():
            continue

        score = _calculate_sentiment(text, ticker)
        if score.confidence > 0:
            scores.append(score)
            result.news_items_analyzed += 1

            for kw, direction, weight in score.keyword_matches:
                if direction == "bullish":
                    bull_keywords[kw] = bull_keywords.get(kw, 0) + weight
                else:
                    bear_keywords[kw] = bear_keywords.get(kw, 0) + weight

    if not scores:
        result.missing_data.append("无有效情绪评分")
        return result

    # 统计
    for s in scores:
        if s.composite_score > 60:
            result.bullish_count += 1
        elif s.composite_score < 40:
            result.bearish_count += 1
        else:
            result.neutral_count += 1

    # 综合评分
    avg_score = np.mean([s.composite_score for s in scores])
    result.overall_score = round(float(avg_score), 1)

    if result.overall_score >= 65:
        result.overall_sentiment = "偏多"
    elif result.overall_score <= 35:
        result.overall_sentiment = "偏空"
    else:
        result.overall_sentiment = "中性"

    # 极端情绪检测（反向指标）
    if result.overall_score >= 80 or result.overall_score <= 20:
        result.sentiment_extreme = True
        result.overall_sentiment += "（极端，可能反向）"

    # 趋势（如果有时间序列）
    if len(scores) >= 3:
        half = len(scores) // 2
        early_avg = np.mean([s.composite_score for s in scores[:half]])
        late_avg = np.mean([s.composite_score for s in scores[half:]])
        if late_avg > early_avg + 5:
            result.sentiment_trend = "上升"
        elif late_avg < early_avg - 5:
            result.sentiment_trend = "下降"
        else:
            result.sentiment_trend = "稳定"

    # 关键词排序
    result.top_bullish_keywords = sorted(bull_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
    result.top_bearish_keywords = sorted(bear_keywords.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info(f"[NLP] {ticker} 情绪分析完成: {result.overall_sentiment}({result.overall_score}), 分析{result.news_items_analyzed}条")
    return result


def format_nlp_result(result: NLPSentimentResult) -> dict:
    """格式化NLP结果为UI展示格式"""
    return {
        "标的": result.ticker,
        "分析时间": result.analysis_time,
        "分析新闻数": result.news_items_analyzed,
        "整体情绪": result.overall_sentiment,
        "情绪评分": result.overall_score,
        "情绪趋势": result.sentiment_trend,
        "情绪分布": {
            "偏多": result.bullish_count,
            "偏空": result.bearish_count,
            "中性": result.neutral_count,
        },
        "极端情绪警告": result.sentiment_extreme,
        "看多关键词": result.top_bullish_keywords,
        "看空关键词": result.top_bearish_keywords,
        "免责声明": result.disclaimer,
        "缺少数据": result.missing_data,
    }


def analyze_market_news_sentiment(
    tickers: list[str],
    fetcher,
    max_items_per_ticker: int = 20,
) -> dict:
    """
    批量分析多个标的的新闻情绪。
    新闻数据通过搜索获取（需外部传入或使用搜索skill）。
    """
    results = {}

    for ticker in tickers:
        try:
            # 尝试通过yfinance获取新闻
            import yfinance as yf
            stock = yf.Ticker(ticker)
            news = stock.news

            news_items = []
            for item in news[:max_items_per_ticker]:
                news_items.append({
                    "source": item.get("publisher", "Unknown"),
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "timestamp": item.get("published", ""),
                })

            if news_items:
                sentiment = analyze_news_sentiment(ticker, news_items)
                results[ticker] = format_nlp_result(sentiment)
            else:
                results[ticker] = {
                    "error": "无新闻数据",
                    "缺少数据": ["yfinance未返回新闻"],
                }
        except Exception as e:
            logger.warning(f"[NLP] {ticker} 新闻获取失败: {e}")
            results[ticker] = {
                "error": str(e),
                "缺少数据": [f"新闻获取失败: {e}"],
            }

    return results
