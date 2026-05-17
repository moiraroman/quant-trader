"""
sentiment_nlp.py — 市场情绪 NLP 分析模块

功能：
    - 基于文本的情感分析（新闻标题、社交媒体）
    - 关键词提取与主题建模
    - 情绪时间序列追踪
    - 与价格数据的交叉验证

数据源策略：
    - 优先：付费新闻 API（Bloomberg、Reuters、NewsAPI）
    - 免费：Reddit API、Twitter API v2（需开发者账号）
    - 回退：内置示例数据 + 标注"缺少实时数据"
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 数据类定义
# ============================================================

@dataclass
class TextSentiment:
    """单条文本的情感分析结果"""
    text: str
    sentiment_score: float          # -1.0 (极负) ~ +1.0 (极正)
    confidence: float               # 0.0 ~ 1.0
    keywords: List[str]
    category: str                   # 'news', 'social', 'earnings', 'analyst'
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    url: Optional[str] = None

@dataclass
class SentimentTimePoint:
    """时间点的情绪聚合"""
    timestamp: datetime
    avg_score: float
    volume: int                     # 文本数量
    pos_ratio: float                # 正面比例
    neg_ratio: float                # 负面比例
    neu_ratio: float                # 中性比例
    top_keywords: List[str]
    sentiment_trend: str            # 'improving', 'worsening', 'stable'

@dataclass
class SentimentNLPResult:
    """NLP 情绪分析完整结果"""
    ticker: str
    analysis_time: datetime
    
    # 整体情绪
    overall_sentiment: str          # 'bullish', 'bearish', 'neutral', 'mixed'
    overall_score: float            # -1.0 ~ +1.0
    confidence: str                 # 'high', 'medium', 'low'
    
    # 时间序列
    time_series: List[SentimentTimePoint] = field(default_factory=list)
    
    # 分类统计
    news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    earnings_sentiment: Optional[float] = None
    analyst_sentiment: Optional[float] = None
    
    # 关键词与主题
    top_keywords: List[str] = field(default_factory=list)
    emerging_themes: List[str] = field(default_factory=list)
    
    # 与价格的相关性
    price_correlation: Optional[float] = None
    sentiment_lead_hours: Optional[int] = None
    
    # 风险信号
    risk_signals: List[str] = field(default_factory=list)
    
    # 数据完整性
    data_completeness: float = 0.0  # 0.0 ~ 1.0
    missing_sources: List[str] = field(default_factory=list)
    
    # 原始文本样本
    sample_texts: List[TextSentiment] = field(default_factory=list)

# ============================================================
# 内置情感词典（简化版，实际应用建议使用更完整的词典）
# ============================================================

BULLISH_KEYWORDS = {
    'strong', 'growth', 'beat', 'outperform', 'upgrade', 'buy', 'bullish',
    'rally', 'surge', 'soar', 'jump', 'rocket', 'moon', 'breakout',
    'record', 'high', 'profit', 'revenue', 'expansion', 'partnership',
    'innovation', 'leader', 'dominant', 'recovery', 'momentum',
    'upgrade', 'target raised', 'price target', 'overweight',
    '推荐', '买入', '增持', '看好', '上涨', '突破', '强劲',
    '增长', '超预期', '盈利', '创新高', '反弹', '牛市',
}

BEARISH_KEYWORDS = {
    'weak', 'decline', 'miss', 'underperform', 'downgrade', 'sell', 'bearish',
    'crash', 'plunge', 'tumble', 'drop', 'fall', 'dump', 'collapse',
    'low', 'loss', 'debt', 'bankruptcy', 'layoff', 'recession',
    'investigation', 'lawsuit', 'fine', 'penalty', 'risk',
    'downgrade', 'target cut', 'price target lowered', 'underweight',
    '卖出', '减持', '看空', '下跌', '跌破', '疲软',
    '衰退', '亏损', '债务', '破产', '裁员', '熊市',
}

NEUTRAL_KEYWORDS = {
    'hold', 'neutral', 'maintain', 'steady', 'flat', 'stable',
    'in line', 'meets expectations', 'unchanged',
    '持有', '中性', '持平', '稳定',
}

RISK_KEYWORDS = {
    'investigation', 'lawsuit', 'recall', 'data breach', 'cyberattack',
    'fraud', 'accounting', 'restatement', 'sec', 'doj',
    '调查', '诉讼', '召回', '数据泄露', '网络攻击',
    '欺诈', '会计', '重述', '监管',
}

# ============================================================
# 核心分析函数
# ============================================================

def analyze_text_sentiment(text: str, category: str = "news") -> TextSentiment:
    """
    分析单条文本的情感倾向。
    
    使用简化规则引擎（实际应用建议使用 VADER、TextBlob 或 Transformer 模型）
    """
    if not text or not isinstance(text, str):
        return TextSentiment(
            text="",
            sentiment_score=0.0,
            confidence=0.0,
            keywords=[],
            category=category,
        )
    
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    # 计算情感得分
    bull_count = len(words & BULLISH_KEYWORDS)
    bear_count = len(words & BEARISH_KEYWORDS)
    neu_count = len(words & NEUTRAL_KEYWORDS)
    risk_count = len(words & RISK_KEYWORDS)
    
    total_sentiment_words = bull_count + bear_count + neu_count
    
    if total_sentiment_words == 0:
        sentiment_score = 0.0
        confidence = 0.3
    else:
        sentiment_score = (bull_count - bear_count) / total_sentiment_words
        # 风险词汇增加负面权重
        if risk_count > 0:
            sentiment_score -= risk_count * 0.1
            sentiment_score = max(-1.0, sentiment_score)
        confidence = min(0.9, 0.4 + total_sentiment_words * 0.1)
    
    # 提取关键词
    all_keywords = list(words & (BULLISH_KEYWORDS | BEARISH_KEYWORDS | RISK_KEYWORDS))
    
    return TextSentiment(
        text=text[:200] + "..." if len(text) > 200 else text,
        sentiment_score=round(sentiment_score, 3),
        confidence=round(confidence, 2),
        keywords=all_keywords[:10],
        category=category,
    )

def aggregate_sentiment(
    texts: List[TextSentiment],
    window: timedelta = timedelta(hours=24)
) -> List[SentimentTimePoint]:
    """
    按时间窗口聚合情绪数据。
    """
    if not texts:
        return []
    
    # 按时间排序
    sorted_texts = sorted(texts, key=lambda x: x.timestamp or datetime.min)
    
    # 创建时间窗口
    if sorted_texts[0].timestamp is None:
        return []
    
    start_time = sorted_texts[0].timestamp.replace(minute=0, second=0, microsecond=0)
    end_time = sorted_texts[-1].timestamp
    
    time_points = []
    current_time = start_time
    
    while current_time <= end_time:
        window_texts = [
            t for t in sorted_texts
            if t.timestamp and current_time <= t.timestamp < current_time + window
        ]
        
        if window_texts:
            scores = [t.sentiment_score for t in window_texts]
            pos_count = sum(1 for s in scores if s > 0.1)
            neg_count = sum(1 for s in scores if s < -0.1)
            neu_count = len(scores) - pos_count - neg_count
            
            # 计算趋势
            trend = "stable"
            if len(time_points) >= 1:
                prev_avg = time_points[-1].avg_score
                curr_avg = np.mean(scores)
                if curr_avg > prev_avg + 0.1:
                    trend = "improving"
                elif curr_avg < prev_avg - 0.1:
                    trend = "worsening"
            
            # 提取关键词
            all_keywords = []
            for t in window_texts:
                all_keywords.extend(t.keywords)
            top_keywords = pd.Series(all_keywords).value_counts().head(5).index.tolist()
            
            time_points.append(SentimentTimePoint(
                timestamp=current_time,
                avg_score=round(np.mean(scores), 3),
                volume=len(window_texts),
                pos_ratio=round(pos_count / len(scores), 2),
                neg_ratio=round(neg_count / len(scores), 2),
                neu_ratio=round(neu_count / len(scores), 2),
                top_keywords=top_keywords,
                sentiment_trend=trend,
            ))
        
        current_time += window
    
    return time_points

def calculate_price_correlation(
    sentiment_series: List[SentimentTimePoint],
    price_df: pd.DataFrame,
    max_lag_hours: int = 48
) -> Tuple[Optional[float], Optional[int]]:
    """
    计算情绪与价格的相关性，并检测情绪领先时间。
    
    Returns:
        (correlation, lead_hours)
    """
    if not sentiment_series or price_df is None or price_df.empty:
        return None, None
    
    try:
        # 创建情绪 DataFrame
        sentiment_df = pd.DataFrame([
            {
                'timestamp': p.timestamp,
                'sentiment': p.avg_score,
            }
            for p in sentiment_series
        ])
        sentiment_df.set_index('timestamp', inplace=True)
        sentiment_df = sentiment_df.resample('H').mean().fillna(method='ffill')
        
        # 准备价格数据
        price_hourly = price_df['Close'].resample('H').last().fillna(method='ffill')
        price_returns = price_hourly.pct_change().dropna()
        
        # 对齐数据
        common_index = sentiment_df.index.intersection(price_returns.index)
        if len(common_index) < 10:
            return None, None
        
        s_aligned = sentiment_df.loc[common_index, 'sentiment']
        p_aligned = price_returns.loc[common_index]
        
        # 计算不同滞后的相关性
        best_corr = 0
        best_lag = 0
        
        for lag in range(0, max_lag_hours + 1, 4):  # 每4小时检查一次
            if lag > 0:
                s_lagged = s_aligned.shift(lag).dropna()
                p_lagged = p_aligned.loc[s_lagged.index]
            else:
                s_lagged = s_aligned
                p_lagged = p_aligned
            
            if len(s_lagged) < 10:
                continue
            
            corr = np.corrcoef(s_lagged, p_lagged)[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        return round(best_corr, 3), best_lag if best_lag > 0 else None
        
    except Exception as e:
        logger.warning(f"[SentimentNLP] 价格相关性计算失败: {e}")
        return None, None

# ============================================================
# 主分析函数
# ============================================================

def analyze_sentiment_nlp(
    ticker: str,
    fetcher,
    news_texts: Optional[List[Dict]] = None,
    price_df: Optional[pd.DataFrame] = None,
    days_back: int = 7,
) -> SentimentNLPResult:
    """
    执行完整的 NLP 情绪分析。
    
    Args:
        ticker: 股票代码
        fetcher: 数据获取器
        news_texts: 外部提供的新闻文本列表（可选）
        price_df: 价格数据（用于相关性分析，可选）
        days_back: 回溯天数
    
    Returns:
        SentimentNLPResult
    """
    missing = []
    all_texts: List[TextSentiment] = []
    
    # 1. 处理外部提供的新闻文本
    if news_texts:
        for item in news_texts:
            text = item.get('text', item.get('title', ''))
            category = item.get('category', 'news')
            ts = item.get('timestamp')
            
            sentiment = analyze_text_sentiment(text, category)
            sentiment.timestamp = ts
            sentiment.source = item.get('source')
            sentiment.url = item.get('url')
            all_texts.append(sentiment)
    else:
        missing.append("实时新闻数据（需 NewsAPI/Bloomberg API）")
    
    # 2. 如果没有外部数据，生成示例数据（标注为模拟）
    if not all_texts:
        logger.info("[SentimentNLP] 无外部文本数据，生成示例数据用于演示")
        sample_headlines = [
            ("Apple reports strong Q4 earnings, beats expectations", "news", 0.8),
            ("Analysts raise price target on growth momentum", "analyst", 0.7),
            ("Supply chain concerns weigh on near-term outlook", "news", -0.3),
            ("New product launch drives investor optimism", "news", 0.6),
            ("Market volatility creates buying opportunity", "social", 0.4),
        ]
        
        for text, cat, score in sample_headlines:
            sentiment = analyze_text_sentiment(text, cat)
            sentiment.timestamp = datetime.now() - timedelta(hours=np.random.randint(0, 48))
            sentiment.sentiment_score = score  # 覆盖自动计算
            all_texts.append(sentiment)
        
        missing.append("实时数据（当前使用示例数据）")
    
    # 3. 按类别聚合
    category_scores = {}
    for cat in ['news', 'social', 'earnings', 'analyst']:
        cat_texts = [t for t in all_texts if t.category == cat]
        if cat_texts:
            category_scores[cat] = round(np.mean([t.sentiment_score for t in cat_texts]), 3)
    
    # 4. 时间序列聚合
    time_series = aggregate_sentiment(all_texts, window=timedelta(hours=6))
    
    # 5. 计算整体情绪
    if all_texts:
        overall_score = round(np.mean([t.sentiment_score for t in all_texts]), 3)
        
        if overall_score > 0.2:
            overall_sentiment = "bullish"
        elif overall_score < -0.2:
            overall_sentiment = "bearish"
        elif abs(overall_score) <= 0.1:
            overall_sentiment = "neutral"
        else:
            overall_sentiment = "mixed"
        
        # 置信度基于文本数量和一致性
        volume_confidence = min(1.0, len(all_texts) / 50)  # 50条为满分
        scores = [t.sentiment_score for t in all_texts]
        consistency = 1.0 - np.std(scores)  # 标准差越小越一致
        confidence_score = volume_confidence * 0.5 + consistency * 0.5
        
        if confidence_score > 0.7:
            confidence = "high"
        elif confidence_score > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        overall_score = 0.0
        overall_sentiment = "neutral"
        confidence = "low"
    
    # 6. 提取关键词
    all_keywords = []
    for t in all_texts:
        all_keywords.extend(t.keywords)
    top_keywords = pd.Series(all_keywords).value_counts().head(10).index.tolist()
    
    # 7. 检测新兴主题（简单实现：最近24小时新出现的关键词）
    emerging_themes = []
    if time_series:
        recent_keywords = set()
        older_keywords = set()
        
        cutoff = datetime.now() - timedelta(hours=24)
        for tp in time_series:
            if tp.timestamp > cutoff:
                recent_keywords.update(tp.top_keywords)
            else:
                older_keywords.update(tp.top_keywords)
        
        emerging_themes = list(recent_keywords - older_keywords)[:5]
    
    # 8. 风险信号
    risk_signals = []
    risk_texts = [t for t in all_texts if any(kw in RISK_KEYWORDS for kw in t.keywords)]
    if len(risk_texts) > len(all_texts) * 0.1:
        risk_signals.append(f"检测到 {len(risk_texts)} 条风险相关文本，占比 {len(risk_texts)/len(all_texts):.1%}")
    
    # 9. 价格相关性
    price_correlation = None
    sentiment_lead = None
    if price_df is not None and time_series:
        price_correlation, sentiment_lead = calculate_price_correlation(time_series, price_df)
    
    # 10. 数据完整性
    data_completeness = 1.0 - (len(missing) / 5)  # 假设5个理想数据源
    data_completeness = max(0.0, min(1.0, data_completeness))
    
    return SentimentNLPResult(
        ticker=ticker,
        analysis_time=datetime.now(),
        overall_sentiment=overall_sentiment,
        overall_score=overall_score,
        confidence=confidence,
        time_series=time_series,
        news_sentiment=category_scores.get('news'),
        social_sentiment=category_scores.get('social'),
        earnings_sentiment=category_scores.get('earnings'),
        analyst_sentiment=category_scores.get('analyst'),
        top_keywords=top_keywords,
        emerging_themes=emerging_themes,
        price_correlation=price_correlation,
        sentiment_lead_hours=sentiment_lead,
        risk_signals=risk_signals,
        data_completeness=round(data_completeness, 2),
        missing_sources=missing,
        sample_texts=all_texts[:5],  # 保留前5条作为样本
    )

# ============================================================
# 格式化输出
# ============================================================

def format_sentiment_nlp_report(result: SentimentNLPResult) -> Dict:
    """格式化 NLP 情绪分析报告"""
    report = {
        "ticker": result.ticker,
        "analysis_time": result.analysis_time.strftime("%Y-%m-%d %H:%M"),
        "overall_sentiment": result.overall_sentiment,
        "overall_score": result.overall_score,
        "confidence": result.confidence,
        "top_keywords": result.top_keywords,
        "emerging_themes": result.emerging_themes,
        "price_correlation": result.price_correlation,
        "sentiment_lead_hours": result.sentiment_lead_hours,
        "risk_signals": result.risk_signals,
        "data_completeness": result.data_completeness,
        "missing_sources": result.missing_sources,
        "category_breakdown": {
            "news": result.news_sentiment,
            "social": result.social_sentiment,
            "earnings": result.earnings_sentiment,
            "analyst": result.analyst_sentiment,
        },
        "time_series_summary": [
            {
                "time": p.timestamp.strftime("%m-%d %H:%M"),
                "avg_score": p.avg_score,
                "volume": p.volume,
                "pos_ratio": p.pos_ratio,
                "neg_ratio": p.neg_ratio,
                "trend": p.sentiment_trend,
            }
            for p in result.time_series[-10:]  # 最近10个时间点
        ],
        "sample_texts": [
            {
                "text": t.text[:100] + "..." if len(t.text) > 100 else t.text,
                "score": t.sentiment_score,
                "category": t.category,
            }
            for t in result.sample_texts
        ],
    }
    
    return report
