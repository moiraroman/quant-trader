# ============================================================
# ai/event_analyzer.py — 事件驱动分析
# 财报/政策/宏观事件前后价格影响分析
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EventImpact:
    """单一事件影响分析"""
    event_type: str  # "earnings", "fed_meeting", "cpi", "nfp", "geopolitical"
    event_date: str
    description: str
    # 价格影响
    pre_event_return: float = 0.0   # 事件前5天收益
    day_of_return: float = 0.0      # 事件当天收益
    post_event_return: float = 0.0  # 事件后5天收益
    post_event_volatility: float = 0.0  # 事件后波动率变化
    # 统计
    historical_avg_impact: float = 0.0   # 历史同类事件平均影响
    historical_win_rate: float = 0.0     # 历史同类事件上涨概率
    sample_size: int = 0                 # 历史样本数
    # 异常检测
    is_outlier: bool = False             # 是否异常（超出2σ）
    # 缺失数据
    missing_data: list = field(default_factory=list)


@dataclass
class EventCalendar:
    """事件日历"""
    ticker: str
    upcoming_events: list = field(default_factory=list)   # 未来已知事件
    recent_events: list = field(default_factory=list)     # 近期已发生事件
    # 风险评级
    event_risk_score: float = 0.0  # 0-100，越高事件风险越大
    # 建议
    recommendation: str = ""


@dataclass
class EventAnalyzerResult:
    """事件分析总结果"""
    ticker: str
    earnings_impact: dict = field(default_factory=dict)
    fed_policy_impact: dict = field(default_factory=dict)
    macro_data_impact: dict = field(default_factory=dict)
    upcoming_risks: list = field(default_factory=list)
    historical_patterns: dict = field(default_factory=dict)
    # 综合
    event_risk_rating: str = "低"  # 低/中/高
    missing_data: list = field(default_factory=list)
    disclaimer: str = "事件分析基于历史模式，每次事件的具体影响可能因市场环境而异"


def _find_earnings_dates(ticker: str, fetcher) -> list[str]:
    """
    获取财报日期。
    注意：yfinance不直接提供历史财报日期，需通过搜索或外部API。
    此处返回空列表并标注缺失。
    """
    return []


def _analyze_earnings_impact(
    ticker: str,
    fetcher,
    lookback_years: int = 3,
) -> dict:
    """
    分析历史财报对价格的影响。
    由于缺少历史财报日期数据，使用季度末作为近似。
    """
    result = {
        "事件类型": "财报",
        "说明": "使用季度末作为财报日期近似（真实财报日期需付费数据）",
        "历史影响": [],
        "平均影响": "N/A",
        "上涨概率": "N/A",
    }

    try:
        df = fetcher.download_history(ticker, period=f"{lookback_years * 365}d", interval="1d")
        if df.empty or len(df) < 100:
            result["缺少数据"] = ["历史数据不足"]
            return result

        # 使用季度末作为近似财报日
        df.index = pd.to_datetime(df.index)
        quarter_ends = df[df.index.is_quarter_end]

        impacts = []
        for date in quarter_ends.index[-lookback_years * 4:]:
            try:
                idx = df.index.get_loc(date)
                if idx < 5 or idx >= len(df) - 5:
                    continue
                pre = (df["Close"].iloc[idx] - df["Close"].iloc[idx - 5]) / df["Close"].iloc[idx - 5] * 100
                day = (df["Close"].iloc[idx + 1] - df["Close"].iloc[idx]) / df["Close"].iloc[idx] * 100
                post = (df["Close"].iloc[idx + 5] - df["Close"].iloc[idx]) / df["Close"].iloc[idx] * 100
                impacts.append({
                    "日期": date.strftime("%Y-%m-%d"),
                    "前5天": round(pre, 2),
                    "当天+1": round(day, 2),
                    "后5天": round(post, 2),
                })
            except Exception:
                continue

        if impacts:
            day_returns = [i["当天+1"] for i in impacts]
            result["历史影响"] = impacts[-8:]  # 最近8次
            result["平均影响"] = f"{np.mean(day_returns):.2f}%"
            result["上涨概率"] = f"{np.mean([r > 0 for r in day_returns]) * 100:.0f}%"
            result["样本数"] = len(impacts)
        else:
            result["缺少数据"] = ["无法识别财报日期"]

    except Exception as e:
        logger.warning(f"[Event] {ticker} 财报分析失败: {e}")
        result["缺少数据"] = [f"分析失败: {e}"]

    return result


def _analyze_fed_meeting_impact(
    ticker: str,
    fetcher,
) -> dict:
    """
    分析FOMC会议对价格的影响。
    FOMC日期固定（每年8次），使用预设日期。
    """
    result = {
        "事件类型": "FOMC会议",
        "说明": "FOMC每年8次，使用预设会议日期（2023-2025）",
        "历史影响": [],
        "平均影响": "N/A",
        "上涨概率": "N/A",
    }

    # 预设FOMC日期（2023-2025主要会议）
    fomc_dates = [
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-11",
    ]

    try:
        df = fetcher.download_history(ticker, period="3y", interval="1d")
        if df.empty:
            result["缺少数据"] = ["历史数据不足"]
            return result

        df.index = pd.to_datetime(df.index)
        impacts = []

        for date_str in fomc_dates:
            try:
                date = pd.Timestamp(date_str)
                if date not in df.index:
                    # 找最近的交易日
                    nearest = df.index[df.index >= date]
                    if len(nearest) == 0:
                        continue
                    date = nearest[0]

                idx = df.index.get_loc(date)
                if idx < 2 or idx >= len(df) - 5:
                    continue

                day = (df["Close"].iloc[idx] - df["Close"].iloc[idx - 1]) / df["Close"].iloc[idx - 1] * 100
                post_1d = (df["Close"].iloc[idx + 1] - df["Close"].iloc[idx]) / df["Close"].iloc[idx] * 100
                post_5d = (df["Close"].iloc[idx + 5] - df["Close"].iloc[idx]) / df["Close"].iloc[idx] * 100

                impacts.append({
                    "日期": date_str,
                    "会议当天": round(day, 2),
                    "会后1天": round(post_1d, 2),
                    "会后5天": round(post_5d, 2),
                })
            except Exception:
                continue

        if impacts:
            day_returns = [i["会议当天"] for i in impacts]
            result["历史影响"] = impacts[-8:]
            result["平均影响"] = f"{np.mean(day_returns):.2f}%"
            result["上涨概率"] = f"{np.mean([r > 0 for r in day_returns]) * 100:.0f}%"
            result["样本数"] = len(impacts)
        else:
            result["缺少数据"] = ["无重叠FOMC日期数据"]

    except Exception as e:
        logger.warning(f"[Event] {ticker} FOMC分析失败: {e}")
        result["缺少数据"] = [f"分析失败: {e}"]

    return result


def _analyze_cpi_impact(
    ticker: str,
    fetcher,
) -> dict:
    """
    分析CPI发布对价格的影响。
    CPI每月发布，使用每月第二个周三作为近似。
    """
    result = {
        "事件类型": "CPI发布",
        "说明": "CPI每月中旬发布，使用每月第12-15个交易日作为近似",
        "历史影响": [],
        "平均影响": "N/A",
        "上涨概率": "N/A",
    }

    try:
        df = fetcher.download_history(ticker, period="2y", interval="1d")
        if df.empty:
            result["缺少数据"] = ["历史数据不足"]
            return result

        df.index = pd.to_datetime(df.index)
        impacts = []

        # 每月取第12-15个交易日作为CPI近似日
        for month_end in pd.date_range(end=df.index[-1], periods=24, freq="ME"):
            month_start = month_end.replace(day=1)
            month_data = df[(df.index >= month_start) & (df.index <= month_end)]
            if len(month_data) >= 15:
                cpi_day = month_data.index[13]  # 第14个交易日
                idx = df.index.get_loc(cpi_day)
                if idx < 1 or idx >= len(df) - 3:
                    continue
                day = (df["Close"].iloc[idx] - df["Close"].iloc[idx - 1]) / df["Close"].iloc[idx - 1] * 100
                post = (df["Close"].iloc[idx + 3] - df["Close"].iloc[idx]) / df["Close"].iloc[idx] * 100
                impacts.append({
                    "月份": month_start.strftime("%Y-%m"),
                    "发布日收益": round(day, 2),
                    "发布后3天": round(post, 2),
                })

        if impacts:
            day_returns = [i["发布日收益"] for i in impacts]
            result["历史影响"] = impacts[-6:]
            result["平均影响"] = f"{np.mean(day_returns):.2f}%"
            result["上涨概率"] = f"{np.mean([r > 0 for r in day_returns]) * 100:.0f}%"
            result["样本数"] = len(impacts)
        else:
            result["缺少数据"] = ["无法识别CPI发布日"]

    except Exception as e:
        logger.warning(f"[Event] {ticker} CPI分析失败: {e}")
        result["缺少数据"] = [f"分析失败: {e}"]

    return result


def analyze_events(
    ticker: str,
    fetcher,
) -> EventAnalyzerResult:
    """
    对指定标的执行完整事件驱动分析。
    """
    result = EventAnalyzerResult(ticker=ticker)

    # 财报影响
    result.earnings_impact = _analyze_earnings_impact(ticker, fetcher)
    if "缺少数据" in result.earnings_impact:
        result.missing_data.extend(result.earnings_impact["缺少数据"])

    # FOMC影响
    result.fed_policy_impact = _analyze_fed_meeting_impact(ticker, fetcher)
    if "缺少数据" in result.fed_policy_impact:
        result.missing_data.extend(result.fed_policy_impact["缺少数据"])

    # CPI影响
    result.macro_data_impact = _analyze_cpi_impact(ticker, fetcher)
    if "缺少数据" in result.macro_data_impact:
        result.missing_data.extend(result.macro_data_impact["缺少数据"])

    # 事件风险评级
    risk_count = sum(1 for k in [result.earnings_impact, result.fed_policy_impact, result.macro_data_impact]
                     if "缺少数据" not in k or k.get("样本数", 0) > 5)

    if risk_count >= 3:
        result.event_risk_rating = "高"
    elif risk_count >= 2:
        result.event_risk_rating = "中"
    else:
        result.event_risk_rating = "低"

    # 未来风险提醒
    today = datetime.now()
    # 下月FOMC（每6周一次，近似）
    next_fomc = today + timedelta(days=45)
    result.upcoming_risks.append(f"下次FOMC会议约在 {next_fomc.strftime('%Y-%m')}（每6周一次）")

    # 下季度财报
    next_quarter = today + timedelta(days=90)
    result.upcoming_risks.append(f"下次财报季约在 {next_quarter.strftime('%Y-%m')}（季度末）")

    return result


def format_event_result(result: EventAnalyzerResult) -> dict:
    """格式化事件分析结果为UI展示格式"""
    return {
        "标的": result.ticker,
        "事件风险评级": result.event_risk_rating,
        "财报影响": result.earnings_impact,
        "FOMC影响": result.fed_policy_impact,
        "CPI影响": result.macro_data_impact,
        " upcoming_risks": result.upcoming_risks,
        "历史模式": result.historical_patterns,
        "免责声明": result.disclaimer,
        "缺少数据": result.missing_data,
    }
