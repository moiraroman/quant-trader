# ============================================================
# ai/options_analyzer.py — 期权链分析
# 基于yfinance期权数据（免费）分析IV Skew、期限结构、Greeks估算
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptionsChainSummary:
    """期权链摘要"""
    ticker: str
    current_price: float
    expiration_date: str
    # 整体统计
    total_call_volume: int = 0
    total_put_volume: int = 0
    put_call_ratio: float = 0.0
    # 行权价分布
    call_oi_distribution: dict = field(default_factory=dict)
    put_oi_distribution: dict = field(default_factory=dict)
    # 最大痛点（近似）
    max_pain_strike: float = 0.0
    max_pain_pain: float = 0.0
    # IV Skew
    atm_iv: float = 0.0
    iv_skew_25d: float = 0.0   # 25 Delta Put IV - ATM IV
    iv_skew_10d: float = 0.0   # 10 Delta Put IV - ATM IV
    # 期限结构
    term_structure: dict = field(default_factory=dict)  # {days_to_expiry: atm_iv}
    # Greeks估算（基于Black-Scholes近似）
    estimated_delta: float = 0.0
    estimated_gamma: float = 0.0
    estimated_theta: float = 0.0
    estimated_vega: float = 0.0
    # 情绪信号
    sentiment_signal: str = "中性"
    sentiment_confidence: float = 0.0
    # 缺失数据
    missing_data: list = field(default_factory=list)
    disclaimer: str = "期权数据来自yfinance（免费），可能存在延迟，Greeks为估算值"


@dataclass
class OptionsAnalyzerResult:
    """期权分析总结果"""
    ticker: str
    current_price: float
    nearest_expiry: str = ""
    # 各到期日分析
    expiry_analysis: dict = field(default_factory=dict)  # {expiry: OptionsChainSummary}
    # 综合分析
    overall_pc_ratio: float = 0.0
    overall_sentiment: str = "中性"
    iv_term_structure_slope: float = 0.0  # 正=远期IV高（ contango），负=backwardation
    # 推荐关注
    key_strikes: list = field(default_factory=list)
    unusual_activity: list = field(default_factory=list)
    missing_data: list = field(default_factory=list)


def _black_scholes_greeks_approx(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """
    Black-Scholes Greeks近似计算。
    T: 年化到期时间
    """
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # 每1% IV变化

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }


def analyze_options_chain(
    ticker: str,
    current_price: float,
    fetcher,
    max_expirations: int = 4,
) -> OptionsAnalyzerResult:
    """
    分析期权链数据。

    参数:
        ticker: 标的代码
        current_price: 当前价格
        fetcher: YFinanceFetcher
        max_expirations: 分析的最大到期日数量

    返回:
        OptionsAnalyzerResult
    """
    result = OptionsAnalyzerResult(ticker=ticker, current_price=current_price)

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            result.missing_data.append("无可用期权数据")
            return result

        result.nearest_expiry = expirations[0]

        total_call_vol = 0
        total_put_vol = 0
        term_structure = {}

        for expiry in expirations[:max_expirations]:
            try:
                chain = stock.option_chain(expiry)
                calls = chain.calls
                puts = chain.puts

                summary = OptionsChainSummary(
                    ticker=ticker,
                    current_price=current_price,
                    expiration_date=expiry,
                )

                # 成交量统计
                summary.total_call_volume = int(calls["volume"].sum()) if "volume" in calls.columns else 0
                summary.total_put_volume = int(puts["volume"].sum()) if "volume" in puts.columns else 0

                if summary.total_call_volume > 0:
                    summary.put_call_ratio = round(summary.total_put_volume / summary.total_call_volume, 2)

                # 找到ATM行权价
                atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
                atm_put = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:1]]

                if "impliedVolatility" in atm_call.columns and len(atm_call) > 0:
                    summary.atm_iv = round(float(atm_call["impliedVolatility"].iloc[0]) * 100, 2)

                # IV Skew（25 Delta Put）
                try:
                    put_25d = puts[puts["strike"] <= current_price * 0.95].iloc[-1:] if len(puts) > 0 else None
                    if put_25d is not None and len(put_25d) > 0 and "impliedVolatility" in put_25d.columns:
                        iv_25d = float(put_25d["impliedVolatility"].iloc[0]) * 100
                        summary.iv_skew_25d = round(iv_25d - summary.atm_iv, 2)
                except Exception:
                    pass

                # 最大痛点近似（最大OI行权价）
                try:
                    if "openInterest" in calls.columns:
                        max_call_oi = calls.loc[calls["openInterest"].idxmax()]
                        summary.max_pain_strike = round(float(max_call_oi["strike"]), 2)
                except Exception:
                    pass

                # Greeks估算（ATM）
                try:
                    T = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days / 365.0
                    if T > 0 and summary.atm_iv > 0:
                        atm_strike = float(atm_call["strike"].iloc[0]) if len(atm_call) > 0 else current_price
                        greeks = _black_scholes_greeks_approx(
                            current_price, atm_strike, T, 0.04, summary.atm_iv / 100, "call"
                        )
                        summary.estimated_delta = greeks["delta"]
                        summary.estimated_gamma = greeks["gamma"]
                        summary.estimated_theta = greeks["theta"]
                        summary.estimated_vega = greeks["vega"]
                except Exception:
                    pass

                # 情绪信号
                if summary.put_call_ratio > 1.5:
                    summary.sentiment_signal = "偏空（Put成交量显著高于Call）"
                    summary.sentiment_confidence = min(80, summary.put_call_ratio * 30)
                elif summary.put_call_ratio < 0.7:
                    summary.sentiment_signal = "偏多（Call成交量显著高于Put）"
                    summary.sentiment_confidence = min(80, (1 / max(summary.put_call_ratio, 0.1)) * 20)
                else:
                    summary.sentiment_signal = "中性"
                    summary.sentiment_confidence = 50

                # 期限结构
                days_to_exp = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
                term_structure[days_to_exp] = summary.atm_iv

                result.expiry_analysis[expiry] = summary
                total_call_vol += summary.total_call_volume
                total_put_vol += summary.total_put_volume

            except Exception as e:
                logger.warning(f"[Options] {ticker} {expiry} 分析失败: {e}")
                continue

        # 整体统计
        result.overall_pc_ratio = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 0
        if result.overall_pc_ratio > 1.2:
            result.overall_sentiment = "偏空"
        elif result.overall_pc_ratio < 0.8:
            result.overall_sentiment = "偏多"
        else:
            result.overall_sentiment = "中性"

        # 期限结构斜率
        if len(term_structure) >= 2:
            days = sorted(term_structure.keys())
            ivs = [term_structure[d] for d in days]
            result.iv_term_structure_slope = round((ivs[-1] - ivs[0]) / (days[-1] - days[0]) * 30, 3)

        result.term_structure = {str(k): v for k, v in term_structure.items()}

        # 关键行权价（最大OI附近）
        try:
            nearest = result.expiry_analysis.get(expirations[0])
            if nearest:
                chain = stock.option_chain(expirations[0])
                calls = chain.calls
                top_oi = calls.nlargest(3, "openInterest") if "openInterest" in calls.columns else None
                if top_oi is not None:
                    result.key_strikes = [
                        {"行权价": round(float(row["strike"]), 2), "OI": int(row["openInterest"])}
                        for _, row in top_oi.iterrows()
                    ]
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"[Options] {ticker} 期权分析失败: {e}")
        result.missing_data.append(f"期权数据获取失败: {e}")

    return result


def format_options_result(result: OptionsAnalyzerResult) -> dict:
    """格式化期权分析结果为UI展示格式"""
    expiry_data = {}
    for exp, summary in result.expiry_analysis.items():
        expiry_data[exp] = {
            "到期日": exp,
            "Call成交量": summary.total_call_volume,
            "Put成交量": summary.total_put_volume,
            "Put/Call比": summary.put_call_ratio,
            "ATM IV": f"{summary.atm_iv}%",
            "IV Skew(25D)": f"{summary.iv_skew_25d}%",
            "最大痛点(近似)": summary.max_pain_strike,
            "Greeks估算": {
                "Delta": summary.estimated_delta,
                "Gamma": summary.estimated_gamma,
                "Theta": summary.estimated_theta,
                "Vega": summary.estimated_vega,
            },
            "情绪信号": summary.sentiment_signal,
            "情绪置信度": f"{summary.sentiment_confidence:.0f}%",
        }

    return {
        "标的": result.ticker,
        "当前价格": result.current_price,
        "最近到期日": result.nearest_expiry,
        "各到期日分析": expiry_data,
        "整体Put/Call比": result.overall_pc_ratio,
        "整体情绪": result.overall_sentiment,
        "IV期限结构斜率": f"{result.iv_term_structure_slope}%/月",
        "期限结构": result.term_structure,
        "关键行权价": result.key_strikes,
        "异常活动": result.unusual_activity,
        "免责声明": "期权数据来自yfinance免费版，Greeks为Black-Scholes估算",
        "缺少数据": result.missing_data,
    }
