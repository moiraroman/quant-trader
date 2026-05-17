# ============================================================
# ai/correlation_matrix.py — 相关性矩阵分析
# 功能：SPY/GLD/VIX/DXY/TLT/QQQ/IWM联动分析
# ============================================================
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationPair:
    """资产对相关性"""
    asset_a: str
    asset_b: str
    correlation_20d: float   # 20日相关系数
    correlation_60d: float   # 60日相关系数
    correlation_120d: float  # 120日相关系数
    beta: float = 0.0        # asset_a 对 asset_b 的beta
    relationship: str = ""   # "正相关"/"负相关"/"弱相关"
    regime: str = ""         # "强化"/"弱化"/"稳定"


@dataclass
class CorrelationResult:
    """相关性矩阵分析结果"""
    ticker: str              # 主分析标的
    analysis_time: str

    # 相关性矩阵数据
    pairs: list = field(default_factory=list)  # CorrelationPair列表

    # 矩阵表格（供UI展示）
    matrix_20d: dict = field(default_factory=dict)
    matrix_60d: dict = field(default_factory=dict)

    # 主标的的关键关系
    spy_correlation: float = 0.0
    gld_correlation: float = 0.0
    vix_correlation: float = 0.0
    dxy_correlation: float = 0.0
    tlt_correlation: float = 0.0

    # 联动解读
    risk_on_off_status: str = ""   # "Risk-On"/"Risk-Off"/"混合"
    diversification_score: float = 0.0  # 分散化评分 0-100

    # 对冲建议
    hedge_suggestions: list = field(default_factory=list)

    # 异常检测
    anomalies: list = field(default_factory=list)

    missing_data: list = field(default_factory=list)


# ============================================================
# 核心计算
# ============================================================

def _calculate_correlation(df_a: pd.Series, df_b: pd.Series, window: int) -> float:
    """计算两个序列的滚动相关系数（最新值）"""
    aligned = pd.concat([df_a, df_b], axis=1).dropna()
    if len(aligned) < window:
        return 0.0
    corr = aligned.iloc[-window:].corr().iloc[0, 1]
    return round(float(corr), 3) if not np.isnan(corr) else 0.0


def _calculate_beta(df_a: pd.Series, df_b: pd.Series, window: int = 60) -> float:
    """计算beta = cov(a,b) / var(b)"""
    aligned = pd.concat([df_a, df_b], axis=1).dropna()
    if len(aligned) < window:
        return 0.0
    recent = aligned.iloc[-window:]
    cov = recent.cov().iloc[0, 1]
    var = recent.iloc[:, 1].var()
    if var == 0:
        return 0.0
    return round(float(cov / var), 3)


def _relationship_label(corr: float) -> str:
    if corr > 0.5:
        return "强正相关"
    elif corr > 0.2:
        return "弱正相关"
    elif corr < -0.5:
        return "强负相关"
    elif corr < -0.2:
        return "弱负相关"
    return "无显著相关"


def _regime_label(corr_short: float, corr_long: float) -> str:
    """判断相关性是否在强化或弱化"""
    diff = abs(corr_short) - abs(corr_long)
    if diff > 0.15:
        return "强化"
    elif diff < -0.15:
        return "弱化"
    return "稳定"


# ============================================================
# 主分析入口
# ============================================================

def analyze_correlations(
    ticker: str,
    fetcher,
    lookback_days: int = 120,
) -> CorrelationResult:
    """
    相关性矩阵分析。

    分析标的：
      SPY(大盘), GLD(黄金), VIX(恐慌), DXY(美元), TLT(长债), QQQ(科技), IWM(小盘)

    数据源：yfinance历史日线（免费）
    """
    from datetime import datetime

    result = CorrelationResult(
        ticker=ticker,
        analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # 资产列表
    assets = {
        "SPY": "SPY",
        "GLD": "GLD",
        "VIX": "^VIX",
        "DXY": "UUP",  # UUP是美元指数ETF，DXY无直接ETF
        "TLT": "TLT",
        "QQQ": "QQQ",
        "IWM": "IWM",
    }

    # 获取所有数据
    price_data = {}
    returns_data = {}

    for name, symbol in assets.items():
        try:
            df = fetcher.download_history(symbol, period="1y", interval="1d")
            if df is not None and not df.empty and len(df) >= 20:
                price_data[name] = df["Close"]
                returns_data[name] = df["Close"].pct_change().dropna()
            else:
                result.missing_data.append(f"{name}({symbol})数据不足")
        except Exception as e:
            logger.warning(f"[Correlation] 获取{name}失败: {e}")
            result.missing_data.append(f"{name}({symbol})获取失败")

    if len(price_data) < 3:
        result.missing_data.append("有效资产数量不足，无法构建相关性矩阵")
        return result

    # 主标的的收益率序列
    main_returns = returns_data.get(ticker.upper(), None)
    if main_returns is None and ticker.upper() in price_data:
        main_returns = price_data[ticker.upper()].pct_change().dropna()

    # 计算所有配对的相关性
    asset_names = list(price_data.keys())
    matrix_20d = {a: {} for a in asset_names}
    matrix_60d = {a: {} for a in asset_names}

    for i, a in enumerate(asset_names):
        for j, b in enumerate(asset_names):
            if i >= j:
                matrix_20d[a][b] = 1.0 if a == b else None
                matrix_60d[a][b] = 1.0 if a == b else None
                continue

            try:
                # 对齐数据
                aligned = pd.concat([returns_data[a], returns_data[b]], axis=1).dropna()
                if len(aligned) < 20:
                    continue

                corr_20 = _calculate_correlation(returns_data[a], returns_data[b], 20)
                corr_60 = _calculate_correlation(returns_data[a], returns_data[b], 60)
                corr_120 = _calculate_correlation(returns_data[a], returns_data[b], 120) if len(aligned) >= 120 else corr_60

                beta = _calculate_beta(returns_data[a], returns_data[b], 60)

                pair = CorrelationPair(
                    asset_a=a,
                    asset_b=b,
                    correlation_20d=corr_20,
                    correlation_60d=corr_60,
                    correlation_120d=corr_120,
                    beta=beta,
                    relationship=_relationship_label(corr_60),
                    regime=_regime_label(corr_20, corr_60),
                )
                result.pairs.append(pair)

                matrix_20d[a][b] = corr_20
                matrix_20d[b][a] = corr_20
                matrix_60d[a][b] = corr_60
                matrix_60d[b][a] = corr_60

                # 记录主标的的关键关系
                if a == ticker.upper() or b == ticker.upper():
                    target = a if b == ticker.upper() else b
                    corr_val = corr_60
                    if target == "SPY":
                        result.spy_correlation = corr_val
                    elif target == "GLD":
                        result.gld_correlation = corr_val
                    elif target == "VIX":
                        result.vix_correlation = corr_val
                    elif target == "DXY":
                        result.dxy_correlation = corr_val
                    elif target == "TLT":
                        result.tlt_correlation = corr_val

            except Exception as e:
                logger.warning(f"[Correlation] 计算{a}-{b}相关性失败: {e}")

    result.matrix_20d = matrix_20d
    result.matrix_60d = matrix_60d

    # Risk-On/Off 判断
    if "VIX" in returns_data and "SPY" in returns_data:
        try:
            vix_spy_corr = _calculate_correlation(returns_data["VIX"], returns_data["SPY"], 20)
            # VIX与SPY通常负相关
            if vix_spy_corr < -0.5:
                result.risk_on_off_status = "典型Risk-Off环境（VIX↑→SPY↓）"
            elif vix_spy_corr > -0.2:
                result.risk_on_off_status = "Risk-On环境（VIX与SPY脱钩）"
            else:
                result.risk_on_off_status = "混合环境"
        except:
            pass

    # 分散化评分（基于与SPY的相关性）
    if ticker.upper() != "SPY" and result.spy_correlation != 0:
        # 与SPY相关性越低，分散化效果越好
        result.diversification_score = round(max(0, (1 - abs(result.spy_correlation)) * 100), 1)
    else:
        result.diversification_score = 0.0

    # 对冲建议
    if ticker.upper() == "SPY":
        if result.vix_correlation < -0.5:
            result.hedge_suggestions.append("VIX与SPY强负相关，可用VIX期权对冲")
        if result.gld_correlation < -0.2:
            result.hedge_suggestions.append("黄金与SPY负相关，可作为避险配置")
        if result.tlt_correlation < -0.3:
            result.hedge_suggestions.append("长债与SPY负相关，利率敏感型对冲")
    elif ticker.upper() == "GLD":
        if result.dxy_correlation < -0.4:
            result.hedge_suggestions.append("黄金与美元强负相关，关注美元指数")
        if result.spy_correlation < 0:
            result.hedge_suggestions.append("黄金与股市负相关，适合Risk-Off配置")

    # 异常检测
    for pair in result.pairs:
        # 相关性突变
        if abs(pair.correlation_20d - pair.correlation_60d) > 0.3:
            direction = "上升" if pair.correlation_20d > pair.correlation_60d else "下降"
            result.anomalies.append(
                f"{pair.asset_a}-{pair.asset_b}: 20日相关性较60日{direction}"
                f"({pair.correlation_20d:+.2f} vs {pair.correlation_60d:+.2f})"
            )
        # 关系反转
        if pair.correlation_20d * pair.correlation_60d < 0:
            result.anomalies.append(
                f"{pair.asset_a}-{pair.asset_b}: 相关性方向反转"
                f"(20日{pair.correlation_20d:+.2f} vs 60日{pair.correlation_60d:+.2f})"
            )

    return result


# ============================================================
# 格式化输出
# ============================================================

def format_correlation_result(result: CorrelationResult) -> dict:
    """格式化相关性结果为字典（供WebUI）"""
    # 矩阵表格格式化
    def _fmt_matrix(matrix: dict) -> list:
        assets = sorted(matrix.keys())
        rows = []
        for a in assets:
            row = {"资产": a}
            for b in assets:
                val = matrix.get(a, {}).get(b)
                if val is None:
                    row[b] = "—"
                elif a == b:
                    row[b] = "1.00"
                else:
                    row[b] = f"{val:+.2f}"
            rows.append(row)
        return rows

    return {
        "标的": result.ticker,
        "市场环境": result.risk_on_off_status,
        "分散化评分": f"{result.diversification_score:.0f}/100",
        "关键相关性(60日)": {
            "SPY": f"{result.spy_correlation:+.2f}" if result.spy_correlation != 0 else "N/A",
            "GLD": f"{result.gld_correlation:+.2f}" if result.gld_correlation != 0 else "N/A",
            "VIX": f"{result.vix_correlation:+.2f}" if result.vix_correlation != 0 else "N/A",
            "DXY": f"{result.dxy_correlation:+.2f}" if result.dxy_correlation != 0 else "N/A",
            "TLT": f"{result.tlt_correlation:+.2f}" if result.tlt_correlation != 0 else "N/A",
        },
        "相关性矩阵(20日)": _fmt_matrix(result.matrix_20d),
        "相关性矩阵(60日)": _fmt_matrix(result.matrix_60d),
        "配对详情": [
            {
                "配对": f"{p.asset_a}-{p.asset_b}",
                "20日": f"{p.correlation_20d:+.2f}",
                "60日": f"{p.correlation_60d:+.2f}",
                "120日": f"{p.correlation_120d:+.2f}",
                "关系": p.relationship,
                "趋势": p.regime,
                "Beta": p.beta,
            }
            for p in result.pairs[:10]
        ],
        "对冲建议": result.hedge_suggestions,
        "异常检测": result.anomalies if result.anomalies else ["无显著异常"],
        "缺少数据": result.missing_data,
    }
