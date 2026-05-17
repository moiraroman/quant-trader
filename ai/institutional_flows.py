# ============================================================
# ai/institutional_flows.py — 机构资金流向分析
# 功能：ETF资金流向近似、板块轮动检测、Smart Money信号
# 数据原则：真实数据优先，付费数据标注"缺少[数据源]"
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ETFFlow:
    """单个ETF资金流信息"""
    ticker: str
    name: str
    price_change_5d: float = 0.0
    price_change_1m: float = 0.0
    volume_ratio: float = 0.0  # 成交量/20日均量
    flow_signal: str = "neutral"  # "inflow" / "outflow" / "neutral"
    flow_strength: float = 0.0  # 0~100
    note: str = ""  # 说明数据来源或限制


@dataclass
class SectorRotation:
    """板块轮动信号"""
    leading_sector: str = ""
    leading_etfs: list = field(default_factory=list)
    lagging_sector: str = ""
    lagging_etfs: list = field(default_factory=list)
    rotation_strength: float = 0.0  # 0~100
    confidence: float = 0.0


@dataclass
class InstitutionalFlowsResult:
    """机构资金流向分析结果"""
    ticker: str
    analysis_time: str

    # 相关ETF流（基于价格+成交量近似）
    related_etfs: list = field(default_factory=list)  # ETFFlow列表

    # 板块轮动
    sector_rotation: Optional[SectorRotation] = None

    # Smart Money近似信号
    smart_money_signal: str = "neutral"  # "accumulation" / "distribution" / "neutral"
    smart_money_strength: float = 0.0
    smart_money_note: str = ""

    # 综合
    overall_flow_trend: str = "neutral"  # "inflow" / "outflow" / "neutral"
    overall_strength: float = 0.0

    # 缺少的数据
    missing_data: list = field(default_factory=list)


# ============================================================
# ETF列表配置
# ============================================================

# 主要ETF及其对应板块
ETF_UNIVERSE = {
    "SPY": {"name": "S&P 500", "sector": "大盘"},
    "QQQ": {"name": "Nasdaq 100", "sector": "科技"},
    "IWM": {"name": "Russell 2000", "sector": "小盘"},
    "VTI": {"name": "Total Stock Market", "sector": "全市场"},
    "GLD": {"name": "Gold", "sector": "贵金属"},
    "SLV": {"name": "Silver", "sector": "贵金属"},
    "USO": {"name": "WTI Oil", "sector": "能源"},
    "XLF": {"name": "Financials", "sector": "金融"},
    "XLK": {"name": "Technology", "sector": "科技"},
    "XLE": {"name": "Energy", "sector": "能源"},
    "XLI": {"name": "Industrials", "sector": "工业"},
    "XLP": {"name": "Consumer Staples", "sector": "必需消费"},
    "XLU": {"name": "Utilities", "sector": "公用事业"},
    "XLV": {"name": "Health Care", "sector": "医疗"},
    "XLY": {"name": "Consumer Disc.", "sector": "可选消费"},
    "XLB": {"name": "Materials", "sector": "材料"},
    "XLRE": {"name": "Real Estate", "sector": "房地产"},
    "TLT": {"name": "20+ Year Treasury", "sector": "长期国债"},
    "IEF": {"name": "7-10 Year Treasury", "sector": "中期国债"},
    "SHY": {"name": "1-3 Year Treasury", "sector": "短期国债"},
    "LQD": {"name": "Investment Grade Corp", "sector": "投资级债"},
    "HYG": {"name": "High Yield Corp", "sector": "高收益债"},
    "EMB": {"name": "Emerging Markets USD", "sector": "新兴市场债"},
    "UUP": {"name": "US Dollar Index", "sector": "美元"},
}


def _get_related_etfs(ticker: str) -> list[str]:
    """根据标的获取相关ETF列表"""
    ticker_upper = ticker.upper()

    # 直接匹配
    if ticker_upper in ETF_UNIVERSE:
        return [ticker_upper]

    # 根据类型推断
    sector_map = {
        "GLD": ["GLD", "SLV"],
        "SLV": ["SLV", "GLD"],
        "USO": ["USO", "XLE"],
        "AAPL": ["QQQ", "XLK"],
        "MSFT": ["QQQ", "XLK"],
        "GOOGL": ["QQQ", "XLK"],
        "AMZN": ["QQQ", "XLY"],
        "TSLA": ["QQQ", "XLY"],
        "JPM": ["XLF"],
        "BAC": ["XLF"],
        "XOM": ["XLE"],
        "CVX": ["XLE"],
        "JNJ": ["XLV"],
        "PFE": ["XLV"],
        "WMT": ["XLP", "VTI"],
        "COST": ["XLP", "XLY"],
    }

    if ticker_upper in sector_map:
        return sector_map[ticker_upper]

    # 默认返回大盘ETF
    return ["SPY", "VTI"]


# ============================================================
# 核心分析函数
# ============================================================

def _analyze_etf_flow(etf_ticker: str, fetcher) -> Optional[ETFFlow]:
    """
    分析单个ETF的"资金流"信号。
    由于免费数据没有真实资金流，用价格+成交量近似：
      - 价升量增 → 疑似流入
      - 价跌量增 → 疑似流出
      - 价升量缩 → 疑似弱势上涨
      - 价跌量缩 → 疑似弱势下跌
    """
    try:
        df_5d = fetcher.download_history(etf_ticker, period="5d", interval="1d")
        df_1m = fetcher.download_history(etf_ticker, period="1mo", interval="1d")

        if df_5d is None or df_5d.empty or len(df_5d) < 2:
            return None

        # 5日价格变化
        price_5d = (df_5d["Close"].iloc[-1] - df_5d["Close"].iloc[0]) / df_5d["Close"].iloc[0] * 100

        # 1月价格变化
        price_1m = 0.0
        if df_1m is not None and len(df_1m) > 5:
            price_1m = (df_1m["Close"].iloc[-1] - df_1m["Close"].iloc[0]) / df_1m["Close"].iloc[0] * 100

        # 成交量比率（最近1日 vs 20日均量）
        vol_ratio = 1.0
        if df_1m is not None and len(df_1m) >= 20:
            recent_vol = df_1m["Volume"].tail(5).mean()
            avg_vol = df_1m["Volume"].tail(20).mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # 资金流信号（近似）
        flow_signal = "neutral"
        flow_strength = 0.0

        if price_5d > 1.0 and vol_ratio > 1.2:
            flow_signal = "inflow"
            flow_strength = min(100, abs(price_5d) * 8 + (vol_ratio - 1) * 20)
        elif price_5d < -1.0 and vol_ratio > 1.2:
            flow_signal = "outflow"
            flow_strength = min(100, abs(price_5d) * 8 + (vol_ratio - 1) * 20)
        elif abs(price_5d) > 2.0:
            flow_signal = "inflow" if price_5d > 0 else "outflow"
            flow_strength = min(60, abs(price_5d) * 5)

        info = ETF_UNIVERSE.get(etf_ticker, {"name": etf_ticker, "sector": "未知"})

        return ETFFlow(
            ticker=etf_ticker,
            name=info["name"],
            price_change_5d=round(price_5d, 2),
            price_change_1m=round(price_1m, 2),
            volume_ratio=round(vol_ratio, 2),
            flow_signal=flow_signal,
            flow_strength=round(flow_strength, 1),
            note="基于价格+成交量的近似估算，非真实资金流",
        )
    except Exception as e:
        logger.warning(f"[InstFlows] {etf_ticker} 分析失败: {e}")
        return None


def _detect_sector_rotation(flows: list[ETFFlow]) -> Optional[SectorRotation]:
    """基于ETF表现检测板块轮动"""
    if len(flows) < 3:
        return None

    # 按5日涨幅排序
    sorted_flows = sorted(flows, key=lambda x: x.price_change_5d, reverse=True)

    leading = sorted_flows[0]
    lagging = sorted_flows[-1]

    # 轮动强度 = 领先与落后的差距
    spread = leading.price_change_5d - lagging.price_change_5d
    rotation_strength = min(100, spread * 5)

    if spread < 1.0:
        return None  # 差距太小，无明显轮动

    leading_info = ETF_UNIVERSE.get(leading.ticker, {"sector": "未知"})
    lagging_info = ETF_UNIVERSE.get(lagging.ticker, {"sector": "未知"})

    return SectorRotation(
        leading_sector=leading_info["sector"],
        leading_etfs=[leading.ticker],
        lagging_sector=lagging_info["sector"],
        lagging_etfs=[lagging.ticker],
        rotation_strength=round(rotation_strength, 1),
        confidence=min(100, spread * 8 + 20),
    )


def _smart_money_approximation(
    ticker: str,
    latest_price: float,
    fetcher,
) -> tuple[str, float, str]:
    """
    Smart Money 近似信号。
    基于：
      1. 收盘价位置（收在高位=积累，收在低位=派发）
      2. 成交量分布（放量上涨 vs 放量下跌）
      3. 与大盘相关性偏离
    注意：这只是近似，真实Smart Money需要付费数据
    """
    try:
        df = fetcher.download_history(ticker, period="20d", interval="1d")
        if df is None or len(df) < 10:
            return "neutral", 0.0, "数据不足"

        # 最近5日分析
        recent = df.tail(5)
        accum_score = 0.0

        for _, row in recent.iterrows():
            open_p = row["Open"]
            high_p = row["High"]
            low_p = row["Low"]
            close_p = row["Close"]
            vol = row["Volume"]

            if high_p == low_p:
                continue

            # 收盘价在日内区间的位置
            position = (close_p - low_p) / (high_p - low_p)

            # 成交量权重（与20日均量比较）
            avg_vol = df["Volume"].mean()
            vol_weight = vol / avg_vol if avg_vol > 0 else 1.0

            if position > 0.7 and vol_weight > 1.1:
                accum_score += position * vol_weight
            elif position < 0.3 and vol_weight > 1.1:
                accum_score -= (1 - position) * vol_weight

        # 标准化
        if accum_score > 2.0:
            return "accumulation", min(100, accum_score * 15), "基于收盘价位置+成交量近似"
        elif accum_score < -2.0:
            return "distribution", min(100, abs(accum_score) * 15), "基于收盘价位置+成交量近似"
        else:
            return "neutral", 0.0, "无明显积累/派发信号"

    except Exception as e:
        logger.warning(f"[InstFlows] Smart Money分析失败: {e}")
        return "neutral", 0.0, f"分析异常: {e}"


# ============================================================
# 主分析入口
# ============================================================

def analyze_institutional_flows(
    ticker: str,
    latest_price: float,
    fetcher,
) -> InstitutionalFlowsResult:
    """
    分析机构资金流向。

    数据源：
      - yfinance ETF历史数据（免费）
      - 近似逻辑：价格+成交量推断资金流向
      - 缺失：真实ETF资金流（Bloomberg/ETF.com付费数据）
    """
    result = InstitutionalFlowsResult(
        ticker=ticker,
        analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # 1. 分析相关ETF
    related = _get_related_etfs(ticker)
    etf_flows = []
    for etf in related:
        flow = _analyze_etf_flow(etf, fetcher)
        if flow:
            etf_flows.append(flow)

    # 额外分析大盘ETF作为基准
    for benchmark in ["SPY", "QQQ", "IWM"]:
        if benchmark not in [f.ticker for f in etf_flows]:
            flow = _analyze_etf_flow(benchmark, fetcher)
            if flow:
                etf_flows.append(flow)

    result.related_etfs = etf_flows

    # 2. 板块轮动
    if len(etf_flows) >= 3:
        result.sector_rotation = _detect_sector_rotation(etf_flows)

    # 3. Smart Money近似
    sm_signal, sm_strength, sm_note = _smart_money_approximation(ticker, latest_price, fetcher)
    result.smart_money_signal = sm_signal
    result.smart_money_strength = round(sm_strength, 1)
    result.smart_money_note = sm_note

    # 4. 综合判断
    inflow_count = sum(1 for f in etf_flows if f.flow_signal == "inflow")
    outflow_count = sum(1 for f in etf_flows if f.flow_signal == "outflow")
    total = len(etf_flows)

    if total > 0:
        inflow_ratio = inflow_count / total
        outflow_ratio = outflow_count / total

        if inflow_ratio > 0.5:
            result.overall_flow_trend = "inflow"
            result.overall_strength = min(100, inflow_ratio * 100)
        elif outflow_ratio > 0.5:
            result.overall_flow_trend = "outflow"
            result.overall_strength = min(100, outflow_ratio * 100)
        else:
            result.overall_flow_trend = "neutral"
            result.overall_strength = 0.0

    # 标注缺失
    result.missing_data.append("真实ETF资金流（付费：Bloomberg/ETF.com/Morningstar）")
    result.missing_data.append("机构13F持仓数据（SEC EDGAR，季度延迟）")
    result.missing_data.append("暗池/大宗交易数据（付费：FINRA）")

    return result


# ============================================================
# 格式化输出
# ============================================================

def format_flows_result(result: InstitutionalFlowsResult) -> dict:
    """格式化资金流向结果为字典（供WebUI）"""
    return {
        "标的": result.ticker,
        "整体流向": result.overall_flow_trend,
        "流向强度": f"{result.overall_strength:.0f}%",
        "Smart Money": {
            "信号": result.smart_money_signal,
            "强度": f"{result.smart_money_strength:.0f}%",
            "说明": result.smart_money_note,
        },
        "相关ETF": [
            {
                "代码": f.ticker,
                "名称": f.name,
                "5日涨跌": f"{f.price_change_5d:+.2f}%",
                "1月涨跌": f"{f.price_change_1m:+.2f}%",
                "量比": f.volume_ratio,
                "流向信号": f.flow_signal,
                "流向强度": f"{f.flow_strength:.0f}%",
                "备注": f.note,
            }
            for f in result.related_etfs
        ],
        "板块轮动": {
            "领先板块": result.sector_rotation.leading_sector if result.sector_rotation else "N/A",
            "落后板块": result.sector_rotation.lagging_sector if result.sector_rotation else "N/A",
            "轮动强度": f"{result.sector_rotation.rotation_strength:.0f}%" if result.sector_rotation else "N/A",
        },
        "缺少数据": result.missing_data,
    }
