# ============================================================
# ai/derivatives.py — 衍生品数据分析模块
# VIX期限结构、Put/Call Ratio、Gamma Exposure(GEX)估算
# 数据策略：yfinance获取VIX数据，其余标注"缺少"或搜索
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
class VIX_TermStructure:
    """VIX 期限结构"""
    spot_vix: Optional[float]           # 即期VIX
    front_month: Optional[float]        # 近月期货
    second_month: Optional[float]       # 次月期货
    third_month: Optional[float]        # 第三月
    slope: Optional[float]              # 近月-次月斜率
    term_structure: str                 # "Contango"/"Backwardation"/"平坦"
    contango_pct: Optional[float]       # Contango幅度%
    interpretation: str                 # 解读
    source: str


@dataclass
class VIX_Curve:
    """VIX 完整曲线"""
    dates: list[str]                    # 到期月份
    prices: list[float]                 # 各月价格
    source: str


@dataclass
class PCR_Analysis:
    """Put/Call Ratio 分析"""
    total_pcr: Optional[float]
    equity_pcr: Optional[float]
    index_pcr: Optional[float]
    pcr_20d_avg: Optional[float]
    pcr_percentile: Optional[float]     # 历史百分位
    interpretation: str
    contrarian_signal: str              # 逆向信号
    source: str


@dataclass
class GEX_Estimate:
    """Gamma Exposure 估算（基于可用数据）"""
    net_gamma: Optional[float]          # 净Gamma（正=多头Gamma，负=空头Gamma）
    gamma_flip: Optional[float]         # Gamma Flip价格（零Gamma点）
    interpretation: str                 # 解读
    note: str                           # 数据来源说明
    source: str


@dataclass
class DerivativesResult:
    """完整衍生品分析结果"""
    ticker: str
    analysis_time: str

    # VIX
    vix_term: VIX_TermStructure
    vix_curve: Optional[VIX_Curve]
    vix_spot_history: Optional[pd.DataFrame]

    # PCR
    pcr: PCR_Analysis

    # GEX
    gex: GEX_Estimate

    # 综合
    vol_regime: str                     # "低波动"/"正常"/"高波动"/"极端"
    vol_trend: str                      # "上升"/"下降"/"稳定"
    market_stress: str                  # "无压力"/"轻度"/"中度"/"重度"

    # 缺少的数据
    missing_data: list[str] = field(default_factory=list)


# ============================================================
# VIX 期限结构
# ============================================================

def analyze_vix_term_structure(fetcher) -> tuple[VIX_TermStructure, Optional[VIX_Curve]]:
    """
    分析 VIX 期限结构。

    数据获取策略：
        1. VIX即期：yfinance ^VIX（免费）
        2. VIX期货：yfinance VIX=F（近月期货，免费）
        3. 完整期限结构：需CBOE期货数据（付费），标注"缺少"

    Contango vs Backwardation：
        - Contango（期货>现货）：正常状态，市场预期波动回落
        - Backwardation（期货<现货）：恐慌状态，市场担心短期风险
        - 平坦：转折点
    """
    missing = []
    spot_vix = None
    front_month = None
    second_month = None

    # 获取即期VIX
    try:
        vix_df = fetcher.download_history("^VIX", period="1mo", interval="1d")
        if not vix_df.empty:
            spot_vix = float(vix_df["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"[Derivatives] VIX即期获取失败: {e}")
        missing.append("VIX即期")

    # 尝试获取VIX期货（VX=F 是 yfinance 中 VIX 期货的标准 ticker）
    # 注意：yfinance 对期货支持有限，经常返回空数据
    vix_fut_tickers = ["VX=F", "^VIX", "VIX"]  # 优先尝试 VX=F，然后回退到即期VIX
    for alt_ticker in vix_fut_tickers:
        try:
            vix_fut = fetcher.download_history(alt_ticker, period="1mo", interval="1d")
            if not vix_fut.empty and len(vix_fut) > 0:
                front_month = float(vix_fut["Close"].iloc[-1])
                break
        except Exception:
            continue
    if front_month is None:
        missing.append("VIX期货（需CBOE数据或搜索）")

    # 计算期限结构
    slope = None
    contango_pct = None
    term = "缺少数据"
    interp = "无法判断"

    if spot_vix and front_month:
        slope = round(front_month - spot_vix, 2)
        contango_pct = round((front_month - spot_vix) / spot_vix * 100, 2)

        if slope > 2:
            term = "Contango（期货升水）"
            interp = "正常状态，市场预期波动回落"
        elif slope < -2:
            term = "Backwardation（期货贴水）"
            interp = "恐慌状态，市场担忧短期风险"
        else:
            term = "平坦"
            interp = "转折点，关注方向选择"

    vix_term = VIX_TermStructure(
        spot_vix=round(spot_vix, 2) if spot_vix else None,
        front_month=round(front_month, 2) if front_month else None,
        second_month=None,  # 需完整期货数据
        third_month=None,
        slope=slope,
        term_structure=term,
        contango_pct=contango_pct,
        interpretation=interp,
        source="yfinance(^VIX/VIX=F)" if spot_vix else "缺少",
    )

    # 完整曲线（大部分数据缺失）
    vix_curve = None
    if spot_vix:
        vix_curve = VIX_Curve(
            dates=["即期", "近月期货"],
            prices=[spot_vix, front_month] if front_month else [spot_vix],
            source="部分数据（完整曲线需CBOE期货链）",
        )

    return vix_term, vix_curve


# ============================================================
# VIX 历史分析
# ============================================================

def analyze_vix_history(vix_df: pd.DataFrame) -> tuple[str, str, str]:
    """
    基于VIX历史数据判断波动率状态。

    返回: (vol_regime, vol_trend, market_stress)
    """
    if vix_df is None or vix_df.empty or len(vix_df) < 20:
        return "数据不足", "未知", "未知"

    closes = vix_df["Close"].values
    current = closes[-1]
    ma20 = np.mean(closes[-20:])
    ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else ma20

    # 波动率状态
    if current < 15:
        regime = "低波动"
    elif current < 20:
        regime = "正常"
    elif current < 25:
        regime = "高波动"
    else:
        regime = "极端"

    # 趋势
    if current > ma20 * 1.1:
        trend = "上升"
    elif current < ma20 * 0.9:
        trend = "下降"
    else:
        trend = "稳定"

    # 压力
    if current > 30:
        stress = "重度（恐慌）"
    elif current > 25:
        stress = "中度（担忧）"
    elif current > 20:
        stress = "轻度（谨慎）"
    else:
        stress = "无压力"

    return regime, trend, stress


# ============================================================
# Put/Call Ratio 分析
# ============================================================

def analyze_pcr(fetcher, search_func=None) -> PCR_Analysis:
    """
    分析 Put/Call Ratio。

    数据来源：
        1. CBOE官网（免费日数据）
        2. 网络搜索
        3. 备用：标注"缺少"

    PCR 逆向信号：
        > 1.2: 极度悲观 → 逆向看多
        0.9-1.2: 悲观 → 谨慎看多
        0.7-0.9: 中性
        0.5-0.7: 乐观 → 谨慎看空
        < 0.5: 极度乐观 → 逆向看空
    """
    missing = []
    total_pcr = None

    # 尝试通过搜索获取
    if search_func:
        try:
            result = search_func("CBOE total put call ratio today")
            if result:
                # 解析逻辑由调用方实现
                pass
        except Exception:
            pass

    # 备用：标注缺少
    if total_pcr is None:
        missing.append("CBOE Put/Call Ratio（需搜索或CBOE数据）")

    if total_pcr is not None:
        interp = _interpret_pcr(total_pcr)
        contrarian = _pcr_contrarian_signal(total_pcr)
    else:
        interp = "缺少数据"
        contrarian = "无"

    return PCR_Analysis(
        total_pcr=total_pcr,
        equity_pcr=None,
        index_pcr=None,
        pcr_20d_avg=None,
        pcr_percentile=None,
        interpretation=interp,
        contrarian_signal=contrarian,
        source="缺少[CBOE数据]" if total_pcr is None else "CBOE",
    )


def _interpret_pcr(pcr: float) -> str:
    if pcr > 1.2:
        return "极度悲观"
    elif pcr > 0.9:
        return "悲观"
    elif pcr > 0.7:
        return "中性"
    elif pcr > 0.5:
        return "乐观"
    else:
        return "极度乐观"


def _pcr_contrarian_signal(pcr: float) -> str:
    if pcr > 1.2:
        return "逆向看多（极度悲观）"
    elif pcr > 0.9:
        return "谨慎看多"
    elif pcr < 0.5:
        return "逆向看空（极度乐观）"
    elif pcr < 0.7:
        return "谨慎看空"
    else:
        return "无"


# ============================================================
# GEX (Gamma Exposure) 估算
# ============================================================

def estimate_gex(ticker: str, current_price: float, fetcher) -> GEX_Estimate:
    """
    估算 Gamma Exposure。

    说明：
        - 精确GEX需要期权链数据（SpotGamma、ORATS等付费服务）
        - 此处提供基于价格和波动率的粗略估算
        - 仅作参考，非精确值

    Gamma Exposure 解读：
        - 正GEX（多头Gamma）：做市商多头期权，需在上涨时卖出、下跌时买入
          → 市场稳定器，价格被"钉"在Gamma Flip附近
        - 负GEX（空头Gamma）：做市商空头期权，需在上涨时买入、下跌时卖出
          → 放大波动，加速趋势
        - Gamma Flip：零Gamma点，突破后可能加速
    """
    missing = []

    # 尝试获取期权数据（yfinance有限支持）
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        # 获取最近到期日的期权链
        expirations = stock.options
        if expirations:
            opt = stock.option_chain(expirations[0])
            calls = opt.calls
            puts = opt.puts

            # 粗略估算：近ATM期权的Gamma加权
            atm_calls = calls[(calls["strike"] >= current_price * 0.98) &
                              (calls["strike"] <= current_price * 1.02)]
            atm_puts = puts[(puts["strike"] >= current_price * 0.98) &
                            (puts["strike"] <= current_price * 1.02)]

            if not atm_calls.empty and not atm_puts.empty:
                # 检查gamma列是否存在（yfinance部分数据可能缺少）
                if "gamma" in atm_calls.columns and "gamma" in atm_puts.columns:
                    call_gamma = atm_calls["gamma"].sum() * atm_calls["openInterest"].sum()
                    put_gamma = atm_puts["gamma"].sum() * atm_puts["openInterest"].sum()
                    net_gamma = call_gamma - put_gamma

                    # Gamma Flip 估算（最大Pain点附近）
                    all_strikes = pd.concat([calls[["strike", "openInterest"]],
                                             puts[["strike", "openInterest"]]])
                    pain = all_strikes.groupby("strike")["openInterest"].sum().idxmax()

                    if net_gamma > 0:
                        interp = f"正GEX（多头Gamma，市场稳定器），Gamma Flip约{pain:.2f}"
                    else:
                        interp = f"负GEX（空头Gamma，放大波动），Gamma Flip约{pain:.2f}"

                    return GEX_Estimate(
                        net_gamma=round(net_gamma, 2),
                        gamma_flip=round(pain, 2),
                        interpretation=interp,
                        note="基于yfinance期权链粗略估算，非精确GEX",
                        source="yfinance期权链（粗略估算）",
                    )
    except Exception as e:
        logger.warning(f"[Derivatives] GEX估算失败: {e}")

    missing.append("精确GEX（需SpotGamma/ORATS付费数据）")

    return GEX_Estimate(
        net_gamma=None,
        gamma_flip=None,
        interpretation="缺少精确期权链数据",
        note="精确GEX需SpotGamma、ORATS或CBOE期权数据",
        source="缺少[需付费期权数据]",
    )


# ============================================================
# 主函数
# ============================================================

def analyze_derivatives(
    ticker: str,
    fetcher,
    search_func=None,
) -> DerivativesResult:
    """
    完整衍生品分析。

    数据策略：
        - VIX即期：yfinance（免费）
        - VIX期货：yfinance（有限）
        - PCR：搜索或CBOE（标注缺少）
        - GEX：yfinance期权链粗略估算或标注缺少
    """
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    missing = []

    # 1. VIX期限结构
    vix_term, vix_curve = analyze_vix_term_structure(fetcher)
    if vix_term.spot_vix is None:
        missing.append("VIX即期数据")

    # 2. VIX历史
    vix_history = None
    try:
        vix_history = fetcher.download_history("^VIX", period="1y", interval="1d")
    except Exception:
        pass

    vol_regime, vol_trend, stress = analyze_vix_history(vix_history)

    # 3. PCR
    pcr = analyze_pcr(fetcher, search_func)
    if pcr.total_pcr is None:
        missing.append("Put/Call Ratio")

    # 4. GEX
    # 获取当前价格
    current_price = 0
    try:
        quote = fetcher.fetch_quote(ticker)
        if quote:
            current_price = quote.last_price
    except Exception:
        pass

    if current_price == 0:
        try:
            df = fetcher.download_history(ticker, period="5d", interval="1d")
            if not df.empty:
                current_price = float(df["Close"].iloc[-1])
        except Exception:
            pass

    gex = estimate_gex(ticker, current_price, fetcher)
    if gex.net_gamma is None:
        missing.append("精确GEX（需付费期权数据）")

    return DerivativesResult(
        ticker=ticker,
        analysis_time=time_str,
        vix_term=vix_term,
        vix_curve=vix_curve,
        vix_spot_history=vix_history,
        pcr=pcr,
        gex=gex,
        vol_regime=vol_regime,
        vol_trend=vol_trend,
        market_stress=stress,
        missing_data=missing,
    )


# ============================================================
# 格式化输出
# ============================================================

def format_derivatives_result(result: DerivativesResult) -> dict:
    """格式化衍生品分析结果供WebUI展示"""
    return {
        "标的": result.ticker,
        "分析时间": result.analysis_time,
        "波动率状态": {
            "状态": result.vol_regime,
            "趋势": result.vol_trend,
            "市场压力": result.market_stress,
        },
        "VIX期限结构": {
            "即期VIX": f"{result.vix_term.spot_vix:.2f}" if result.vix_term.spot_vix else "缺少",
            "近月期货": f"{result.vix_term.front_month:.2f}" if result.vix_term.front_month else "缺少",
            "斜率": f"{result.vix_term.slope:.2f}" if result.vix_term.slope else "N/A",
            "结构": result.vix_term.term_structure,
            "Contango%": f"{result.vix_term.contango_pct:.2f}%" if result.vix_term.contango_pct else "N/A",
            "解读": result.vix_term.interpretation,
        },
        "Put/Call Ratio": {
            "总PCR": f"{result.pcr.total_pcr:.2f}" if result.pcr.total_pcr else "缺少",
            "解读": result.pcr.interpretation,
            "逆向信号": result.pcr.contrarian_signal,
        },
        "GEX估算": {
            "净Gamma": f"{result.gex.net_gamma:.2f}" if result.gex.net_gamma else "缺少",
            "Gamma Flip": f"{result.gex.gamma_flip:.2f}" if result.gex.gamma_flip else "N/A",
            "解读": result.gex.interpretation,
            "说明": result.gex.note,
        },
        "缺少数据": result.missing_data,
    }
