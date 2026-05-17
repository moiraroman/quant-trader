# ============================================================
# ai/full_analyzer.py — 全量AI分析编排器 (v2)
# 新增P5模块集成 + 异步分析支持
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构（Step汇总）
# ============================================================

@dataclass
class Step1PriceResult:
    """步骤1：实时价格"""
    ticker: str
    latest_price: float
    currency: str = "USD"
    session: str = "盘中"    # "盘前" / "盘中" / "盘后" / "休市"
    quote_source: str = ""   # 数据来源
    quote_time: str = ""
    price_change_pct: float = 0.0
    previous_close: float = 0.0


@dataclass
class Step2TechResult:
    """步骤2：技术面分析（70%权重）"""
    ticker: str
    # 多时间框架
    mtf_summary: dict = field(default_factory=dict)    # format_mtf_result()
    # 支撑阻力
    sr_summary: dict = field(default_factory=dict)     # format_sr_result()
    # 关键数字
    rsi: Optional[float] = None
    adx: Optional[float] = None
    atr_pct: Optional[float] = None
    macd_hist: Optional[float] = None
    # 形态
    patterns: list = field(default_factory=list)
    # 背离
    divergences: list = field(default_factory=list)
    # 衍生品
    derivatives_summary: dict = field(default_factory=dict)
    # 成交量分布
    vp_summary: dict = field(default_factory=dict)
    # 经典理论
    classical_summary: dict = field(default_factory=dict)
    # 综合结论
    consensus_trend: str = ""
    consensus_confidence: float = 0.0
    top_bull_signals: list = field(default_factory=list)
    top_bear_signals: list = field(default_factory=list)
    # 缺少的数据
    missing_data: list = field(default_factory=list)


@dataclass
class Step3MacroResult:
    """步骤3：宏观基本面+情绪（30%权重）"""
    ticker: str
    macro_score: float = 50.0  # 0~100
    score_interpretation: str = "中性"
    # 各维度
    equity_score: float = 50.0
    vix_score: float = 50.0
    breadth_score: float = 50.0
    credit_score: float = 50.0
    safe_haven_score: float = 50.0
    dxy_score: float = 50.0
    # 情绪
    fear_greed: Optional[str] = None  # "恐惧" / "贪婪" / "中性"
    fear_greed_value: Optional[int] = None  # 0~100
    # 情绪详细（P1）
    sentiment_summary: dict = field(default_factory=dict)
    # 机构资金流（P2）
    flows_summary: dict = field(default_factory=dict)
    # 宏观政策（P2）
    policy_summary: dict = field(default_factory=dict)
    # 机构异动（P2）
    institutional_summary: dict = field(default_factory=dict)
    # 相关新闻
    recent_news: list = field(default_factory=list)
    # 缺少的数据
    missing_data: list = field(default_factory=list)
    # 综合
    outlook: str = "中性"


@dataclass
class Step4StrategyResult:
    """步骤4：交易策略"""
    ticker: str
    short_term_prob: dict = field(default_factory=dict)
    medium_term_prob: dict = field(default_factory=dict)
    short_strategy: dict = field(default_factory=dict)
    medium_strategy: dict = field(default_factory=dict)
    composite_signal: str = ""
    composite_confidence: float = 0.0


@dataclass
class Step5Summary:
    """步骤5：总结"""
    ticker: str
    composite_signal: str = ""
    composite_confidence: float = 0.0
    key_opportunities: list = field(default_factory=list)
    key_risks: list = field(default_factory=list)
    tech_bull: str = ""
    tech_bear: str = ""
    macro_bull: str = ""
    macro_bear: str = ""
    final_recommendation: str = ""


@dataclass
class FullAnalysisResult:
    """完整分析结果（5步骤 + P3 + P4 + P5）"""
    ticker: str
    tickers: list = field(default_factory=list)  # 多个标的
    analysis_time_et: str = ""
    step1_price: Step1PriceResult = None
    step2_tech: Step2TechResult = None
    step3_macro: Step3MacroResult = None
    step4_strategy: Step4StrategyResult = None
    step5_summary: Step5Summary = None
    # P3高级功能
    pattern_match: dict = field(default_factory=dict)      # 历史模式匹配
    dynamic_weights: dict = field(default_factory=dict)    # 动态权重
    scenarios: dict = field(default_factory=dict)          # 情景分析
    correlations: dict = field(default_factory=dict)       # 相关性矩阵
    # P4扩展功能
    backtest: dict = field(default_factory=dict)           # 策略回测
    monte_carlo: dict = field(default_factory=dict)        # 蒙特卡洛模拟
    ml_prediction: dict = field(default_factory=dict)      # ML预测
    alerts: dict = field(default_factory=dict)             # 预警系统
    # P5未来功能
    portfolio: dict = field(default_factory=dict)          # 组合优化
    events: dict = field(default_factory=dict)             # 事件驱动分析
    options: dict = field(default_factory=dict)            # 期权链分析
    nlp_sentiment: dict = field(default_factory=dict)      # NLP情绪
    # 数据完整性
    data_completeness: float = 0.0  # 0~100%，实际获取数据的比例
    missing_data_sources: list = field(default_factory=list)
    # 性能
    analysis_duration_sec: float = 0.0


# ============================================================
# Step 1: 实时价格
# ============================================================

def _fetch_realtime_price(ticker: str, fetcher) -> Step1PriceResult:
    """
    获取实时价格 + 判断交易时段。
    数据来源：yfinance.fast_info（免费，约15min延迟）
    """
    now = datetime.now(timezone.utc)
    hour_et = (now.hour - 5) % 24  # UTC → ET

    if hour_et < 4 or hour_et >= 20:
        session = "休市"
    elif hour_et < 9 or (hour_et >= 16):
        session = "盘后"
    elif 9 <= hour_et < 9.5:
        session = "盘前"
    else:
        session = "盘中"

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        price = info.last_price or 0
        prev_close = info.previous_close or 0
        currency = info.currency or "USD"
    except Exception as e:
        logger.warning(f"[FullAnalyzer] {ticker} 实时价格失败: {e}")
        price = 0
        prev_close = 0
        currency = "USD"

    change_pct = 0.0
    if prev_close and price:
        change_pct = round((price - prev_close) / prev_close * 100, 2)

    return Step1PriceResult(
        ticker=ticker,
        latest_price=price,
        currency=currency,
        session=session,
        quote_source="yfinance（延迟约15分钟）",
        quote_time=now.strftime("%Y-%m-%d %H:%M ET"),
        price_change_pct=change_pct,
        previous_close=prev_close,
    )


# ============================================================
# Step 2: 技术面分析（70%）
# ============================================================

def _run_technical_analysis(
    ticker: str,
    latest_price: float,
    fetcher,
) -> Step2TechResult:
    """
    执行多时间框架技术分析（含P1模块：衍生品、成交量分布、经典理论）。
    数据来源：yfinance 历史K线（免费真实数据）
    """
    from ai.multi_timeframe import analyze_multi_timeframe, format_mtf_result
    from ai.support_resistance import analyze_support_resistance, format_sr_result

    result = Step2TechResult(ticker=ticker)

    try:
        # 多时间框架分析
        mtf = analyze_multi_timeframe(ticker, latest_price, fetcher)
        result.mtf_summary = format_mtf_result(mtf)

        # 提取关键指标
        daily_ind = mtf.indicators.get("日线")
        if daily_ind:
            result.rsi = daily_ind.rsi
            result.adx = daily_ind.adx
            result.atr_pct = daily_ind.atr_percent
            result.macd_hist = daily_ind.macd_hist

        result.patterns = [p for p in mtf.patterns]
        result.divergences = mtf.divergences
        result.consensus_trend = mtf.consensus_trend
        result.consensus_confidence = mtf.consensus_confidence
        result.top_bull_signals = mtf.bullish_signals[:3]
        result.top_bear_signals = mtf.bearish_signals[:3]
    except Exception as e:
        logger.warning(f"[FullAnalyzer] MTF分析失败: {e}")
        result.missing_data.append("多时间框架分析（数据不足）")

    try:
        # 支撑阻力分析
        sr = analyze_support_resistance(ticker, latest_price, fetcher)
        result.sr_summary = format_sr_result(sr)
    except Exception as e:
        logger.warning(f"[FullAnalyzer] SR分析失败: {e}")
        result.missing_data.append("支撑阻力分析（数据不足）")

    # === P1: 衍生品分析 ===
    try:
        from ai.derivatives import analyze_derivatives, format_derivatives_result
        der = analyze_derivatives(ticker, fetcher)
        result.derivatives_summary = format_derivatives_result(der)
        if der.missing_data:
            result.missing_data.extend([f"衍生品:{m}" for m in der.missing_data])
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 衍生品分析失败: {e}")
        result.missing_data.append("衍生品分析（异常）")

    # === P1: 成交量分布分析 ===
    try:
        from ai.volume_profile import analyze_volume_profile, format_vp_result
        vp = analyze_volume_profile(ticker, latest_price, fetcher)
        result.vp_summary = format_vp_result(vp)
        if vp.missing_data:
            result.missing_data.extend([f"成交量分布:{m}" for m in vp.missing_data])
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 成交量分布分析失败: {e}")
        result.missing_data.append("成交量分布分析（异常）")

    # === P1: 经典理论分析 ===
    try:
        from ai.classical_theory import analyze_classical_theory, format_classical_result
        ct = analyze_classical_theory(ticker, latest_price, fetcher)
        result.classical_summary = format_classical_result(ct)
        if ct.missing_data:
            result.missing_data.extend([f"经典理论:{m}" for m in ct.missing_data])
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 经典理论分析失败: {e}")
        result.missing_data.append("经典理论分析（异常）")

    return result


# ============================================================
# Step 3: 宏观+情绪（30%）
# ============================================================

def _run_macro_sentiment_analysis(
    ticker: str,
    latest_price: float,
    fetcher,
) -> Step3MacroResult:
    """
    宏观基本面+情绪分析。
    数据来源：
      - macro_scanner.py（免费指标：VIX/Equity/Breadth/Credit/DXY/GLD）
      - 搜索（恐惧贪婪指数 CNN Fear & Greed）
      - 搜索（AAII情绪调查）
    缺失付费数据：SpotGamma GEX、Barchart Max Pain、UnusualWhales
    """
    from ai.macro_scanner import MacroScanner

    result = Step3MacroResult(ticker=ticker)
    missing = []

    try:
        scanner = MacroScanner()
        macro_result = scanner.scan()
        # MacroScanResult 是 dataclass，不是 dict，需要访问 environment 属性
        env = macro_result.environment
        
        # 取环境评分
        if env and latest_price > 0:
            result.macro_score = round(float(env.environment_score), 1)
        else:
            result.macro_score = 50.0

        # 各模块评分（从 module_scores 获取）
        module_scores = env.module_scores if env else {}
        result.equity_score = round(float(module_scores.get("equity", 50)), 1)
        result.vix_score = round(float(module_scores.get("vix", 50)), 1)
        result.breadth_score = round(float(module_scores.get("breadth", 50)), 1)
        result.credit_score = round(float(module_scores.get("credit", 50)), 1)
        result.safe_haven_score = round(float(module_scores.get("safe_haven", 50)), 1)

        # 解读
        if result.macro_score >= 65:
            result.score_interpretation = "偏多"
        elif result.macro_score <= 35:
            result.score_interpretation = "偏空"
        else:
            result.score_interpretation = "中性"

        result.outlook = result.score_interpretation
    except Exception as e:
        logger.warning(f"[FullAnalyzer] MacroScanner失败: {e}")
        result.macro_score = 50.0
        result.missing_data.append("宏观环境评分（扫描失败，使用中性50）")
        missing.append("MacroScanner")

    # 情绪数据 — 通过搜索获取
    try:
        fear_greed_data = _fetch_fear_greed_index()
        if fear_greed_data:
            result.fear_greed = fear_greed_data.get("label", "中性")
            result.fear_greed_value = fear_greed_data.get("value")
        else:
            missing.append("CNN Fear & Greed指数（搜索无结果）")
    except Exception:
        missing.append("CNN Fear & Greed指数（获取失败）")

    # === P1: 详细情绪分析 ===
    try:
        from ai.sentiment import analyze_market_sentiment, format_sentiment_result
        sent = analyze_market_sentiment(ticker, fetcher)
        result.sentiment_summary = format_sentiment_result(sent)
        if sent.missing_data:
            missing.extend([f"情绪:{m}" for m in sent.missing_data])
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 情绪分析失败: {e}")
        missing.append("情绪分析（异常）")

    # === P2: 机构资金流向 ===
    try:
        from ai.institutional_flows import analyze_institutional_flows, format_flows_result
        flows = analyze_institutional_flows(ticker, latest_price, fetcher)
        result.flows_summary = format_flows_result(flows)
        if flows.missing_data:
            missing.extend([f"资金流:{m}" for m in flows.missing_data])
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 资金流向分析失败: {e}")
        missing.append("资金流向分析（异常）")

    # === P2: 宏观政策分析 ===
    try:
        from ai.macro_policy import analyze_macro_policy, format_macro_policy_result
        policy = analyze_macro_policy(ticker, latest_price, fetcher)
        result.policy_summary = format_macro_policy_result(policy)
        if policy.missing_data:
            missing.extend([f"政策:{m}" for m in policy.missing_data])
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 宏观政策分析失败: {e}")
        missing.append("宏观政策分析（异常）")

    # === P2: 机构异动检测 ===
    try:
        from ai.institutional_detector import analyze_institutional_activity, format_institutional_result
        inst = analyze_institutional_activity(ticker, latest_price, fetcher)
        result.institutional_summary = format_institutional_result(inst)
        if inst.missing_data:
            missing.extend([f"机构异动:{m}" for m in inst.missing_data])
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 机构异动检测失败: {e}")
        missing.append("机构异动检测（异常）")

    result.missing_data = missing
    return result


def _fetch_fear_greed_index() -> Optional[dict]:
    """
    获取CNN Fear & Greed指数。
    数据来源：网络搜索
    注意：CNN网站本身需要付费，但搜索引擎可能返回近期数据
    如无法获取，返回None（标注缺失）
    """
    # 搜索逻辑由调用方通过 search_skill 执行
    # 此处仅提供数据获取提示，返回None表示需外部搜索
    return None  # 实际由search_skill返回数据后填入


# ============================================================
# Step 4: 策略生成
# ============================================================

def _generate_strategies(
    ticker: str,
    latest_price: float,
    step2: Step2TechResult,
    step3: Step3MacroResult,
) -> Step4StrategyResult:
    """基于Step2+Step3生成交易策略"""
    from ai.strategy_advisor import generate_trading_strategies

    result = Step4StrategyResult(ticker=ticker)

    try:
        # 从Step2构建MTF结果
        class _MockMTFResult:
            def __init__(self, s2):
                self.ticker = s2.ticker
                self.latest_price = latest_price
                self.consensus_trend = s2.consensus_trend
                self.consensus_confidence = s2.consensus_confidence
                self.bullish_signals = s2.top_bull_signals
                self.bearish_signals = s2.top_bear_signals
                self.divergences = s2.divergences
                self.indicators = {}
                class _Ind:
                    def __init__(self, s):
                        self.rsi = s.rsi
                        self.macd_hist = s.macd_hist
                        self.adx = s.adx
                        self.atr = None
                        self.atr_percent = s.atr_pct
                self.indicators["日线"] = _Ind(s2)

        class _MockSRResult:
            def __init__(self, s2):
                self.ticker = s2.ticker
                self.latest_price = latest_price
                self.nearest_support = None
                self.nearest_resistance = None
                sd = s2.sr_summary
                if sd:
                    ns = sd.get("最近支撑")
                    nr = sd.get("最近阻力")
                    if ns and ns != "无数据":
                        self.nearest_support = float(str(ns).replace(",", ""))
                    if nr and nr != "无数据":
                        self.nearest_resistance = float(str(nr).replace(",", ""))

        mtf_mock = _MockMTFResult(step2)
        sr_mock = _MockSRResult(step2)
        macro_score = step3.macro_score

        adv_result = generate_trading_strategies(
            ticker, latest_price, mtf_mock, sr_mock, macro_score
        )

        from ai.strategy_advisor import format_strategy_result
        fmt = format_strategy_result(adv_result)
        result.short_term_prob = fmt["5天概率判断"]
        result.medium_term_prob = fmt["1个月概率判断"]
        result.short_strategy = fmt["5天交易策略"]
        result.medium_strategy = fmt["1个月交易策略"]
        result.composite_signal = fmt["综合信号"]
        result.composite_confidence = adv_result.composite_confidence

    except Exception as e:
        logger.warning(f"[FullAnalyzer] 策略生成失败: {e}")
        result.short_term_prob = {"error": str(e)}
        result.medium_term_prob = {"error": str(e)}
        result.composite_signal = "生成失败"
        result.composite_confidence = 0.0

    return result


# ============================================================
# Step 5: 总结
# ============================================================

def _generate_summary(
    ticker: str,
    step2: Step2TechResult,
    step3: Step3MacroResult,
    step4: Step4StrategyResult,
) -> Step5Summary:
    """生成步骤5总结"""
    s5 = Step5Summary(ticker=ticker)

    # 技术面
    s5.tech_bull = " | ".join(step2.top_bull_signals[:2]) if step2.top_bull_signals else "无明确看多信号"
    s5.tech_bear = " | ".join(step2.top_bear_signals[:2]) if step2.top_bear_signals else "无明确看空信号"

    # 宏观
    if step3.macro_score >= 65:
        s5.macro_bull = f"宏观偏多（评分{step3.macro_score:.0f}）"
    elif step3.macro_score <= 35:
        s5.macro_bear = f"宏观偏空（评分{step3.macro_score:.0f}）"
    else:
        s5.macro_bull = "宏观中性"
        s5.macro_bear = "宏观中性"

    # 机会
    s5.key_opportunities = [
        f"技术面: {s5.tech_bull}",
        f"宏观: {s5.macro_bull}",
        f"近支撑: {step2.sr_summary.get('最近支撑', 'N/A')}",
    ]

    # 风险
    risk_list = []
    if step2.divergences:
        risk_list.append(f"存在背离: {step2.divergences[0]}")
    if step3.macro_score < 40:
        risk_list.append(f"宏观逆风（评分{step3.macro_score:.0f}<40）")
    if step2.rsi and step2.rsi > 70:
        risk_list.append(f"RSI超买({step2.rsi:.0f})")
    if step2.rsi and step2.rsi < 30:
        risk_list.append(f"RSI超卖({step2.rsi:.0f})")
    if not risk_list:
        risk_list = ["无明显风险提示"]
    s5.key_risks = risk_list

    # 最终建议
    s5.composite_signal = step4.composite_signal
    s5.composite_confidence = step4.composite_confidence

    bull_5d = float(str(step4.short_term_prob.get("看多概率", "0%")).replace("%", ""))
    bull_1m = float(str(step4.medium_term_prob.get("看多概率", "0%")).replace("%", ""))

    if bull_5d > 60 and bull_1m > 55:
        s5.final_recommendation = f"综合信号看多（置信度{step4.composite_confidence:.0f}%）"
    elif bull_5d < 40 and bull_1m < 45:
        s5.final_recommendation = f"综合信号看空（置信度{step4.composite_confidence:.0f}%）"
    else:
        s5.final_recommendation = f"信号中性，建议观望（置信度{step4.composite_confidence:.0f}%）"

    return s5


# ============================================================
# P5: 组合优化
# ============================================================

def _run_portfolio_analysis(
    ticker: str,
    latest_price: float,
    fetcher,
) -> dict:
    """执行组合优化分析（P5）"""
    try:
        from ai.portfolio_optimizer import run_portfolio_analysis
        # 使用SPY+GLD+QQQ+IWM+TLT作为示例组合
        tickers = [ticker, "GLD", "QQQ", "IWM", "TLT"] if ticker != "SPY" else ["SPY", "GLD", "QQQ", "IWM", "TLT"]
        return run_portfolio_analysis(tickers, fetcher)
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 组合优化失败: {e}")
        return {"error": str(e)}


# ============================================================
# P5: 事件驱动分析
# ============================================================

def _run_event_analysis(
    ticker: str,
    fetcher,
) -> dict:
    """执行事件驱动分析（P5）"""
    try:
        from ai.event_analyzer import analyze_events, format_event_result
        result = analyze_events(ticker, fetcher)
        return format_event_result(result)
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 事件分析失败: {e}")
        return {"error": str(e)}


# ============================================================
# P5: 期权链分析
# ============================================================

def _run_options_analysis(
    ticker: str,
    latest_price: float,
    fetcher,
) -> dict:
    """执行期权链分析（P5）"""
    try:
        from ai.options_analyzer import analyze_options_chain, format_options_result
        result = analyze_options_chain(ticker, latest_price, fetcher)
        return format_options_result(result)
    except Exception as e:
        logger.warning(f"[FullAnalyzer] 期权分析失败: {e}")
        return {"error": str(e)}


# ============================================================
# P5: NLP情绪分析
# ============================================================

def _run_nlp_analysis(
    ticker: str,
    fetcher,
) -> dict:
    """执行NLP情绪分析（P5）"""
    try:
        from ai.nlp_sentiment import analyze_market_news_sentiment
        result = analyze_market_news_sentiment([ticker], fetcher)
        return result.get(ticker, {"error": "无数据"})
    except Exception as e:
        logger.warning(f"[FullAnalyzer] NLP分析失败: {e}")
        return {"error": str(e)}


# ============================================================
# 主入口
# ============================================================

def run_full_analysis(
    tickers: list[str],
    fetcher,
    search_func=None,  # 可选：搜索函数，用于获取恐惧贪婪指数等
    current_time_str: str = "",
    use_async: bool = False,  # 是否使用异步并行
    include_p5: bool = True,  # 是否包含P5模块
) -> dict[str, FullAnalysisResult]:
    """
    对多个标的执行完整AI分析（5步骤 + P3 + P4 + P5）。

    参数:
        tickers: 标的列表，如 ["SPY", "GLD"]
        fetcher: YFinanceFetcher 实例
        search_func: 可选的搜索函数（用于获取CNN Fear & Greed等实时情绪数据）
        current_time_str: 当前时间字符串
        use_async: 是否使用异步并行加速（多标的时推荐）
        include_p5: 是否包含P5模块（组合优化/事件/期权/NLP）

    返回:
        dict[ticker -> FullAnalysisResult]

    数据完整性说明：
        - 每个step均使用真实数据
        - 无法获取的数据标注"缺少[数据源]"，不影响其他分析
        - 最终 data_completeness 反映实际获取数据的比例
    """
    import time
    start_time = time.time()
    
    time_str = current_time_str or datetime.now().strftime("%Y-%m-%d %H:%M ET")
    results = {}

    for ticker in tickers:
        logger.info(f"[FullAnalyzer] 开始分析 {ticker}")
        fr = FullAnalysisResult(ticker=ticker, tickers=tickers, analysis_time_et=time_str)

        # Step 1: 实时价格
        fr.step1_price = _fetch_realtime_price(ticker, fetcher)
        latest_price = fr.step1_price.latest_price
        if latest_price == 0:
            # 尝试用前一天收盘价
            try:
                df = fetcher.download_history(ticker, period="5d", interval="1d")
                if not df.empty:
                    latest_price = float(df["Close"].iloc[-1])
                    fr.step1_price.latest_price = latest_price
            except Exception:
                pass
        logger.info(f"[FullAnalyzer] {ticker} 最新价格: {latest_price}")

        if latest_price == 0:
            logger.warning(f"[FullAnalyzer] {ticker} 无法获取价格，跳过分析")
            results[ticker] = fr
            continue

        # Step 2: 技术面
        fr.step2_tech = _run_technical_analysis(ticker, latest_price, fetcher)

        # Step 3: 宏观+情绪
        fr.step3_macro = _run_macro_sentiment_analysis(ticker, latest_price, fetcher)

        # Step 4: 策略
        fr.step4_strategy = _generate_strategies(
            ticker, latest_price, fr.step2_tech, fr.step3_macro
        )

        # Step 5: 总结
        fr.step5_summary = _generate_summary(
            ticker, fr.step2_tech, fr.step3_macro, fr.step4_strategy
        )

        # === P3: 动态权重 ===
        try:
            from ai.dynamic_weights import calculate_dynamic_weights, format_dynamic_weights_result
            dw = calculate_dynamic_weights(ticker, latest_price, fetcher)
            fr.dynamic_weights = format_dynamic_weights_result(dw)
        except Exception as e:
            logger.warning(f"[FullAnalyzer] 动态权重计算失败: {e}")

        # === P3: 历史模式匹配 ===
        try:
            from ai.pattern_matcher import analyze_pattern_match, format_pattern_match_result
            pm = analyze_pattern_match(ticker, latest_price, fetcher, lookback_days=20, top_n=10)
            fr.pattern_match = format_pattern_match_result(pm)
        except Exception as e:
            logger.warning(f"[FullAnalyzer] 模式匹配失败: {e}")

        # === P3: 情景分析 ===
        try:
            from ai.scenario_analysis import analyze_scenarios, format_scenario_result
            # 提取支撑阻力位
            sr_support = 0.0
            sr_resistance = 0.0
            if fr.step2_tech and fr.step2_tech.sr_summary:
                ns = fr.step2_tech.sr_summary.get("最近支撑")
                nr = fr.step2_tech.sr_summary.get("最近阻力")
                if ns and ns != "无数据":
                    try: sr_support = float(str(ns).replace(",", ""))
                    except: pass
                if nr and nr != "无数据":
                    try: sr_resistance = float(str(nr).replace(",", ""))
                    except: pass

            # 提取模式匹配胜率
            pm_winrate = 50.0
            if fr.pattern_match and "回测统计" in fr.pattern_match:
                try:
                    wr_str = fr.pattern_match["回测统计"].get("20日胜率", "50%")
                    pm_winrate = float(wr_str.replace("%", ""))
                except: pass

            # 提取VIX
            vix = 20.0
            if fr.step3_macro and fr.step3_macro.vix_score:
                vix = fr.step3_macro.vix_score

            sc = analyze_scenarios(
                ticker=ticker,
                current_price=latest_price,
                tech_score=fr.step2_tech.consensus_confidence if fr.step2_tech else 50.0,
                macro_score=fr.step3_macro.macro_score if fr.step3_macro else 50.0,
                vix=vix,
                trend=fr.step2_tech.consensus_trend if fr.step2_tech else "neutral",
                atr_pct=fr.step2_tech.atr_pct if fr.step2_tech else 1.0,
                sr_support=sr_support,
                sr_resistance=sr_resistance,
                pattern_match_winrate=pm_winrate,
            )
            fr.scenarios = format_scenario_result(sc)
        except Exception as e:
            logger.warning(f"[FullAnalyzer] 情景分析失败: {e}")

        # === P3: 相关性矩阵 ===
        try:
            from ai.correlation_matrix import analyze_correlations, format_correlation_result
            corr = analyze_correlations(ticker, fetcher)
            fr.correlations = format_correlation_result(corr)
        except Exception as e:
            logger.warning(f"[FullAnalyzer] 相关性分析失败: {e}")

        # === P4: 策略回测 ===
        try:
            from ai.backtest_engine import backtest_strategy, format_backtest_result
            bt = backtest_strategy(ticker, fetcher, strategy_type="combined", holding_period=5)
            fr.backtest = format_backtest_result(bt)
        except Exception as e:
            logger.warning(f"[FullAnalyzer] 策略回测失败: {e}")

        # === P4: 蒙特卡洛模拟 ===
        try:
            from ai.monte_carlo import run_monte_carlo, format_monte_carlo_result
            # 提取目标价/止损价
            target = None
            stop = None
            if fr.step4_strategy and fr.step4_strategy.short_strategy:
                try:
                    tp_str = str(fr.step4_strategy.short_strategy.get("止盈", ""))
                    sl_str = str(fr.step4_strategy.short_strategy.get("止损", ""))
                    target = float(tp_str) if tp_str else None
                    stop = float(sl_str) if sl_str else None
                except: pass

            mc = run_monte_carlo(
                ticker=ticker,
                current_price=latest_price,
                fetcher=fetcher,
                simulation_days=20,
                num_simulations=1000,
                target_price=target,
                stop_price=stop,
            )
            fr.monte_carlo = format_monte_carlo_result(mc)
        except Exception as e:
            logger.warning(f"[FullAnalyzer] 蒙特卡洛模拟失败: {e}")

        # === P4: ML预测 ===
        try:
            from ai.ml_predictor import predict_with_ml, format_ml_result
            ml = predict_with_ml(ticker, latest_price, fetcher, prediction_horizon=5, model_type="random_forest")
            fr.ml_prediction = format_ml_result(ml)
        except Exception as e:
            logger.warning(f"[FullAnalyzer] ML预测失败: {e}")

        # === P4: 预警检查 ===
        try:
            from ai.alert_system import check_alerts_for_ticker
            alerts = check_alerts_for_ticker(ticker, fetcher)
            fr.alerts = alerts
        except Exception as e:
            logger.warning(f"[FullAnalyzer] 预警检查失败: {e}")

        # === P5: 组合优化 ===
        if include_p5:
            try:
                fr.portfolio = _run_portfolio_analysis(ticker, latest_price, fetcher)
            except Exception as e:
                logger.warning(f"[FullAnalyzer] P5组合优化失败: {e}")

        # === P5: 事件驱动分析 ===
        if include_p5:
            try:
                fr.events = _run_event_analysis(ticker, fetcher)
            except Exception as e:
                logger.warning(f"[FullAnalyzer] P5事件分析失败: {e}")

        # === P5: 期权链分析 ===
        if include_p5:
            try:
                fr.options = _run_options_analysis(ticker, latest_price, fetcher)
            except Exception as e:
                logger.warning(f"[FullAnalyzer] P5期权分析失败: {e}")

        # === P5: NLP情绪分析 ===
        if include_p5:
            try:
                fr.nlp_sentiment = _run_nlp_analysis(ticker, fetcher)
            except Exception as e:
                logger.warning(f"[FullAnalyzer] P5 NLP分析失败: {e}")

        # 数据完整性评估（更新总项数以反映P5模块）
        total_items = 30  # P0=5, P1=4, P2=3, P3=4, P4=4, P5=4 + 基础6项
        missing = len(fr.step2_tech.missing_data) + len(fr.step3_macro.missing_data)
        fr.data_completeness = round(max(0, (total_items - missing) / total_items * 100), 0)
        fr.missing_data_sources = fr.step2_tech.missing_data + fr.step3_macro.missing_data

        # 分析耗时
        fr.analysis_duration_sec = round(time.time() - start_time, 2)

        results[ticker] = fr
        logger.info(f"[FullAnalyzer] {ticker} 分析完成，完整性: {fr.data_completeness:.0f}%, 耗时: {fr.analysis_duration_sec:.1f}s")

    return results


# ============================================================
# 格式化输出（供WebUI）
# ============================================================

def format_full_analysis_for_ui(results: dict[str, FullAnalysisResult]) -> dict:
    """将分析结果格式化为WebUI展示格式"""

    def _fmt_price(s1):
        if s1 is None:
            return {"error": "价格获取失败"}
        return {
            "标的": s1.ticker,
            "最新价格": s1.latest_price,
            "涨跌": f"{'+' if s1.price_change_pct > 0 else ''}{s1.price_change_pct:.2f}%",
            "昨收": s1.previous_close,
            "交易时段": s1.session,
            "数据来源": s1.quote_source,
            "更新时间": s1.quote_time,
        }

    def _fmt_tech(s2):
        if s2 is None:
            return {"error": "技术分析失败"}
        bull = s2.top_bull_signals
        bear = s2.top_bear_signals
        div = s2.divergences
        return {
            "标的": s2.ticker,
            "综合趋势": s2.consensus_trend,
            "趋势置信度": f"{s2.consensus_confidence:.0%}",
            "RSI": f"{s2.rsi:.1f}" if s2.rsi else "N/A",
            "ADX": f"{s2.adx:.1f}" if s2.adx else "N/A",
            "ATR%": f"{s2.atr_pct:.2f}%" if s2.atr_pct else "N/A",
            "MACD柱": f"{'+' if s2.macd_hist and s2.macd_hist > 0 else ''}{s2.macd_hist:.4f}" if s2.macd_hist else "N/A",
            "K线形态": [f"{p.tf_label}: {p.pattern_name}" for p in s2.patterns[:3]],
            "背离": div[:2] if div else [],
            "看多信号": bull[:3],
            "看空信号": bear[:3],
            "支撑阻力": s2.sr_summary,
            "衍生品": s2.derivatives_summary,
            "成交量分布": s2.vp_summary,
            "经典理论": s2.classical_summary,
            "缺少数据": s2.missing_data,
        }

    def _fmt_macro(s3):
        if s3 is None:
            return {"error": "宏观分析失败"}
        return {
            "标的": s3.ticker,
            "宏观评分": f"{s3.macro_score:.0f}/100",
            "解读": s3.score_interpretation,
            "权益评分": s3.equity_score,
            "VIX评分": s3.vix_score,
            "广度评分": s3.breadth_score,
            "信贷评分": s3.credit_score,
            "避险评分": s3.safe_haven_score,
            "恐惧贪婪": f"{s3.fear_greed}({s3.fear_greed_value})" if s3.fear_greed else "缺少",
            "情绪详情": s3.sentiment_summary,
            "机构资金流": s3.flows_summary,
            "宏观政策": s3.policy_summary,
            "机构异动": s3.institutional_summary,
            "缺少数据": s3.missing_data,
        }

    def _fmt_all(results_dict):
        out = {}
        for t, r in results_dict.items():
            out[t] = {
                "Step1_价格": _fmt_price(r.step1_price),
                "Step2_技术面": _fmt_tech(r.step2_tech),
                "Step3_宏观情绪": _fmt_macro(r.step3_macro),
                "Step4_5天策略": r.step4_strategy.short_strategy if r.step4_strategy else {},
                "Step4_1月策略": r.step4_strategy.medium_strategy if r.step4_strategy else {},
                "Step5_总结": {
                    "综合信号": r.step5_summary.composite_signal if r.step5_summary else "",
                    "置信度": f"{r.step5_summary.composite_confidence:.0f}%" if r.step5_summary else "N/A",
                    "技术面机会": r.step5_summary.key_opportunities if r.step5_summary else [],
                    "风险提示": r.step5_summary.key_risks if r.step5_summary else [],
                    "最终建议": r.step5_summary.final_recommendation if r.step5_summary else "",
                },
                "P3_动态权重": r.dynamic_weights,
                "P3_历史模式匹配": r.pattern_match,
                "P3_情景分析": r.scenarios,
                "P3_相关性矩阵": r.correlations,
                "P4_策略回测": r.backtest,
                "P4_蒙特卡洛模拟": r.monte_carlo,
                "P4_ML预测": r.ml_prediction,
                "P4_预警系统": r.alerts,
                "P5_组合优化": r.portfolio,
                "P5_事件分析": r.events,
                "P5_期权分析": r.options,
                "P5_NLP情绪": r.nlp_sentiment,
                "数据完整性": f"{r.data_completeness:.0f}%",
                "分析耗时": f"{r.analysis_duration_sec:.1f}s",
                "缺少数据": r.missing_data_sources,
            }
        return out

    return {
        "分析时间": list(results.values())[0].analysis_time_et if results else "",
        "标的列表": list(results.keys()),
        "各标的分析": _fmt_all(results),
    }
