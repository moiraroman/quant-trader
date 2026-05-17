# ============================================================
# ai/market_breadth.py — 市场广度分析模块 (独立版)
# 分析涨跌比、新高新低、发散检测、A/D线等
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================
# 常量定义
# ============================================================

# 市场广度指标ETF
BREADTH_TICKERS = {
    "RSP": "S&P 500 Equal Weight",  # 等权vs市值
    "VWO": "Emerging Markets",
    "VEA": "Developed Markets Ex-US",
    "^GSPC": "S&P 500 Index",
}

# 波动率相关
VOLATILITY_TICKERS = {
    "^VIX": "CBOE Volatility Index",
    "^VIX3M": "VIX 3-Month",
    "^VIX1Y": "VIX 1-Year",
}

# 市场情绪指标
SENTIMENT_TICKERS = {
    "CNNE": "Cannabis ETF (prox for speculation)",
    "ARKK": "ARK Innovation (speculation proxy)",
}


# ============================================================
# 数据类
# ============================================================

@dataclass
class BreadthStep:
    """分析步骤记录"""
    step_name: str
    input_data: str
    calculation: str
    result: str
    reasoning: str


@dataclass
class AdvanceDeclineAnalysis:
    """涨跌分析 (使用代理指标)"""
    ticker: str
    name: str
    adv_dec_ratio: float  # 代理: RSP/SPY ratio
    trend: str  # improvement / deterioration / neutral
    breadth_score: float  # 0-100
    analysis_steps: List[BreadthStep] = field(default_factory=list)


@dataclass
class NewHighLowAnalysis:
    """新高新低分析 (使用代理指标)"""
    ticker: str
    name: str
    new_highs_proxy: float  # 代理: 52-week high proximity
    new_lows_proxy: float  # 代理: 52-week low proximity
    hh_ll_ratio: float  # HH/(HH+LL)
    signal: str  # strong_bull / bull / neutral / bear / strong_bear
    analysis_steps: List[BreadthStep] = field(default_factory=list)


@dataclass
class DivergenceAnalysis:
    """发散检测"""
    index_ticker: str
    breadth_ticker: str
    correlation: float  # 相关性
    divergence_detected: bool
    divergence_type: str  # bullish / bearish / none
    severity: float  # 0-1
    analysis_steps: List[BreadthStep] = field(default_factory=list)


@dataclass
class ADLineAnalysis:
    """A/D线分析 (代理)"""
    ticker: str
    ad_line_value: float  # 代理: 累计RSP-SPY差值
    ad_line_trend: str  # rising / falling / flat
    divergence_with_index: bool
    analysis_steps: List[BreadthStep] = field(default_factory=list)


@dataclass
class BreadthRegressionSignal:
    """广度回归信号"""
    signal_type: str  # overbought / oversold / normal
    strength: float  # 0-100
    expected_return_5d: float  # 预期5日收益率
    confidence: str  # high / medium / low
    analysis_steps: List[BreadthStep] = field(default_factory=list)


@dataclass
class MarketBreadthResult:
    """市场广度分析结果"""
    timestamp: datetime
    advance_decline: List[AdvanceDeclineAnalysis]
    new_high_low: List[NewHighLowAnalysis]
    divergences: List[DivergenceAnalysis]
    ad_line: Optional[ADLineAnalysis] = None
    regression_signal: Optional[BreadthRegressionSignal] = None
    overall_breadth_score: int = 5  # 1-10
    breadth_regime: str = "neutral"  # strong_bull / bull / neutral / bear / strong_bear
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    analysis_steps: List[BreadthStep] = field(default_factory=list)


# ============================================================
# 核心分析器
# ============================================================

class MarketBreadthAnalyzer:
    """市场广度分析器 (独立版)"""

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.cache = {}

    def analyze(self) -> MarketBreadthResult:
        """执行完整的市场广度分析"""
        steps = []

        # Step 1: 获取数据
        data = self._fetch_data()
        steps.append(BreadthStep(
            step_name="获取数据",
            input_data=f"Tickers: {list(BREADTH_TICKERS.keys())}",
            calculation="yf.download()",
            result=f"获取到 {len(data)} 个数据集",
            reasoning="获取等权指数、波动率等广度相关指标"
        ))

        # Step 2: 分析涨跌比 (使用RSP/SPY代理)
        adv_dec = self._analyze_advance_decline(data)
        steps.append(BreadthStep(
            step_name="分析涨跌比",
            input_data="RSP, SPY data",
            calculation="RSP/SPY ratio, rolling correlation",
            result=f"分析了 {len(adv_dec)} 个涨跌指标",
            reasoning="使用RSP/SPY比值作为涨跌比代理"
        ))

        # Step 3: 分析新高新低 (代理)
        new_hl = self._analyze_new_high_low(data)
        steps.append(BreadthStep(
            step_name="分析新高新低",
            input_data="SPY 52-week high/low data",
            calculation="proximity to 52w H/L",
            result=f"分析了 {len(new_hl)} 个新高新低指标",
            reasoning="使用距离52周高低点的距离作为代理"
        ))

        # Step 4: 检测发散
        divergences = self._detect_divergences(data)
        steps.append(BreadthStep(
            step_name="检测发散",
            input_data="SPY vs RSP correlation",
            calculation="rolling correlation, trend comparison",
            result=f"检测到 {len(divergences)} 个潜在发散",
            reasoning="等权指数与市值指数发散暗示广度问题"
        ))

        # Step 5: A/D线分析 (代理)
        ad_line = self._analyze_ad_line(data)
        steps.append(BreadthStep(
            step_name="A/D线分析",
            input_data="RSP-SPY cumulative",
            calculation="cumulative sum of ratio deviation",
            result=f"A/D线趋势: {ad_line.ad_line_trend if ad_line else 'N/A'}",
            reasoning="A/D线上升表示广度改善"
        ))

        # Step 6: 广度回归信号
        regression = self._generate_regression_signal(adv_dec, new_hl, divergences)
        steps.append(BreadthStep(
            step_name="生成广度回归信号",
            input_data="breadth_score, divergences",
            calculation="historical percentile, mean reversion",
            result=f"回归信号: {regression.signal_type if regression else 'N/A'}",
            reasoning="极端广度通常伴随均值回归"
        ))

        # Step 7: 计算综合广度评分
        overall_score = self._calculate_overall_score(adv_dec, new_hl, divergences, ad_line)
        steps.append(BreadthStep(
            step_name="计算综合广度评分",
            input_data="adv_dec_score, new_hl_score, divergence_severity",
            calculation="weighted average",
            result=f"综合评分: {overall_score}/10",
            reasoning="加权平均各维度广度得分"
        ))

        # Step 8: 判断广度regime
        breadth_regime = self._classify_regime(overall_score, new_hl, divergences)
        steps.append(BreadthStep(
            step_name="判断广度regime",
            input_data="overall_score, signals",
            calculation="score threshold + signal confirmation",
            result=f"广度环境: {breadth_regime}",
            reasoning="综合评分和信号确定市场环境"
        ))

        # Step 9: 生成警告
        warnings = self._generate_warnings(divergences, ad_line, new_hl)
        steps.append(BreadthStep(
            step_name="生成风险警告",
            input_data="divergences, ad_line_trend",
            calculation="check severity and trends",
            result=f"发现 {len(warnings)} 个警告",
            reasoning="发散和趋势恶化是风险提示"
        ))

        # Step 10: 生成摘要
        summary = self._generate_summary(
            adv_dec, new_hl, divergences, ad_line, regression, overall_score, breadth_regime
        )
        steps.append(BreadthStep(
            step_name="生成分析摘要",
            input_data="all analysis results",
            calculation="summarize key findings",
            result=summary[:100] + "...",
            reasoning="汇总所有分析结果"
        ))

        return MarketBreadthResult(
            timestamp=datetime.now(),
            advance_decline=adv_dec,
            new_high_low=new_hl,
            divergences=divergences,
            ad_line=ad_line,
            regression_signal=regression,
            overall_breadth_score=overall_score,
            breadth_regime=breadth_regime,
            summary=summary,
            warnings=warnings,
            analysis_steps=steps
        )

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """获取市场广度相关数据"""
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days * 2)

        all_tickers = {}
        all_tickers.update(BREADTH_TICKERS)
        all_tickers.update(VOLATILITY_TICKERS)
        all_tickers.update(SENTIMENT_TICKERS)

        for ticker in all_tickers.keys():
            try:
                if ticker in self.cache:
                    data[ticker] = self.cache[ticker]
                    continue

                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    data[ticker] = df
                    self.cache[ticker] = df
            except Exception as e:
                logger.warning(f"获取 {ticker} 数据失败: {e}")

        return data

    def _analyze_advance_decline(self, data: Dict[str, pd.DataFrame]) -> List[AdvanceDeclineAnalysis]:
        """分析涨跌比 (使用RSP/SPY代理)"""
        results = []

        if "RSP" in data and "^GSPC" in data:
            rsp = data["RSP"]
            spy = data["^GSPC"]

            # 计算RSP/SPY比值 (等权/市值)
            rsp_returns = rsp['Close'].pct_change(20).dropna()  # 20日收益率
            spy_returns = spy['Close'].pct_change(20).dropna()

            # 对齐数据
            common_idx = rsp_returns.index.intersection(spy_returns.index)
            if len(common_idx) > 0:
                rsp_aligned = rsp_returns[common_idx]
                spy_aligned = spy_returns[common_idx]

                # 计算相关系数
                correlation = rsp_aligned.corr(spy_aligned)

                # 计算比值
                ratio = (rsp['Close'].iloc[-1] / spy['Close'].iloc[-1]) if len(rsp) > 0 else 1.0

                # 判断趋势
                if len(rsp) > 20:
                    ratio_20d = (rsp['Close'].iloc[-1] / spy['Close'].iloc[-1]) / \
                                (rsp['Close'].iloc[-20] / spy['Close'].iloc[-20]) - 1
                    if ratio_20d > 0.01:
                        trend = "improvement"
                    elif ratio_20d < -0.01:
                        trend = "deterioration"
                    else:
                        trend = "neutral"
                else:
                    trend = "neutral"
                    ratio_20d = 0.0

                # 计算广度评分 (correlation高 = 广度好)
                breadth_score = max(0, min(100, correlation * 100))

                steps = [
                    BreadthStep(
                        step_name="计算RSP/SPY相关性",
                        input_data=f"RSP data: {len(rsp)} rows, SPY data: {len(spy)} rows",
                        calculation=f"20日收益率相关系数",
                        result=f"Correlation: {correlation:.3f}",
                        reasoning="相关性高表示等权与市值同步，广度健康"
                    ),
                    BreadthStep(
                        step_name="计算RSP/SPY比值",
                        input_data=f"Current RSP: {rsp['Close'].iloc[-1]:.2f}, SPY: {spy['Close'].iloc[-1]:.2f}",
                        calculation="RSP/SPY ratio",
                        result=f"Ratio: {ratio:.4f}, 20日变化: {ratio_20d:+.2%}",
                        reasoning="比值上升表示等权表现优于市值，广度改善"
                    )
                ]

                results.append(AdvanceDeclineAnalysis(
                    ticker="RSP/SPY",
                    name="Equal Weight vs Market Cap",
                    adv_dec_ratio=ratio,
                    trend=trend,
                    breadth_score=breadth_score,
                    analysis_steps=steps
                ))

        return results

    def _analyze_new_high_low(self, data: Dict[str, pd.DataFrame]) -> List[NewHighLowAnalysis]:
        """分析新高新低 (使用52周高低点代理)"""
        results = []

        if "^GSPC" in data:
            spy = data["^GSPC"]
            current = spy['Close'].iloc[-1]
            high_52w = spy['Close'].rolling(252).max().iloc[-1]
            low_52w = spy['Close'].rolling(252).min().iloc[-1]

            # 计算距离高低点的百分比
            dist_to_high = (high_52w - current) / high_52w
            dist_to_low = (current - low_52w) / low_52w

            # 代理指标
            new_highs_proxy = 1 - dist_to_high  # 越接近高点，值越大
            new_lows_proxy = dist_to_low  # 越接近低点，值越大

            # HH/(HH+LL) 比值
            hh_ll_ratio = new_highs_proxy / (new_highs_proxy + new_lows_proxy + 1e-6)

            # 判断信号
            if hh_ll_ratio > 0.7:
                signal = "strong_bull"
            elif hh_ll_ratio > 0.5:
                signal = "bull"
            elif hh_ll_ratio > 0.3:
                signal = "neutral"
            elif hh_ll_ratio > 0.1:
                signal = "bear"
            else:
                signal = "strong_bear"

            steps = [
                BreadthStep(
                    step_name="计算52周高低点距离",
                    input_data=f"Current: {current:.2f}, 52w High: {high_52w:.2f}, 52w Low: {low_52w:.2f}",
                    calculation="distance to high/low",
                    result=f"距离高点: {dist_to_high:.2%}, 距离低点: {dist_to_low:.2%}",
                    reasoning="接近高点表示强势，接近低点表示弱势"
                ),
                BreadthStep(
                    step_name="计算HH/LL比值",
                    input_data=f"new_highs_proxy: {new_highs_proxy:.3f}, new_lows_proxy: {new_lows_proxy:.3f}",
                    calculation="HH/(HH+LL)",
                    result=f"HH/LL Ratio: {hh_ll_ratio:.3f}",
                    reasoning="比值>0.5表示更多股票接近高点"
                )
            ]

            results.append(NewHighLowAnalysis(
                ticker="^GSPC",
                name="S&P 500 New High/Low Proxy",
                new_highs_proxy=new_highs_proxy,
                new_lows_proxy=new_lows_proxy,
                hh_ll_ratio=hh_ll_ratio,
                signal=signal,
                analysis_steps=steps
            ))

        return results

    def _detect_divergences(self, data: Dict[str, pd.DataFrame]) -> List[DivergenceAnalysis]:
        """检测发散"""
        results = []

        if "RSP" in data and "^GSPC" in data:
            rsp = data["RSP"]
            spy = data["^GSPC"]

            # 计算20日滚动相关性
            window = 20
            rsp_ret = rsp['Close'].pct_change()
            spy_ret = spy['Close'].pct_change()

            # 对齐数据
            common_idx = rsp_ret.index.intersection(spy_ret.index)
            rsp_aligned = rsp_ret[common_idx]
            spy_aligned = spy_ret[common_idx]

            # 计算滚动相关性
            rolling_corr = rsp_aligned.rolling(window).corr(spy_aligned)

            if len(rolling_corr) > window:
                current_corr = rolling_corr.iloc[-1]
                avg_corr = rolling_corr.mean()

                # 判断发散
                divergence_detected = False
                divergence_type = "none"
                severity = 0.0

                if not np.isnan(current_corr) and not np.isnan(avg_corr):
                    if current_corr < avg_corr - 0.2:
                        divergence_detected = True
                        # 判断是牛市发散还是熊市发散
                        if rsp_aligned.iloc[-1] > 0:  # RSP近期上涨
                            divergence_type = "bearish"  # 等权落后
                        else:
                            divergence_type = "bearish"
                        severity = abs(current_corr - avg_corr)

                steps = [
                    BreadthStep(
                        step_name="计算滚动相关性",
                        input_data=f"RSP and SPY {window}日收益率",
                        calculation="rolling correlation",
                        result=f"Current: {current_corr:.3f}, Avg: {avg_corr:.3f}",
                        reasoning="相关性下降暗示发散"
                    ),
                    BreadthStep(
                        step_name="判断发散类型",
                        input_data=f"current_corr: {current_corr:.3f}, avg_corr: {avg_corr:.3f}",
                        calculation="compare with threshold",
                        result=f"发散: {divergence_detected}, 类型: {divergence_type}, 严重程度: {severity:.3f}",
                        reasoning="等权与市值发散表示广度问题"
                    )
                ]

                results.append(DivergenceAnalysis(
                    index_ticker="^GSPC",
                    breadth_ticker="RSP",
                    correlation=current_corr if not np.isnan(current_corr) else 0.0,
                    divergence_detected=divergence_detected,
                    divergence_type=divergence_type,
                    severity=severity,
                    analysis_steps=steps
                ))

        return results

    def _analyze_ad_line(self, data: Dict[str, pd.DataFrame]) -> Optional[ADLineAnalysis]:
        """分析A/D线 (代理)"""
        if "RSP" in data and "^GSPC" in data:
            rsp = data["RSP"]
            spy = data["^GSPC"]

            # 计算RSP-SPY差值 (代理A/D线)
            ratio = rsp['Close'] / spy['Close']
            ratio_dev = ratio - ratio.rolling(60).mean()

            # A/D线 = 累计deviation
            ad_line_value = ratio_dev.cumsum().iloc[-1]

            # 判断趋势
            if len(ratio_dev) > 20:
                ad_trend_20d = ratio_dev.iloc[-1] - ratio_dev.iloc[-20]
                if ad_trend_20d > 0.01:
                    ad_line_trend = "rising"
                elif ad_trend_20d < -0.01:
                    ad_line_trend = "falling"
                else:
                    ad_line_trend = "flat"
            else:
                ad_line_trend = "flat"

            # 检测与指数的发散
            divergence_with_index = False
            if "divergences" in dir(self) and self._detect_divergences(data):
                divergence_with_index = any(d.divergence_detected for d in self._detect_divergences(data))

            steps = [
                BreadthStep(
                    step_name="计算A/D线代理",
                    input_data=f"RSP/SPY ratio deviation",
                    calculation="cumulative sum of deviation",
                    result=f"A/D线值: {ad_line_value:.4f}",
                    reasoning="A/D线上升表示广度改善"
                ),
                BreadthStep(
                    step_name="判断A/D线趋势",
                    input_data=f"20日A/D线变化: {ad_trend_20d if len(ratio_dev) > 20 else 'N/A'}",
                    calculation="20日趋势",
                    result=f"A/D线趋势: {ad_line_trend}",
                    reasoning="上升趋势表示广度改善"
                )
            ]

            return ADLineAnalysis(
                ticker="RSP/SPY",
                ad_line_value=ad_line_value,
                ad_line_trend=ad_line_trend,
                divergence_with_index=divergence_with_index,
                analysis_steps=steps
            )

        return None

    def _generate_regression_signal(
            self,
            adv_dec: List[AdvanceDeclineAnalysis],
            new_hl: List[NewHighLowAnalysis],
            divergences: List[DivergenceAnalysis]
    ) -> Optional[BreadthRegressionSignal]:
        """生成广度回归信号"""
        # 计算综合广度百分位 (简化)
        breadth_score = 50.0  # 默认中位数

        if adv_dec:
            breadth_score = adv_dec[0].breadth_score

        # 判断极值
        if breadth_score > 80:
            signal_type = "overbought"
            expected_return = -0.02  # 预期回调
            confidence = "high"
        elif breadth_score < 20:
            signal_type = "oversold"
            expected_return = 0.02  # 预期反弹
            confidence = "high"
        else:
            signal_type = "normal"
            expected_return = 0.0
            confidence = "low"

        strength = abs(breadth_score - 50) * 2  # 0-100

        steps = [
            BreadthStep(
                step_name="计算广度百分位",
                input_data=f"breadth_score: {breadth_score:.1f}",
                calculation="historical percentile (simplified)",
                result=f"信号: {signal_type}, 强度: {strength:.1f}",
                reasoning="极端广度通常伴随均值回归"
            ),
            BreadthStep(
                step_name="生成回归信号",
                input_data=f"signal_type: {signal_type}",
                calculation="mean reversion model",
                result=f"预期5日收益: {expected_return:+.2%}, 置信度: {confidence}",
                reasoning="超买看跌，超卖看涨"
            )
        ]

        return BreadthRegressionSignal(
            signal_type=signal_type,
            strength=strength,
            expected_return_5d=expected_return,
            confidence=confidence,
            analysis_steps=steps
        )

    def _calculate_overall_score(
            self,
            adv_dec: List[AdvanceDeclineAnalysis],
            new_hl: List[NewHighLowAnalysis],
            divergences: List[DivergenceAnalysis],
            ad_line: Optional[ADLineAnalysis]
    ) -> int:
        """计算综合广度评分 (1-10)"""
        score = 5  # 基准分

        # 1. 涨跌比 (权重 0.3)
        if adv_dec:
            adv_score = adv_dec[0].breadth_score / 10  # 转换为 0-10
            score += (adv_score - 5) * 0.3

        # 2. 新高新低 (权重 0.3)
        if new_hl:
            hh_ll = new_hl[0].hh_ll_ratio
            score += (hh_ll * 10 - 5) * 0.3

        # 3. 发散 (权重 0.2)
        if divergences:
            for d in divergences:
                if d.divergence_detected:
                    if d.divergence_type == "bearish":
                        score -= d.severity * 2
                    else:
                        score += d.severity * 2

        # 4. A/D线 (权重 0.2)
        if ad_line:
            if ad_line.ad_line_trend == "rising":
                score += 1
            elif ad_line.ad_line_trend == "falling":
                score -= 1

        return max(1, min(10, int(score)))

    def _classify_regime(
            self,
            overall_score: int,
            new_hl: List[NewHighLowAnalysis],
            divergences: List[DivergenceAnalysis]
    ) -> str:
        """分类广度regime"""
        # 检查熊市发散
        has_bearish_divergence = any(
            d.divergence_detected and d.divergence_type == "bearish"
            for d in divergences
        )

        if overall_score >= 8 and not has_bearish_divergence:
            return "strong_bull"
        elif overall_score >= 6:
            return "bull"
        elif overall_score <= 3:
            return "strong_bear"
        elif overall_score <= 5 or has_bearish_divergence:
            return "bear"
        else:
            return "neutral"

    def _generate_warnings(
            self,
            divergences: List[DivergenceAnalysis],
            ad_line: Optional[ADLineAnalysis],
            new_hl: List[NewHighLowAnalysis]
    ) -> List[str]:
        """生成风险警告"""
        warnings = []

        # 1. 发散警告
        for d in divergences:
            if d.divergence_detected and d.severity > 0.3:
                warnings.append(
                    f"⚠️ 广度发散: {d.index_ticker} vs {d.breadth_ticker} "
                    f"({d.divergence_type}, 严重程度: {d.severity:.2f})"
                )

        # 2. A/D线恶化
        if ad_line and ad_line.ad_line_trend == "falling":
            warnings.append(f"⚠️ A/D线下降: 广度正在恶化")

        # 3. 新高减少
        if new_hl and new_hl[0].hh_ll_ratio < 0.3:
            warnings.append(f"⚠️ 新高新低比值低: {new_hl[0].hh_ll_ratio:.2f} (市场弱势)")

        return warnings


# ============================================================
# 辅助函数
# ============================================================

def get_breadth_summary(result: MarketBreadthResult) -> str:
    """生成市场广度分析摘要 (供UI显示)"""
    summary = f"""
## 市场广度分析摘要

**综合广度评分**: {result.overall_breadth_score}/10
**广度环境**: {result.breadth_regime}

### 涨跌比分析
"""
    for adv in result.advance_decline:
        summary += f"- {adv.name}: 比值 {adv.adv_dec_ratio:.4f}, 趋势 {adv.trend}, 评分 {adv.breadth_score:.1f}/100\n"

    summary += "\n### 新高新低分析\n"
    for hl in result.new_high_low:
        summary += f"- {hl.name}: HH/LL比值 {hl.hh_ll_ratio:.3f}, 信号 {hl.signal}\n"

    if result.divergences:
        summary += "\n### 发散检测\n"
        for d in result.divergences:
            if d.divergence_detected:
                summary += f"- ⚠️ {d.index_ticker} vs {d.breadth_ticker}: {d.divergence_type}发散 (严重程度: {d.severity:.2f})\n"

    if result.ad_line:
        summary += f"\n### A/D线分析\n"
        summary += f"- A/D线趋势: {result.ad_line.ad_line_trend}\n"
        summary += f"- 与指数发散: {'是' if result.ad_line.divergence_with_index else '否'}\n"

    if result.regression_signal:
        summary += f"\n### 回归信号\n"
        summary += f"- 信号类型: {result.regression_signal.signal_type}\n"
        summary += f"- 预期5日收益: {result.regression_signal.expected_return_5d:+.2%}\n"
        summary += f"- 置信度: {result.regression_signal.confidence}\n"

    if result.warnings:
        summary += "\n### ⚠️ 风险警告\n"
        for w in result.warnings:
            summary += f"- {w}\n"

    return summary


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 测试市场广度分析器
    analyzer = MarketBreadthAnalyzer(lookback_days=60)
    result = analyzer.analyze()

    print(f"广度评分: {result.overall_breadth_score}/10")
    print(f"广度环境: {result.breadth_regime}")
    print(f"\n摘要:\n{result.summary}")

    if result.warnings:
        print("\n⚠️ 警告:")
        for w in result.warnings:
            print(f"  - {w}")
