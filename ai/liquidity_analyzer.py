# ============================================================
# ai/liquidity_analyzer.py — 流动性分析模块
# 分析收益率曲线、利率环境、信用利差、流动性条件
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

# 国债收益率
TREASURY_TICKERS = {
    "^IRX": "13W Treasury Yield",
    "^FVX": "5Y Treasury Yield",
    "^TNX": "10Y Treasury Yield",
    "^TYX": "30Y Treasury Yield",
}

# 收益率曲线关键利差
YIELD_CURVE_SPREADS = {
    "10Y-2Y": ("^TNX", "^IRX"),  # 简化：用13W代替2Y
    "10Y-3M": ("^TNX", "^IRX"),
    "30Y-10Y": ("^TYX", "^TNX"),
}

# 信用利差
CREDIT_TICKERS = {
    "HYG": "High Yield Bond ETF",
    "LQD": "Investment Grade Bond ETF",
    "JNK": "High Yield Bond ETF (SPDR)",
    "USO": "WTI Crude Oil",
}

# 美元流动性指标
LIQUIDITY_TICKERS = {
    "UUP": "US Dollar Index Bullish",
    "FXE": "Euro Trust",
    "FXY": "Japanese Yen Trust",
    "BLOK": "Blockchain and Cryptocurrency ETF",
}


# ============================================================
# 数据类
# ============================================================

@dataclass
class YieldCurvePoint:
    """收益率曲线数据点"""
    tenor: str
    yield_value: float
    date: datetime


@dataclass
class YieldCurveAnalysis:
    """收益率曲线分析"""
    ticker: str
    name: str
    current_yield: float
    yield_1m_change: float
    yield_3m_change: float
    trend: str  # rising / falling / stable
    analysis_steps: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class YieldCurveSpread:
    """收益率曲线利差"""
    name: str
    spread: float  # 基点
    is_inverted: bool
    historical_percentile: float  # 0-100
    signal: str  # normal / inverted / steepening / flattening
    analysis_steps: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CreditSpreadAnalysis:
    """信用利差分析"""
    ticker: str
    name: str
    current_price: float
    price_1m_change: float
    spread_proxy: float  # HYG/LQD ratio as spread proxy
    spread_percentile: float
    credit_stress: str  # low / medium / high
    analysis_steps: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class LiquidityCondition:
    """流动性条件"""
    metric: str
    value: float
    interpretation: str  # ample / normal / tight
    trend: str  # improving / deteriorating / stable
    analysis_steps: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class FedPolicyImplication:
    """美联储政策暗示"""
    rate_trend: str
    curve_shape: str
    policy_bias: str  # dovish / hawkish / neutral
    next_move_probability: Dict[str, float]  # {"hike": 0.2, "cut": 0.6, "hold": 0.2}
    analysis_steps: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class LiquidityAnalysisResult:
    """流动性分析结果"""
    timestamp: datetime
    yield_curve: List[YieldCurveAnalysis]
    spreads: List[YieldCurveSpread]
    credit: List[CreditSpreadAnalysis]
    liquidity_conditions: List[LiquidityCondition]
    fed_policy: Optional[FedPolicyImplication] = None
    overall_liquidity_score: int = 5  # 1-10, 10=最宽松
    liquidity_regime: str = "normal"  # ample / normal / tight
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    analysis_steps: List[Dict[str, str]] = field(default_factory=list)


# ============================================================
# 核心分析器
# ============================================================

class LiquidityAnalyzer:
    """流动性分析器"""

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.cache = {}

    def analyze(self) -> LiquidityAnalysisResult:
        """执行完整的流动性分析"""
        steps = []
        steps.append({
            "step": "开始流动性分析",
            "action": "初始化分析器",
            "result": f"回看周期: {self.lookback_days}天"
        })

        # Step 1: 获取国债收益率数据
        yield_data = self._fetch_treasury_yields()
        steps.append({
            "step": "获取国债收益率",
            "action": f"下载 {list(TREASURY_TICKERS.keys())} 数据",
            "result": f"获取到 {len(yield_data)} 个收益率数据"
        })

        # Step 2: 分析收益率曲线
        yield_analysis = self._analyze_yield_curve(yield_data)
        steps.append({
            "step": "分析收益率曲线",
            "action": "计算各期限收益率趋势",
            "result": f"分析了 {len(yield_analysis)} 个期限"
        })

        # Step 3: 计算关键利差
        spreads = self._calculate_spreads(yield_data)
        steps.append({
            "step": "计算收益率曲线利差",
            "action": "计算10Y-3M, 10Y-2Y, 30Y-10Y利差",
            "result": f"计算出 {len(spreads)} 个利差"
        })

        # Step 4: 分析信用利差
        credit_analysis = self._analyze_credit_spreads()
        steps.append({
            "step": "分析信用利差",
            "action": "分析HYG/LQD等信用ETF",
            "result": f"分析了 {len(credit_analysis)} 个信用指标"
        })

        # Step 5: 评估流动性条件
        liquidity_conditions = self._assess_liquidity_conditions(yield_data, credit_analysis)
        steps.append({
            "step": "评估流动性条件",
            "action": "综合收益率和信用指标",
            "result": f"评估了 {len(liquidity_conditions)} 个流动性维度"
        })

        # Step 6: 解读美联储政策暗示
        fed_policy = self._interpret_fed_policy(yield_analysis, spreads)
        steps.append({
            "step": "解读美联储政策",
            "action": "分析收益率曲线形状和利率趋势",
            "result": f"政策偏向: {fed_policy.policy_bias}"
        })

        # Step 7: 计算综合流动性评分
        overall_score = self._calculate_overall_score(
            yield_analysis, spreads, credit_analysis, liquidity_conditions
        )
        steps.append({
            "step": "计算综合流动性评分",
            "action": "加权平均各维度得分",
            "result": f"综合评分: {overall_score}/10"
        })

        # Step 8: 判断流动性regime
        liquidity_regime = self._classify_regime(overall_score, spreads, credit_analysis)
        steps.append({
            "step": "判断流动性regime",
            "action": "基于综合评分和信用利差",
            "result": f"流动性环境: {liquidity_regime}"
        })

        # Step 9: 生成警告
        warnings = self._generate_warnings(spreads, credit_analysis, liquidity_conditions)
        steps.append({
            "step": "生成风险警告",
            "action": "检查收益率曲线倒挂、信用利差扩大等",
            "result": f"发现 {len(warnings)} 个警告"
        })

        # Step 10: 生成摘要
        summary = self._generate_summary(
            yield_analysis, spreads, credit_analysis,
            liquidity_conditions, fed_policy, overall_score, liquidity_regime
        )
        steps.append({
            "step": "生成分析摘要",
            "action": "汇总所有分析结果",
            "result": summary[:100] + "..."
        })

        return LiquidityAnalysisResult(
            timestamp=datetime.now(),
            yield_curve=yield_analysis,
            spreads=spreads,
            credit=credit_analysis,
            liquidity_conditions=liquidity_conditions,
            fed_policy=fed_policy,
            overall_liquidity_score=overall_score,
            liquidity_regime=liquidity_regime,
            summary=summary,
            warnings=warnings,
            analysis_steps=steps
        )

    def _fetch_treasury_yields(self) -> Dict[str, pd.DataFrame]:
        """获取国债收益率数据"""
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days * 2)  # 双倍回看期

        for ticker in TREASURY_TICKERS.keys():
            try:
                if ticker in self.cache:
                    data[ticker] = self.cache[ticker]
                    continue

                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                if not df.empty:
                    data[ticker] = df
                    self.cache[ticker] = df
            except Exception as e:
                logger.warning(f"获取 {ticker} 数据失败: {e}")

        return data

    def _analyze_yield_curve(self, yield_data: Dict[str, pd.DataFrame]) -> List[YieldCurveAnalysis]:
        """分析收益率曲线"""
        results = []

        for ticker, name in TREASURY_TICKERS.items():
            if ticker not in yield_data or yield_data[ticker].empty:
                continue

            df = yield_data[ticker]
            current_yield = float(df['Close'].iloc[-1])

            # 计算变化
            yield_1m_change = 0.0
            yield_3m_change = 0.0
            if len(df) > 20:
                yield_1m_change = current_yield - float(df['Close'].iloc[-20])
            if len(df) > 60:
                yield_3m_change = current_yield - float(df['Close'].iloc[-60])

            # 判断趋势
            if yield_1m_change > 0.1:
                trend = "rising"
            elif yield_1m_change < -0.1:
                trend = "falling"
            else:
                trend = "stable"

            steps = [
                {
                    "metric": "current_yield",
                    "value": f"{current_yield:.3f}%",
                    "interpretation": f"当前{ticker}收益率"
                },
                {
                    "metric": "1m_change",
                    "value": f"{yield_1m_change:+.3f}%",
                    "interpretation": f"1个月变化"
                }
            ]

            results.append(YieldCurveAnalysis(
                ticker=ticker,
                name=name,
                current_yield=current_yield,
                yield_1m_change=yield_1m_change,
                yield_3m_change=yield_3m_change,
                trend=trend,
                analysis_steps=steps
            ))

        return results

    def _calculate_spreads(self, yield_data: Dict[str, pd.DataFrame]) -> List[YieldCurveSpread]:
        """计算收益率曲线利差"""
        results = []

        # 10Y-3M spread
        if "^TNX" in yield_data and "^IRX" in yield_data:
            tnx = yield_data["^TNX"]['Close'].iloc[-1]
            irx = yield_data["^IRX"]['Close'].iloc[-1]
            spread = (tnx - irx) * 100  # 转换为基点

            is_inverted = spread < 0

            # 简化：使用固定历史范围估算百分位
            # 正常范围: -200bp 到 400bp
            historical_percentile = np.clip((spread + 200) / 600 * 100, 0, 100)

            if is_inverted:
                signal = "inverted"
            elif spread > 200:
                signal = "steepening"
            elif spread < 50:
                signal = "flattening"
            else:
                signal = "normal"

            steps = [
                {
                    "metric": "10Y_yield",
                    "value": f"{tnx:.3f}%",
                    "interpretation": "10年期国债收益率"
                },
                {
                    "metric": "3M_yield",
                    "value": f"{irx:.3f}%",
                    "interpretation": "3个月国债收益率"
                },
                {
                    "metric": "spread",
                    "value": f"{spread:.1f}bp",
                    "interpretation": "10Y-3M利差" + (" (倒挂!)" if is_inverted else "")
                }
            ]

            results.append(YieldCurveSpread(
                name="10Y-3M",
                spread=spread,
                is_inverted=is_inverted,
                historical_percentile=historical_percentile,
                signal=signal,
                analysis_steps=steps
            ))

        return results

    def _analyze_credit_spreads(self) -> List[CreditSpreadAnalysis]:
        """分析信用利差"""
        results = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        for ticker, name in CREDIT_TICKERS.items():
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    continue

                current_price = float(df['Close'].iloc[-1])
                price_1m_change = 0.0
                if len(df) > 20:
                    price_1m_change = (current_price / float(df['Close'].iloc[-20]) - 1) * 100

                # 使用HYG/LQD比值作为信用利差代理
                spread_proxy = 0.0
                if ticker == "HYG" and "LQD" in CREDIT_TICKERS:
                    try:
                        lqd_df = yf.download("LQD", start=start_date, end=end_date, progress=False)
                        if not lqd_df.empty:
                            spread_proxy = current_price / float(lqd_df['Close'].iloc[-1])
                    except:
                        pass

                # 判断信用压力
                # HYG价格下降 = 信用利差扩大 = 信用压力上升
                if price_1m_change < -3:
                    credit_stress = "high"
                elif price_1m_change < -1:
                    credit_stress = "medium"
                else:
                    credit_stress = "low"

                # 简化百分位
                spread_percentile = 50.0  # 默认值

                steps = [
                    {
                        "metric": "current_price",
                        "value": f"${current_price:.2f}",
                        "interpretation": f"{ticker}当前价格"
                    },
                    {
                        "metric": "1m_change",
                        "value": f"{price_1m_change:+.2f}%",
                        "interpretation": "1个月价格变化" + (" (信用压力上升)" if price_1m_change < 0 else " (信用压力下降)")
                    }
                ]

                results.append(CreditSpreadAnalysis(
                    ticker=ticker,
                    name=name,
                    current_price=current_price,
                    price_1m_change=price_1m_change,
                    spread_proxy=spread_proxy,
                    spread_percentile=spread_percentile,
                    credit_stress=credit_stress,
                    analysis_steps=steps
                ))

            except Exception as e:
                logger.warning(f"分析 {ticker} 失败: {e}")

        return results

    def _assess_liquidity_conditions(
            self,
            yield_data: Dict[str, pd.DataFrame],
            credit_analysis: List[CreditSpreadAnalysis]
    ) -> List[LiquidityCondition]:
        """评估流动性条件"""
        conditions = []

        # 1. 短端利率水平 (^IRX)
        if "^IRX" in yield_data and not yield_data["^IRX"].empty:
            short_rate = float(yield_data["^IRX"]['Close'].iloc[-1])

            if short_rate < 1.0:
                interpretation = "ample"
            elif short_rate < 3.0:
                interpretation = "normal"
            else:
                interpretation = "tight"

            # 判断趋势
            if len(yield_data["^IRX"]) > 20:
                rate_1m = float(yield_data["^IRX"]['Close'].iloc[-20])
                if short_rate > rate_1m + 0.1:
                    trend = "deteriorating"
                elif short_rate < rate_1m - 0.1:
                    trend = "improving"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            conditions.append(LiquidityCondition(
                metric="short_term_rate",
                value=short_rate,
                interpretation=interpretation,
                trend=trend,
                analysis_steps=[
                    {"metric": "13W_yield", "value": f"{short_rate:.3f}%", "interpretation": f"短端利率水平: {interpretation}"}
                ]
            ))

        # 2. 信用环境
        if credit_analysis:
            high_stress_count = sum(1 for c in credit_analysis if c.credit_stress == "high")
            med_stress_count = sum(1 for c in credit_analysis if c.credit_stress == "medium")

            if high_stress_count > 0:
                credit_condition = "tight"
                trend = "deteriorating"
            elif med_stress_count > 0:
                credit_condition = "normal"
                trend = "stable"
            else:
                credit_condition = "ample"
                trend = "improving"

            conditions.append(LiquidityCondition(
                metric="credit_environment",
                value=high_stress_count,
                interpretation=credit_condition,
                trend=trend,
                analysis_steps=[
                    {"metric": "high_stress_count", "value": str(high_stress_count), "interpretation": f"信用环境: {credit_condition}"}
                ]
            ))

        return conditions

    def _interpret_fed_policy(
            self,
            yield_analysis: List[YieldCurveAnalysis],
            spreads: List[YieldCurveSpread]
    ) -> FedPolicyImplication:
        """解读美联储政策暗示"""
        steps = []

        # 利率趋势
        rising_count = sum(1 for y in yield_analysis if y.trend == "rising")
        falling_count = sum(1 for y in yield_analysis if y.trend == "falling")

        if rising_count > falling_count:
            rate_trend = "rising"
        elif falling_count > rising_count:
            rate_trend = "falling"
        else:
            rate_trend = "stable"

        steps.append({
            "factor": "rate_trend",
            "value": rate_trend,
            "implication": "利率趋势暗示"
        })

        # 收益率曲线形状
        curve_shape = "normal"
        if spreads:
            for s in spreads:
                if s.is_inverted:
                    curve_shape = "inverted"
                    break
                elif s.signal == "flattening":
                    curve_shape = "flattening"
                elif s.signal == "steepening":
                    curve_shape = "steepening"

        steps.append({
            "factor": "curve_shape",
            "value": curve_shape,
            "implication": "收益率曲线形状"
        })

        # 政策偏向
        if curve_shape == "inverted":
            policy_bias = "dovish"  # 倒挂通常预示降息
        elif rate_trend == "rising":
            policy_bias = "hawkish"
        elif rate_trend == "falling":
            policy_bias = "dovish"
        else:
            policy_bias = "neutral"

        steps.append({
            "factor": "policy_bias",
            "value": policy_bias,
            "implication": "美联储政策偏向"
        })

        # 下一步行动概率 (简化版)
        if policy_bias == "dovish":
            next_move_probability = {"hike": 0.1, "cut": 0.6, "hold": 0.3}
        elif policy_bias == "hawkish":
            next_move_probability = {"hike": 0.6, "cut": 0.1, "hold": 0.3}
        else:
            next_move_probability = {"hike": 0.25, "cut": 0.25, "hold": 0.5}

        return FedPolicyImplication(
            rate_trend=rate_trend,
            curve_shape=curve_shape,
            policy_bias=policy_bias,
            next_move_probability=next_move_probability,
            analysis_steps=steps
        )

    def _calculate_overall_score(
            self,
            yield_analysis: List[YieldCurveAnalysis],
            spreads: List[YieldCurveSpread],
            credit_analysis: List[CreditSpreadAnalysis],
            liquidity_conditions: List[LiquidityCondition]
    ) -> int:
        """计算综合流动性评分 (1-10)"""
        score = 5  # 基准分

        # 1. 短端利率 (权重 0.3)
        for cond in liquidity_conditions:
            if cond.metric == "short_term_rate":
                if cond.interpretation == "ample":
                    score += 2
                elif cond.interpretation == "tight":
                    score -= 2

        # 2. 收益率曲线 (权重 0.3)
        for spread in spreads:
            if spread.is_inverted:
                score -= 2
            elif spread.signal == "steepening":
                score += 1
            elif spread.signal == "flattening":
                score -= 1

        # 3. 信用环境 (权重 0.4)
        for cond in liquidity_conditions:
            if cond.metric == "credit_environment":
                if cond.interpretation == "ample":
                    score += 2
                elif cond.interpretation == "tight":
                    score -= 2

        # 确保分数在 1-10 范围内
        return max(1, min(10, score))

    def _classify_regime(
            self,
            overall_score: int,
            spreads: List[YieldCurveSpread],
            credit_analysis: List[CreditSpreadAnalysis]
    ) -> str:
        """分类流动性regime"""
        # 检查警告信号
        has_inversion = any(s.is_inverted for s in spreads)
        has_high_credit_stress = any(c.credit_stress == "high" for c in credit_analysis)

        if has_inversion or has_high_credit_stress:
            return "tight"
        elif overall_score >= 7:
            return "ample"
        elif overall_score <= 4:
            return "tight"
        else:
            return "normal"

    def _generate_warnings(
            self,
            spreads: List[YieldCurveSpread],
            credit_analysis: List[CreditSpreadAnalysis],
            liquidity_conditions: List[LiquidityCondition]
    ) -> List[str]:
        """生成风险警告"""
        warnings = []

        # 1. 收益率曲线倒挂
        for spread in spreads:
            if spread.is_inverted:
                warnings.append(f"⚠️ 收益率曲线倒挂: {spread.name} = {spread.spread:.1f}bp")

        # 2. 信用压力上升
        for credit in credit_analysis:
            if credit.credit_stress == "high":
                warnings.append(f"⚠️ 信用压力高: {credit.ticker} 1个月下跌 {credit.price_1m_change:.2f}%")

        # 3. 流动性收紧
        for cond in liquidity_conditions:
            if cond.interpretation == "tight" and cond.trend == "deteriorating":
                warnings.append(f"⚠️ 流动性正在收紧: {cond.metric}")

        return warnings


# ============================================================
# 辅助函数
# ============================================================

def get_liquidity_summary(result: LiquidityAnalysisResult) -> str:
    """生成流动性分析摘要 (供UI显示)"""
    summary = f"""
## 流动性分析摘要

**综合流动性评分**: {result.overall_liquidity_score}/10
**流动性环境**: {result.liquidity_regime}

### 收益率曲线
"""
    for y in result.yield_curve:
        summary += f"- {y.name}: {y.current_yield:.3f}% ({y.trend}, 1个月变化 {y.yield_1m_change:+.3f}%)\n"

    summary += "\n### 关键利差\n"
    for s in result.spreads:
        inv = " (倒挂!)" if s.is_inverted else ""
        summary += f"- {s.name}: {s.spread:.1f}bp{inv} ({s.signal})\n"

    summary += "\n### 信用环境\n"
    for c in result.credit:
        summary += f"- {c.name}: ${c.current_price:.2f} ({c.credit_stress}信用压力, 1个月 {c.price_1m_change:+.2f}%)\n"

    if result.fed_policy:
        summary += f"\n### 美联储政策暗示\n"
        summary += f"- 利率趋势: {result.fed_policy.rate_trend}\n"
        summary += f"- 收益率曲线: {result.fed_policy.curve_shape}\n"
        summary += f"- 政策偏向: {result.fed_policy.policy_bias}\n"

    if result.warnings:
        summary += "\n### ⚠️ 风险警告\n"
        for w in result.warnings:
            summary += f"- {w}\n"

    return summary


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 测试流动性分析器
    analyzer = LiquidityAnalyzer(lookback_days=60)
    result = analyzer.analyze()

    print(f"流动性评分: {result.overall_liquidity_score}/10")
    print(f"流动性环境: {result.liquidity_regime}")
    print(f"\n摘要:\n{result.summary}")

    if result.warnings:
        print("\n⚠️ 警告:")
        for w in result.warnings:
            print(f"  - {w}")
