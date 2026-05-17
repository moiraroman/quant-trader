# ============================================================
# ai/macro_scanner.py — 宏观全景扫描模块 v2
# 三层架构：规则引擎 → 打分引擎 → 概率引擎
# ============================================================
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 常量定义
# ============================================================

INDEX_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000",
    "SOXX": "Semiconductors",
}

SAFE_HAVEN_TICKERS = {
    "TLT": "US Treasury Bond",
    "GLD": "Gold",
    "UUP": "US Dollar",
    "FXY": "Japanese Yen",
    "BTC-USD": "Bitcoin",
}

CREDIT_TICKERS = {
    "HYG": "High Yield Bond",
    "LQD": "Investment Grade Bond",
}

YIELD_TICKERS = {
    "^TNX": "10Y Treasury Yield",
    "^IRX": "13W Treasury Yield",
}

BREADTH_TICKERS = {
    "RSP": "S&P 500 Equal Weight",
}

VIX_TICKER = "^VIX"
VIX3M_TICKER = "^VIX3M"

# 模块权重
MODULE_WEIGHTS = {
    "equity": 0.25,
    "breadth": 0.20,
    "vix": 0.20,
    "credit": 0.15,
    "safe_haven": 0.10,
    "dxy": 0.05,
    "btc": 0.05,
}

# ============================================================
# 数据类
# ============================================================

@dataclass
class AnalysisStep:
    """分析步骤记录"""
    step_name: str
    input_data: str
    calculation: str
    result: str
    reasoning: str


@dataclass
class AssetAnalysis:
    """资产分析基类"""
    ticker: str
    name: str
    trend: str  # bull / bear / sideways
    trend_pct: float
    current_price: float
    ma20: float
    ma50: float
    ma200: float
    ret_5d: float
    ret_20d: float
    ret_60d: float
    vol20: float
    zscore_20: float
    analysis_steps: List[AnalysisStep] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)


@dataclass
class IndexAnalysis(AssetAnalysis):
    """大盘指数分析结果"""
    volume_ratio: float = 1.0
    trend_signals: List[str] = field(default_factory=list)


@dataclass
class SafeHavenAnalysis(AssetAnalysis):
    """避险资产分析结果"""
    is_abnormal: bool = False
    abnormal_signals: List[str] = field(default_factory=list)


@dataclass
class CreditAnalysis:
    """信用/流动性分析结果"""
    hyg_lqd_ratio: float
    hyg_ret_20d: float
    lqd_ret_20d: float
    spread_10y_2y: float
    spread_rising: bool
    curve_status: str  # normal / inverted / steep
    credit_score: float
    analysis_steps: List[AnalysisStep] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)


@dataclass
class BreadthAnalysis:
    """市场广度分析结果 v2 - 增强版（含标准差阶段、趋势质量）"""
    rsp_vs_spy_diff: float
    rsp_ret_20d: float
    spy_ret_20d: float
    breadth_signal: str  # healthy / weak / neutral / cap_driven
    breadth_score: float
    # 新增：标准差/波动阶段
    std_phase: str = "unknown"  # contraction / expansion_early / expansion_extreme
    bb_width: float = 0.0
    atr_pct: float = 0.0
    zscore: float = 0.0
    # 新增：趋势质量评分
    trend_quality: str = "unknown"  # 5star / 2star / 1star / warning
    trend_quality_score: int = 0  # 1-5
    # 新增：A/D 与 NH/NL 近似指标
    ad_ratio_approx: Optional[float] = None  # 近似 A/D 比
    nh_nl_signal: str = "unknown"  # strong / weak / diverging
    # 分析步骤
    analysis_steps: List[AnalysisStep] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)


@dataclass
class ModuleBias:
    """单个模块的多空倾向"""
    name: str
    bias: str  # 多 / 空 / 中性
    strength: float  # 0.0-1.0 倾向强度
    detail: str = ""  # 简要说明


@dataclass
class MacroEnvironment:
    """宏观市场环境 v2 - 增强版"""
    regime: str  # Risk-On / Neutral / Risk-Off
    macro_env_score: int  # 1-10
    confidence: str  # high / medium / low
    risk_appetite: str  # 兼容旧接口
    risk_score: float
    environment_score: int
    bull_count: int
    bear_count: int
    vix_level: float
    vix_signal: str
    btc_weekly_return: float
    key_drivers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    P_risk_on: float = 0.0
    P_risk_off: float = 0.0
    P_neutral: float = 1.0
    summary: str = ""
    module_scores: Dict[str, float] = field(default_factory=dict)
    module_scores_norm: Dict[str, float] = field(default_factory=dict)
    # 新增：各模块多空倾向
    module_bias: List[ModuleBias] = field(default_factory=list)
    # 新增：未来倾向预测
    forecast_5d: str = "中性"  # 多 / 空 / 中性
    forecast_30d: str = "中性"
    forecast_confidence_5d: float = 0.0
    forecast_confidence_30d: float = 0.0
    forecast_reason: str = ""


@dataclass
class MacroScanResult:
    """宏观扫描最终结果"""
    timestamp: datetime
    index_results: List[IndexAnalysis]
    haven_results: List[SafeHavenAnalysis]
    credit_result: Optional[CreditAnalysis]
    breadth_result: Optional[BreadthAnalysis]
    vix: float
    environment: MacroEnvironment
    all_steps: List[AnalysisStep] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典格式（用于序列化）"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "environment": {
                "regime": self.environment.regime,
                "macro_env_score": self.environment.macro_env_score,
                "confidence": self.environment.confidence,
                "risk_appetite": self.environment.risk_appetite,
                "risk_score": self.environment.risk_score,
                "environment_score": self.environment.environment_score,
                "bull_count": self.environment.bull_count,
                "bear_count": self.environment.bear_count,
                "vix_level": self.environment.vix_level,
                "vix_signal": self.environment.vix_signal,
                "btc_weekly_return": self.environment.btc_weekly_return,
                "key_drivers": self.environment.key_drivers,
                "warnings": self.environment.warnings,
                "P_risk_on": self.environment.P_risk_on,
                "P_risk_off": self.environment.P_risk_off,
                "P_neutral": self.environment.P_neutral,
                "summary": self.environment.summary,
                "module_scores": self.environment.module_scores,
                "module_scores_norm": self.environment.module_scores_norm,
            },
            "index_count": len(self.index_results),
            "haven_count": len(self.haven_results),
            "vix": self.vix,
            "all_steps_count": len(self.all_steps),
        }


# ============================================================
# 宏观市场扫描器 v2
# ============================================================

class MacroScanner:
    """
    宏观全景扫描器 v2
    
    三层架构：
    1. 规则引擎：快速排除明显异常
    2. 打分引擎：输出环境分数
    3. 概率引擎：输出 Risk-On/Neutral/Risk-Off 概率
    
    Example:
        scanner = MacroScanner()
        result = scanner.scan()
        print(f"Regime: {result.environment.regime}")
        print(f"Score: {result.environment.macro_env_score}/10")
        print(f"Confidence: {result.environment.confidence}")
    """

    def __init__(self, data_fetcher=None):
        self.fetcher = data_fetcher
        self._steps: List[AnalysisStep] = []
        self._data_cache: Dict[str, pd.DataFrame] = {}

    # ---- 主入口 ----

    def scan(self, period_days: int = 100) -> MacroScanResult:
        """执行完整的宏观市场扫描"""
        self._steps = []
        self._data_cache = {}
        timestamp = datetime.now()

        logger.info("=" * 60)
        logger.info("开始宏观全景扫描 v2...")

        # 初始化 fetcher
        if self.fetcher is None:
            from data.fetcher import YFinanceFetcher
            self.fetcher = YFinanceFetcher()

        # Step 1: 拉取所有数据
        self._fetch_all_data(period_days)

        # Step 2: 分析各模块
        index_results = self._analyze_all_indices()
        haven_results = self._analyze_all_havens()
        credit_result = self._analyze_credit()
        breadth_result = self._analyze_breadth()

        # Step 3: 提取 VIX
        vix_value = self._get_vix_value()

        # Step 4: 计算各模块分数
        module_scores = self._calculate_all_module_scores(
            index_results, haven_results, credit_result, breadth_result, vix_value
        )

        # Step 5: 标准化并加权合成
        module_scores_norm = self._normalize_scores(module_scores)
        macro_raw = self._weighted_sum(module_scores_norm)
        macro_env_score = self._map_to_score(macro_raw)

        # Step 6: 三层分类
        regime, P_on, P_off, P_neutral = self._classify_regime(
            macro_env_score, module_scores_norm, module_scores, vix_value
        )

        # Step 7: 冲突检测与置信度
        conflict_score = self._calculate_conflict(module_scores_norm, regime)
        confidence = self._calculate_confidence(module_scores, conflict_score)

        # Step 8: 提取主导因子和警告
        key_drivers = self._extract_key_drivers(module_scores_norm, regime)
        warnings = self._extract_warnings(module_scores, regime, vix_value)

        # Step 9: 各模块多空倾向
        module_bias = self._calculate_module_bias(
            index_results, haven_results, credit_result, breadth_result, vix_value, module_scores_norm
        )

        # Step 10: 未来5天/30天倾向预测
        forecast_5d, forecast_30d, fc_conf_5d, fc_conf_30d, fc_reason = self._forecast_outlook(
            regime, macro_env_score, module_scores_norm, module_bias, vix_value, breadth_result
        )

        # Step 11: 生成摘要
        summary = self._generate_summary(
            regime, macro_env_score, confidence, key_drivers, warnings,
            module_bias, forecast_5d, forecast_30d
        )

        # Step 12: 构建环境对象
        bull_count = sum(1 for r in index_results if r.trend == "bull")
        bear_count = sum(1 for r in index_results if r.trend == "bear")

        environment = MacroEnvironment(
            regime=regime,
            macro_env_score=macro_env_score,
            confidence=confidence,
            risk_appetite=regime,  # 兼容旧接口
            risk_score=100 - macro_env_score * 10,  # 反向映射
            environment_score=macro_env_score,
            bull_count=bull_count,
            bear_count=bear_count,
            vix_level=vix_value,
            vix_signal=self._get_vix_signal(vix_value),
            btc_weekly_return=module_scores.get("btc", 0),
            key_drivers=key_drivers,
            warnings=warnings,
            P_risk_on=P_on,
            P_risk_off=P_off,
            P_neutral=P_neutral,
            summary=summary,
            module_scores=module_scores,
            module_scores_norm=module_scores_norm,
            module_bias=module_bias,
            forecast_5d=forecast_5d,
            forecast_30d=forecast_30d,
            forecast_confidence_5d=fc_conf_5d,
            forecast_confidence_30d=fc_conf_30d,
            forecast_reason=fc_reason,
        )

        # Step 11: 构建最终结果
        result = MacroScanResult(
            timestamp=timestamp,
            index_results=index_results,
            haven_results=haven_results,
            credit_result=credit_result,
            breadth_result=breadth_result,
            vix=vix_value,
            environment=environment,
            all_steps=self._steps,
        )

        logger.info(f"宏观扫描完成: Regime={regime}, Score={macro_env_score}/10, Confidence={confidence}")
        logger.info("=" * 60)

        return result

    def quick_scan(self) -> Dict:
        """快速扫描（返回简化结果）"""
        result = self.scan()
        return result.to_dict()

    # ---- 数据获取 ----

    def _fetch_all_data(self, period_days: int):
        """拉取所有需要的数据"""
        all_tickers = (
            list(INDEX_TICKERS.keys()) +
            list(SAFE_HAVEN_TICKERS.keys()) +
            list(CREDIT_TICKERS.keys()) +
            list(YIELD_TICKERS.keys()) +
            list(BREADTH_TICKERS.keys()) +
            [VIX_TICKER, VIX3M_TICKER]
        )

        for ticker in all_tickers:
            try:
                df = self.fetcher.download_history(ticker, period=f"{period_days}d")
                if df is not None and not df.empty:
                    self._data_cache[ticker] = df
                    logger.debug(f"  ✓ {ticker}: {len(df)} 条")
            except Exception as e:
                logger.warning(f"  ✗ {ticker} 获取失败: {e}")

    def _get_df(self, ticker: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        return self._data_cache.get(ticker)

    # ---- 资产分析 ----

    def _compute_asset_features(self, df: pd.DataFrame) -> Dict:
        """计算资产的通用特征"""
        if df is None or df.empty or len(df) < 5:
            return {}

        close = df["Close"]
        
        # 收益率
        ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
        ret_60d = (close.iloc[-1] / close.iloc[-60] - 1) * 100 if len(close) >= 60 else 0

        # 均线
        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.mean()
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50

        # 波动率
        ret = close.pct_change().dropna()
        vol20 = ret.rolling(20).std().iloc[-1] * 100 * (252**0.5) if len(ret) >= 20 else 0

        # Z-score
        mean20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        zscore_20 = (close.iloc[-1] - mean20) / std20 if std20 and std20 > 0 else 0

        current_price = close.iloc[-1]

        return {
            "current_price": current_price,
            "ma20": ma20,
            "ma50": ma50,
            "ma200": ma200,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "ret_60d": ret_60d,
            "vol20": vol20,
            "zscore_20": zscore_20,
        }

    def _determine_trend(self, current: float, ma20: float, ma50: float, ma200: float) -> Tuple[str, float]:
        """判定趋势方向和强度"""
        if current > ma20 > ma50 > ma200:
            return "bull", 100.0
        elif current < ma20 < ma50 < ma200:
            return "bear", 0.0
        elif current > ma50:
            dev = ((current - ma50) / ma50) * 100
            strength = min(100, 50 + dev * 2)
            return "bull", strength
        elif current < ma50:
            dev = ((ma50 - current) / ma50) * 100
            strength = max(0, 50 - dev * 2)
            return "bear", strength
        else:
            return "sideways", 50.0

    def _analyze_all_indices(self) -> List[IndexAnalysis]:
        """分析所有大盘指数"""
        results = []
        for ticker, name in INDEX_TICKERS.items():
            df = self._get_df(ticker)
            if df is None or df.empty:
                continue
            result = self._analyze_index(df, ticker, name)
            results.append(result)
            self._steps.extend(result.analysis_steps)
        return results

    def _analyze_index(self, df: pd.DataFrame, ticker: str, name: str) -> IndexAnalysis:
        """分析单个指数"""
        steps = []
        features = self._compute_asset_features(df)
        
        if not features:
            return IndexAnalysis(
                ticker=ticker, name=name, trend="sideways", trend_pct=50.0,
                current_price=0, ma20=0, ma50=0, ma200=0,
                ret_5d=0, ret_20d=0, ret_60d=0, vol20=0, zscore_20=0
            )

        current = features["current_price"]
        ma20 = features["ma20"]
        ma50 = features["ma50"]
        ma200 = features["ma200"]

        trend, trend_pct = self._determine_trend(current, ma20, ma50, ma200)

        # 成交量比
        volume = df["Volume"]
        avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1.0

        # 趋势信号
        trend_signals = []
        if current > ma20:
            trend_signals.append("价格>MA20")
        else:
            trend_signals.append("价格<MA20")
        if current > ma50:
            trend_signals.append("价格>MA50")
        else:
            trend_signals.append("价格<MA50")
        if current > ma200:
            trend_signals.append("价格>MA200")
        else:
            trend_signals.append("价格<MA200")

        # 记录分析步骤
        steps.append(AnalysisStep(
            step_name=f"{ticker} - 均线分析",
            input_data=f"价格={current:.2f}, MA20={ma20:.2f}, MA50={ma50:.2f}, MA200={ma200:.2f}",
            calculation="均线排列判断",
            result=f"趋势={trend}, 强度={trend_pct:.1f}%",
            reasoning="完美多头排列为bull 100%，完美空头排列为bear 0%"
        ))

        steps.append(AnalysisStep(
            step_name=f"{ticker} - 动量分析",
            input_data=f"ret_5d={features['ret_5d']:.2f}%, ret_20d={features['ret_20d']:.2f}%, ret_60d={features['ret_60d']:.2f}%",
            calculation="动量得分 = 0.4*5d + 0.3*20d + 0.3*60d",
            result=f"动量得分={0.4*features['ret_5d'] + 0.3*features['ret_20d'] + 0.3*features['ret_60d']:.2f}%",
            reasoning="加权动量反映短期趋势强度"
        ))

        return IndexAnalysis(
            ticker=ticker,
            name=name,
            trend=trend,
            trend_pct=trend_pct,
            current_price=current,
            ma20=ma20,
            ma50=ma50,
            ma200=ma200,
            ret_5d=features["ret_5d"],
            ret_20d=features["ret_20d"],
            ret_60d=features["ret_60d"],
            vol20=features["vol20"],
            zscore_20=features["zscore_20"],
            volume_ratio=volume_ratio,
            trend_signals=trend_signals,
            analysis_steps=steps,
            raw_data=features,
        )

    def _analyze_all_havens(self) -> List[SafeHavenAnalysis]:
        """分析所有避险资产"""
        results = []
        for ticker, name in SAFE_HAVEN_TICKERS.items():
            df = self._get_df(ticker)
            if df is None or df.empty:
                continue
            result = self._analyze_haven(df, ticker, name)
            results.append(result)
            self._steps.extend(result.analysis_steps)
        return results

    def _analyze_haven(self, df: pd.DataFrame, ticker: str, name: str) -> SafeHavenAnalysis:
        """分析单个避险资产"""
        steps = []
        features = self._compute_asset_features(df)

        if not features:
            return SafeHavenAnalysis(
                ticker=ticker, name=name, trend="sideways", trend_pct=50.0,
                current_price=0, ma20=0, ma50=0, ma200=0,
                ret_5d=0, ret_20d=0, ret_60d=0, vol20=0, zscore_20=0
            )

        current = features["current_price"]
        ma20 = features["ma20"]
        ma50 = features["ma50"]
        ma200 = features["ma200"]

        trend, trend_pct = self._determine_trend(current, ma20, ma50, ma200)

        # 异动检测
        abnormal_signals = []
        is_abnormal = False

        # 1. 周涨幅异常
        if abs(features["ret_5d"]) > 5:
            is_abnormal = True
            abnormal_signals.append(f"周涨幅异常: {features['ret_5d']:+.1f}%")

        # 2. 单日波幅 > 3%
        daily_ret = df["Close"].pct_change().dropna() * 100
        if len(daily_ret) > 0 and daily_ret.abs().iloc[-1] > 3:
            is_abnormal = True
            abnormal_signals.append(f"单日波幅: {daily_ret.iloc[-1]:+.1f}%")

        steps.append(AnalysisStep(
            step_name=f"{ticker} ({name}) - 趋势分析",
            input_data=f"价格={current:.2f}, MA50={ma50:.2f}",
            calculation=f"偏离MA50: {((current - ma50) / ma50) * 100:+.2f}%",
            result=f"趋势={trend}, 异动={'是' if is_abnormal else '否'}",
            reasoning=f"{'检测到异动: ' + ', '.join(abnormal_signals) if is_abnormal else '无明显异动'}"
        ))

        return SafeHavenAnalysis(
            ticker=ticker,
            name=name,
            trend=trend,
            trend_pct=trend_pct,
            current_price=current,
            ma20=ma20,
            ma50=ma50,
            ma200=ma200,
            ret_5d=features["ret_5d"],
            ret_20d=features["ret_20d"],
            ret_60d=features["ret_60d"],
            vol20=features["vol20"],
            zscore_20=features["zscore_20"],
            is_abnormal=is_abnormal,
            abnormal_signals=abnormal_signals,
            analysis_steps=steps,
            raw_data=features,
        )

    def _analyze_credit(self) -> Optional[CreditAnalysis]:
        """分析信用/流动性"""
        steps = []
        hyg_df = self._get_df("HYG")
        lqd_df = self._get_df("LQD")
        tnx_df = self._get_df("^TNX")
        irx_df = self._get_df("^IRX")

        if hyg_df is None or hyg_df.empty:
            logger.warning("无法获取 HYG 数据")
            return None

        hyg_features = self._compute_asset_features(hyg_df)
        lqd_features = self._compute_asset_features(lqd_df) if lqd_df is not None and not lqd_df.empty else {}

        hyg_ret_20d = hyg_features.get("ret_20d", 0)
        lqd_ret_20d = lqd_features.get("ret_20d", 0)
        hyg_lqd_ratio = hyg_features.get("current_price", 1) / max(lqd_features.get("current_price", 1), 0.01)

        # 收益率曲线
        spread_10y_2y = 0
        spread_rising = False
        curve_status = "unknown"

        if tnx_df is not None and not tnx_df.empty and irx_df is not None and not irx_df.empty:
            tnx_yield = tnx_df["Close"].iloc[-1]
            irx_yield = irx_df["Close"].iloc[-1]
            spread_10y_2y = tnx_yield - irx_yield

            if len(tnx_df) >= 20:
                old_tnx = tnx_df["Close"].iloc[-20]
                old_irx = irx_df["Close"].iloc[-20]
                old_spread = old_tnx - old_irx
                spread_rising = spread_10y_2y > old_spread

            if spread_10y_2y > 0.5:
                curve_status = "normal"
            elif spread_10y_2y > 0:
                curve_status = "flat"
            else:
                curve_status = "inverted"

        # 信用评分
        credit_score = 0.0

        # HYG/LQD 信号
        if hyg_ret_20d > lqd_ret_20d and hyg_ret_20d > 0:
            credit_score += 1.0
            credit_reason = "高收益债跑赢投资级，信用偏好强"
        elif hyg_ret_20d < lqd_ret_20d:
            credit_score -= 1.0
            credit_reason = "高收益债弱于投资级，信用收缩"
        else:
            credit_reason = "信用中性"

        # 收益率曲线信号
        if spread_10y_2y > 0 and spread_rising:
            credit_score += 0.5
            curve_reason = "收益率曲线陡峭化，经济预期改善"
        elif spread_10y_2y < 0 and not spread_rising:
            credit_score -= 1.0
            curve_reason = "收益率曲线倒挂且恶化，经济压力"
        elif spread_10y_2y < 0:
            credit_score -= 0.5
            curve_reason = "收益率曲线倒挂"
        else:
            curve_reason = "曲线中性"

        steps.append(AnalysisStep(
            step_name="信用/流动性分析",
            input_data=f"HYG 20d={hyg_ret_20d:.2f}%, LQD 20d={lqd_ret_20d:.2f}%",
            calculation=f"HYG/LQD 比值={hyg_lqd_ratio:.3f}",
            result=f"信用分={credit_score:.2f}",
            reasoning=credit_reason
        ))

        steps.append(AnalysisStep(
            step_name="收益率曲线分析",
            input_data=f"10Y={tnx_df['Close'].iloc[-1] if tnx_df is not None else 'N/A'}, 2Y={irx_df['Close'].iloc[-1] if irx_df is not None else 'N/A'}",
            calculation=f"Spread={spread_10y_2y:.2f}%, 上升={spread_rising}",
            result=f"曲线状态={curve_status}",
            reasoning=curve_reason
        ))

        return CreditAnalysis(
            hyg_lqd_ratio=hyg_lqd_ratio,
            hyg_ret_20d=hyg_ret_20d,
            lqd_ret_20d=lqd_ret_20d,
            spread_10y_2y=spread_10y_2y,
            spread_rising=spread_rising,
            curve_status=curve_status,
            credit_score=credit_score,
            analysis_steps=steps,
            raw_data={"hyg": hyg_features, "lqd": lqd_features},
        )

    def _analyze_breadth(self) -> Optional[BreadthAnalysis]:
        """分析市场广度 v2 - 增强版（含标准差阶段 + 趋势质量）"""
        steps = []
        rsp_df = self._get_df("RSP")
        spy_df = self._get_df("SPY")

        if rsp_df is None or rsp_df.empty or spy_df is None or spy_df.empty:
            logger.warning("无法获取 RSP 或 SPY 数据")
            return None

        rsp_features = self._compute_asset_features(rsp_df)
        spy_features = self._compute_asset_features(spy_df)

        rsp_ret_20d = rsp_features.get("ret_20d", 0)
        spy_ret_20d = spy_features.get("ret_20d", 0)
        rsp_vs_spy_diff = rsp_ret_20d - spy_ret_20d

        # ===== 1. 标准差/波动阶段分析 =====
        close = spy_df["Close"]
        # Bollinger Band 宽度
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_width = ((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / ma20.iloc[-1] * 100) if ma20.iloc[-1] > 0 else 0

        # ATR%
        high = spy_df["High"]
        low = spy_df["Low"]
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = (atr.iloc[-1] / close.iloc[-1] * 100) if close.iloc[-1] > 0 else 0

        # Z-score
        zscore = rsp_features.get("zscore_20", 0)

        # 标准差阶段判断
        # 历史 BB 宽度参考
        hist_bb_width = ((bb_upper - bb_lower) / ma20 * 100).dropna()
        bb_pctile = (hist_bb_width < bb_width).mean() * 100 if len(hist_bb_width) > 0 else 50

        if bb_pctile < 20:
            std_phase = "contraction"  # 收缩
            std_reason = "Bollinger收窄，市场蓄力"
        elif bb_pctile < 70:
            std_phase = "expansion_early"  # 扩张初期
            std_reason = "波动刚扩张，趋势确认中"
        else:
            std_phase = "expansion_extreme"  # 极端扩张
            std_reason = "波动极端，警惕回归均值"

        # ===== 2. A/D 近似（用涨跌天数比）=====
        spy_daily_ret = spy_df["Close"].pct_change().dropna() * 100
        up_days = (spy_daily_ret > 0).sum()
        down_days = (spy_daily_ret < 0).sum()
        ad_ratio_approx = up_days / max(down_days, 1)

        # ===== 3. NH/NL 近似信号（用创新高/新低天数）=====
        rolling_high = spy_df["Close"].rolling(20).max()
        rolling_low = spy_df["Close"].rolling(20).min()
        is_nh = spy_df["Close"] >= rolling_high
        is_nl = spy_df["Close"] <= rolling_low
        nh_count = is_nh.sum()
        nl_count = is_nl.sum()

        if nh_count > nl_count * 2 and rsp_vs_spy_diff > 0:
            nh_nl_signal = "strong"
            nh_nl_reason = "新高远多于新低，趋势强劲"
        elif nl_count > nh_count and rsp_vs_spy_diff < -1:
            nh_nl_signal = "weak"
            nh_nl_reason = "新低增多，趋势转弱"
        elif nh_count > 0 and nl_count > 0 and rsp_vs_spy_diff < -0.5:
            nh_nl_signal = "diverging"
            nh_nl_reason = "指数强但广度弱，顶背离风险"
        else:
            nh_nl_signal = "neutral"
            nh_nl_reason = "新高新低均衡"

        # ===== 4. 广度信号（四种状态）=====
        if rsp_vs_spy_diff > 1.0 and ad_ratio_approx > 1.5:
            breadth_signal = "healthy"
            breadth_score = 2.0
            breadth_reason = "广度健康：等权强于指数，大部分股票在涨"
        elif rsp_vs_spy_diff < -1.5:
            breadth_signal = "weak"
            breadth_score = -2.0
            breadth_reason = "广度背离：少数权重股拉动，虚假强势"
        elif rsp_vs_spy_diff < -0.5 and spy_ret_20d > 0:
            breadth_signal = "cap_driven"
            breadth_score = -1.0
            breadth_reason = "权重驱动：指数涨但等权弱，靠少数大票"
        else:
            breadth_signal = "neutral"
            breadth_score = rsp_vs_spy_diff
            breadth_reason = "广度中性"

        # ===== 5. 趋势质量评分（广度 × 标准差阶段）=====
        # 广度强度: 强=3, 中=2, 弱=1
        breadth_strength = 3 if breadth_signal == "healthy" else (2 if breadth_signal == "neutral" else 1)
        # 标准差阶段: 扩张初期=3, 收缩=2, 极端=1
        std_strength = 3 if std_phase == "expansion_early" else (2 if std_phase == "contraction" else 1)
        trend_quality_score = breadth_strength * std_strength  # 1-9

        if trend_quality_score >= 7:
            trend_quality = "5star"  # ⭐⭐⭐⭐⭐
            tq_reason = "广度强 + 标准差扩张初期 → 最佳做多点"
        elif trend_quality_score >= 5:
            trend_quality = "3star"  # ⭐⭐⭐
            tq_reason = "趋势尚可，但非最佳"
        elif trend_quality_score >= 3:
            trend_quality = "2star"  # ⭐⭐
            tq_reason = "弱势或极端，谨慎"
        else:
            trend_quality = "1star"  # ⭐
            tq_reason = "假突破或崩盘风险"

        steps.append(AnalysisStep(
            step_name="市场广度分析 v2",
            input_data=f"RSP 20d={rsp_ret_20d:.2f}%, SPY 20d={spy_ret_20d:.2f}%, A/D≈{ad_ratio_approx:.2f}",
            calculation=f"BB宽度={bb_width:.2f}%, ATR%={atr_pct:.2f}%, Z={zscore:.2f}",
            result=f"广度={breadth_signal}, 标准差={std_phase}, 趋势质量={trend_quality}",
            reasoning=f"{breadth_reason}; {std_reason}; {tq_reason}"
        ))

        return BreadthAnalysis(
            rsp_vs_spy_diff=rsp_vs_spy_diff,
            rsp_ret_20d=rsp_ret_20d,
            spy_ret_20d=spy_ret_20d,
            breadth_signal=breadth_signal,
            breadth_score=breadth_score,
            std_phase=std_phase,
            bb_width=bb_width,
            atr_pct=atr_pct,
            zscore=zscore,
            trend_quality=trend_quality,
            trend_quality_score=trend_quality_score,
            ad_ratio_approx=round(ad_ratio_approx, 2),
            nh_nl_signal=nh_nl_signal,
            analysis_steps=steps,
            raw_data={"rsp": rsp_features, "spy": spy_features, "bb_pctile": bb_pctile},
        )

    def _get_vix_value(self) -> float:
        """获取 VIX 当前值"""
        vix_df = self._get_df(VIX_TICKER)
        if vix_df is not None and not vix_df.empty:
            return float(vix_df["Close"].iloc[-1])
        return 20.0  # 默认值

    # ---- 模块评分 ----

    def _calculate_all_module_scores(
        self,
        index_results: List[IndexAnalysis],
        haven_results: List[SafeHavenAnalysis],
        credit_result: Optional[CreditAnalysis],
        breadth_result: Optional[BreadthAnalysis],
        vix: float
    ) -> Dict[str, float]:
        """计算所有模块的原始分数"""
        scores = {}

        # 1. Equity Score
        scores["equity"] = self._calc_equity_score(index_results)

        # 2. VIX Score
        scores["vix"] = self._calc_vix_score(vix)

        # 3. Breadth Score
        scores["breadth"] = self._calc_breadth_score(breadth_result)

        # 4. Credit Score
        scores["credit"] = self._calc_credit_score(credit_result)

        # 5. Safe Haven Score (注意取反，因为避险涨=风险偏好弱)
        scores["safe_haven"] = -self._calc_safe_haven_score(haven_results, vix, scores.get("equity", 0))

        # 6. DXY Score
        scores["dxy"] = self._calc_dxy_score()

        # 7. BTC Score
        scores["btc"] = self._calc_btc_score(haven_results)

        return scores

    def _calc_equity_score(self, index_results: List[IndexAnalysis]) -> float:
        """计算权益指数得分"""
        if not index_results:
            return 0.0

        total_score = 0.0
        for idx in index_results:
            # 趋势得分
            if idx.trend == "bull":
                trend_score = 2.0
            elif idx.trend == "bear":
                trend_score = -2.0
            else:
                trend_score = 1.0 if idx.current_price > idx.ma50 else -1.0
                if idx.current_price > idx.ma200:
                    trend_score += 0.5
                elif idx.current_price < idx.ma200:
                    trend_score -= 0.5
                if idx.ma20 > idx.ma50:
                    trend_score += 0.5
                elif idx.ma20 < idx.ma50:
                    trend_score -= 0.5

            # 动量得分
            momentum_score = 0.4 * idx.ret_5d + 0.3 * idx.ret_20d + 0.3 * idx.ret_60d

            total_score += trend_score + momentum_score

        return total_score / len(index_results)

    def _calc_vix_score(self, vix: float) -> float:
        """计算 VIX 得分"""
        # 绝对值分档
        if vix < 15:
            level_score = 1.0
        elif vix < 20:
            level_score = 0.5
        elif vix < 25:
            level_score = 0.0
        elif vix < 35:
            level_score = -1.0
        else:
            level_score = -2.0

        # 动量
        vix_df = self._get_df(VIX_TICKER)
        momentum_score = 0.0
        if vix_df is not None and len(vix_df) >= 5:
            vix_ret_5d = (vix_df["Close"].iloc[-1] / vix_df["Close"].iloc[-5] - 1) * 100
            if vix_ret_5d > 10:
                momentum_score = -0.5
            elif vix_ret_5d < -10:
                momentum_score = 0.3

        # 期限结构
        term_score = 0.0
        vix3m_df = self._get_df(VIX3M_TICKER)
        if vix3m_df is not None and not vix3m_df.empty:
            vix3m = vix3m_df["Close"].iloc[-1]
            if vix < vix3m:
                term_score = 0.5  # contango, calm
            else:
                term_score = -0.5  # backwardation, fear

        total = level_score + momentum_score + term_score
        
        self._steps.append(AnalysisStep(
            step_name="VIX 评分",
            input_data=f"VIX={vix:.2f}",
            calculation=f"level={level_score}, momentum={momentum_score}, term={term_score}",
            result=f"vix_score={total:.2f}",
            reasoning=f"VIX {vix:.0f}档位={level_score:+.1f}, 动量={momentum_score:+.1f}, 期限结构={term_score:+.1f}"
        ))

        return total

    def _calc_breadth_score(self, breadth_result: Optional[BreadthAnalysis]) -> float:
        """计算广度得分"""
        if breadth_result is None:
            return 0.0
        return breadth_result.breadth_score

    def _calc_credit_score(self, credit_result: Optional[CreditAnalysis]) -> float:
        """计算信用得分"""
        if credit_result is None:
            return 0.0
        return credit_result.credit_score

    def _calc_safe_haven_score(
        self,
        haven_results: List[SafeHavenAnalysis],
        vix: float,
        equity_score: float
    ) -> float:
        """计算避险资产得分（不取反，在总计算时取反）"""
        total = 0.0

        # 债券组 (TLT)
        tlt = next((r for r in haven_results if r.ticker == "TLT"), None)
        bond_score = 0.0
        safety_confirm = 1.0

        if tlt:
            if tlt.current_price > tlt.ma50 and tlt.ret_20d > 0:
                bond_score = 1.0
            elif tlt.current_price < tlt.ma50 and tlt.ret_20d < 0:
                bond_score = -1.0
            else:
                bond_score = 0.0

            if tlt.current_price > tlt.ma200:
                bond_score += 0.5
            elif tlt.current_price < tlt.ma200:
                bond_score -= 0.5

            # 条件修正
            vix_rising = False
            vix_df = self._get_df(VIX_TICKER)
            if vix_df is not None and len(vix_df) >= 5:
                vix_ret = (vix_df["Close"].iloc[-1] / vix_df["Close"].iloc[-5] - 1) * 100
                vix_rising = vix_ret > 0

            equity_falling = equity_score < 0
            tlt_rising = tlt.ret_20d > 0

            if vix_rising and equity_falling and tlt_rising:
                safety_confirm = 1.5
            elif tlt_rising and bond_score > 0:
                safety_confirm = 1.2
            else:
                safety_confirm = 0.8

            bond_score *= safety_confirm

        # 货币组 (UUP, FXY)
        uup = next((r for r in haven_results if r.ticker == "UUP"), None)
        fxy = next((r for r in haven_results if r.ticker == "FXY"), None)

        dollar_score = 0.0
        if uup:
            if uup.current_price > uup.ma50 and uup.ret_20d > 0:
                dollar_score = 1.0
            elif uup.current_price < uup.ma50 and uup.ret_20d < 0:
                dollar_score = -1.0

        jpy_score = 0.0
        if fxy:
            if fxy.current_price > fxy.ma50 and fxy.ret_20d > 0:
                jpy_score = 1.0
            elif fxy.current_price < fxy.ma50 and fxy.ret_20d < 0:
                jpy_score = -1.0

        currency_score = (dollar_score + jpy_score) / 2

        # 黄金组 (GLD)
        gld = next((r for r in haven_results if r.ticker == "GLD"), None)
        gold_score = 0.0

        if gld:
            if gld.current_price > gld.ma50 and gld.ret_20d > 0:
                gold_score = 1.0
            elif gld.current_price < gld.ma50 and gld.ret_20d < 0:
                gold_score = -1.0

        total = (bond_score + currency_score + gold_score) / 3

        self._steps.append(AnalysisStep(
            step_name="避险资产评分",
            input_data=f"TLT={tlt.trend if tlt else 'N/A'}, GLD={gld.trend if gld else 'N/A'}, UUP/FXY",
            calculation=f"bond={bond_score:.2f}, currency={currency_score:.2f}, gold={gold_score:.2f}",
            result=f"safe_haven={total:.2f}",
            reasoning="避险资产上涨表示风险偏好下降（取反后使用）"
        ))

        return total

    def _calc_dxy_score(self) -> float:
        """计算美元指数得分"""
        uup = self._get_df("UUP")
        if uup is None or uup.empty:
            return 0.0

        features = self._compute_asset_features(uup)
        if not features:
            return 0.0

        if features["ret_20d"] > 0:
            return -1.0  # 美元强 = 压制风险资产
        else:
            return 1.0   # 美元弱 = 利好风险资产

    def _calc_btc_score(self, haven_results: List[SafeHavenAnalysis]) -> float:
        """计算 BTC 得分"""
        btc = next((r for r in haven_results if r.ticker == "BTC-USD"), None)
        if btc is None:
            return 0.0

        score = 0.0
        if btc.current_price > btc.ma50 and btc.ret_20d > 0:
            score = 1.0
        elif btc.current_price < btc.ma50 and btc.ret_20d < 0:
            score = -1.0

        # 暴跌额外扣分
        if btc.ret_5d < -5:
            score -= 0.5

        return score

    # ---- 标准化与合成 ----

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """将各模块分数标准化到 [-1, 1]"""
        normalized = {}
        for module, score in scores.items():
            if abs(score) <= 2:
                normalized[module] = score / 2
            else:
                normalized[module] = 1.0 if score > 0 else -1.0
        return normalized

    def _weighted_sum(self, norm_scores: Dict[str, float]) -> float:
        """加权求和"""
        total = 0.0
        for module, weight in MODULE_WEIGHTS.items():
            total += weight * norm_scores.get(module, 0.0)
        return total

    def _map_to_score(self, macro_raw: float) -> int:
        """映射到 1-10 分"""
        score = 5.5 + 4.5 * macro_raw
        score = max(1, min(10, round(score)))
        return int(score)

    # ---- 三层分类 ----

    def _classify_regime(
        self,
        macro_score: int,
        norm_scores: Dict[str, float],
        raw_scores: Dict[str, float],
        vix: float
    ) -> Tuple[str, float, float, float]:
        """三层架构分类"""
        equity_norm = norm_scores.get("equity", 0)
        vix_norm = norm_scores.get("vix", 0)

        # 第一层：规则引擎
        if vix > 35:
            regime = "Risk-Off"
            P_on, P_off, P_neutral = 0.05, 0.85, 0.10
            self._steps.append(AnalysisStep(
                step_name="规则引擎 - VIX恐慌",
                input_data=f"VIX={vix:.2f}>35",
                calculation="直接判定 Risk-Off",
                result=f"Regime={regime}",
                reasoning="VIX超过35触发恐慌条款，直接Risk-Off"
            ))
            return regime, P_on, P_off, P_neutral

        if equity_norm > 0.75 and vix_norm > 0:
            regime = "Risk-On"
            P_on, P_off, P_neutral = 0.80, 0.05, 0.15
            self._steps.append(AnalysisStep(
                step_name="规则引擎 - 强多头",
                input_data=f"equity_norm={equity_norm:.2f}>0.75, vix_norm={vix_norm:.2f}>0",
                calculation="直接判定 Risk-On",
                result=f"Regime={regime}",
                reasoning="权益指数强多且VIX平静，直接Risk-On"
            ))
            return regime, P_on, P_off, P_neutral

        # 第二层：打分引擎
        if macro_score >= 7 and vix_norm > 0 and equity_norm > 0:
            regime = "Risk-On"
        elif macro_score <= 4 and vix_norm < 0 and equity_norm < 0:
            regime = "Risk-Off"
        else:
            regime = "Neutral"

        # 第三层：概率引擎
        def sigmoid(x):
            return 1 / (1 + math.exp(-max(-10, min(10, x))))

        P_on = sigmoid(
            2 * equity_norm +
            1.5 * norm_scores.get("breadth", 0) -
            2 * vix_norm -
            norm_scores.get("credit", 0)
        )
        P_off = sigmoid(
            2 * vix_norm +
            1.5 * norm_scores.get("credit", 0) +
            norm_scores.get("dxy", 0) -
            2 * equity_norm -
            norm_scores.get("breadth", 0)
        )

        # 归一化
        total = P_on + P_off
        if total > 1:
            P_on = P_on / total * 0.95
            P_off = P_off / total * 0.95
        P_neutral = 1 - P_on - P_off

        self._steps.append(AnalysisStep(
            step_name="概率引擎",
            input_data=f"equity={equity_norm:.2f}, vix={vix_norm:.2f}",
            calculation=f"sigmoid组合",
            result=f"P(Risk-On)={P_on:.2f}, P(Risk-Off)={P_off:.2f}",
            reasoning="基于sigmoid的概率分布计算"
        ))

        return regime, P_on, P_off, P_neutral

    # ---- 置信度 ----

    def _calculate_conflict(self, norm_scores: Dict[str, float], regime: str) -> float:
        """计算冲突分"""
        disagree = 0
        total = 0
        for module, score in norm_scores.items():
            if module in ("safe_haven", "dxy"):
                # 这些模块反向
                expected = -1 if regime == "Risk-On" else 1 if regime == "Risk-Off" else 0
            else:
                expected = 1 if regime == "Risk-On" else -1 if regime == "Risk-Off" else 0

            if expected != 0:
                total += 1
                if (score > 0 and expected < 0) or (score < 0 and expected > 0):
                    disagree += 1

        return disagree / max(total, 1)

    def _calculate_confidence(self, scores: Dict[str, float], conflict: float) -> str:
        """计算置信度"""
        strength = np.mean([abs(s) for s in scores.values()]) if scores else 0.5
        confidence_raw = (1 - conflict) * strength

        self._steps.append(AnalysisStep(
            step_name="置信度计算",
            input_data=f"conflict={conflict:.2f}, strength={strength:.2f}",
            calculation=f"(1-{conflict:.2f})*{strength:.2f}={confidence_raw:.2f}",
            result=f"confidence={confidence_raw:.2f}",
            reasoning=f"冲突率{conflict*100:.0f}%, 信号强度{strength:.2f}"
        ))

        if confidence_raw >= 0.75:
            return "high"
        elif confidence_raw >= 0.50:
            return "medium"
        else:
            return "low"

    # ---- 主导因子与警告 ----

    def _extract_key_drivers(self, norm_scores: Dict[str, float], regime: str) -> List[str]:
        """提取主导因子"""
        drivers = []
        for module, score in sorted(norm_scores.items(), key=lambda x: -abs(x[1])):
            if module == "safe_haven":
                score = -score  # 反向

            if regime == "Risk-On" and score > 0.3:
                drivers.append(self._module_display_name(module))
            elif regime == "Risk-Off" and score < -0.3:
                drivers.append(self._module_display_name(module))
            elif regime == "Neutral" and abs(score) > 0.3:
                drivers.append(self._module_display_name(module))

            if len(drivers) >= 4:
                break

        return drivers if drivers else ["综合评估"]

    def _module_display_name(self, module: str) -> str:
        names = {
            "equity": "权益指数",
            "breadth": "市场广度",
            "vix": "波动率",
            "credit": "信用/流动性",
            "safe_haven": "避险资产",
            "dxy": "美元",
            "btc": "比特币",
        }
        return names.get(module, module)

    def _extract_warnings(
        self,
        scores: Dict[str, float],
        regime: str,
        vix: float
    ) -> List[str]:
        """提取警告"""
        warnings = []

        # 冲突信号
        equity = scores.get("equity", 0)
        breadth = scores.get("breadth", 0)

        if regime == "Risk-On":
            if breadth < -0.5:
                warnings.append("⚠️ 广度背离：指数上涨但广度恶化，可能仅由少数权重股拉动")
            if vix > 25:
                warnings.append(f"⚠️ VIX偏高：当前{vix:.0f}处于紧张区间")
        elif regime == "Risk-Off":
            if equity > 0:
                warnings.append("⚠️ 表面强势：权益指数仍在均线上方，存在技术支撑")
        else:
            pass

        # BTC 暴跌
        btc = scores.get("btc", 0)
        if btc < -1:
            warnings.append(f"⚠️ BTC大幅下跌：风险偏好指标走弱")

        # 收益率曲线
        credit = scores.get("credit", 0)
        if credit < -1:
            warnings.append("⚠️ 收益率曲线倒挂：经济衰退预警信号")

        return warnings

    def _get_vix_signal(self, vix: float) -> str:
        """VIX 信号描述"""
        if vix < 15:
            return "calm"
        elif vix < 25:
            return "elevated"
        elif vix < 35:
            return "fear"
        else:
            return "panic"

    def _calculate_module_bias(
        self,
        index_results: List[IndexAnalysis],
        haven_results: List[SafeHavenAnalysis],
        credit_result: Optional[CreditAnalysis],
        breadth_result: Optional[BreadthAnalysis],
        vix: float,
        norm_scores: Dict[str, float]
    ) -> List[ModuleBias]:
        """计算各模块的多空倾向"""
        biases = []

        # 1. 权益指数
        bull = sum(1 for r in index_results if r.trend == "bull")
        bear = sum(1 for r in index_results if r.trend == "bear")
        if bull > bear + 1:
            biases.append(ModuleBias("权益指数", "多", 0.8, f"{bull}牛{bear}熊，多头占优"))
        elif bear > bull:
            biases.append(ModuleBias("权益指数", "空", 0.8, f"{bull}牛{bear}熊，空头占优"))
        else:
            biases.append(ModuleBias("权益指数", "中性", 0.3, f"{bull}牛{bear}熊，均衡"))

        # 2. 市场广度
        if breadth_result:
            if breadth_result.breadth_signal == "healthy":
                biases.append(ModuleBias("市场广度", "多", 0.9, "广度健康，大部分股票在涨"))
            elif breadth_result.breadth_signal in ("weak", "cap_driven"):
                biases.append(ModuleBias("市场广度", "空", 0.7, "广度背离，少数权重股拉动"))
            else:
                biases.append(ModuleBias("市场广度", "中性", 0.4, "广度中性"))
            # 趋势质量
            if breadth_result.trend_quality == "5star":
                biases.append(ModuleBias("趋势质量", "多", 0.9, "⭐⭐⭐⭐⭐ 最佳做多点"))
            elif breadth_result.trend_quality == "1star":
                biases.append(ModuleBias("趋势质量", "空", 0.7, "⭐ 假突破或崩盘风险"))
            else:
                biases.append(ModuleBias("趋势质量", "中性", 0.5, f"{breadth_result.trend_quality} 一般"))
        else:
            biases.append(ModuleBias("市场广度", "中性", 0.0, "数据缺失"))

        # 3. VIX/波动率
        if vix < 15:
            biases.append(ModuleBias("波动率(VIX)", "多", 0.7, f"VIX={vix:.1f} 平静，利于风险资产"))
        elif vix < 25:
            biases.append(ModuleBias("波动率(VIX)", "中性", 0.4, f"VIX={vix:.1f} 正常"))
        else:
            biases.append(ModuleBias("波动率(VIX)", "空", 0.8, f"VIX={vix:.1f} 恐慌，避险"))

        # 4. 信用/流动性
        if credit_result:
            if credit_result.credit_score > 0.5:
                biases.append(ModuleBias("信用/流动性", "多", 0.7, "高收益债跑赢，信用偏好强"))
            elif credit_result.credit_score < -0.5:
                biases.append(ModuleBias("信用/流动性", "空", 0.7, "信用收缩，曲线倒挂"))
            else:
                biases.append(ModuleBias("信用/流动性", "中性", 0.4, "信用中性"))
        else:
            biases.append(ModuleBias("信用/流动性", "中性", 0.0, "数据缺失"))

        # 5. 避险资产
        tlt = next((r for r in haven_results if r.ticker == "TLT"), None)
        gld = next((r for r in haven_results if r.ticker == "GLD"), None)
        if tlt and tlt.trend == "bull":
            biases.append(ModuleBias("避险资产", "空", 0.6, "债券涨=风险偏好弱"))
        elif gld and gld.trend == "bull":
            biases.append(ModuleBias("避险资产", "空", 0.5, "黄金涨=避险情绪"))
        else:
            biases.append(ModuleBias("避险资产", "多", 0.5, "避险资产弱=风险偏好强"))

        # 6. 美元
        uup = next((r for r in haven_results if r.ticker == "UUP"), None)
        if uup and uup.ret_20d > 0:
            biases.append(ModuleBias("美元", "空", 0.5, "美元强=压制风险资产"))
        else:
            biases.append(ModuleBias("美元", "多", 0.4, "美元弱=利好风险资产"))

        # 7. BTC
        btc = next((r for r in haven_results if r.ticker == "BTC-USD"), None)
        if btc and btc.ret_20d > 0:
            biases.append(ModuleBias("比特币", "多", 0.5, "BTC涨=风险偏好强"))
        elif btc and btc.ret_20d < -5:
            biases.append(ModuleBias("比特币", "空", 0.6, "BTC暴跌=风险偏好弱"))
        else:
            biases.append(ModuleBias("比特币", "中性", 0.3, "BTC震荡"))

        return biases

    def _forecast_outlook(
        self,
        regime: str,
        score: int,
        norm_scores: Dict[str, float],
        module_bias: List[ModuleBias],
        vix: float,
        breadth_result: Optional[BreadthAnalysis]
    ) -> Tuple[str, str, float, float, str]:
        """预测未来5天/30天市场倾向"""
        # 统计多空倾向
        bull_modules = [b for b in module_bias if b.bias == "多"]
        bear_modules = [b for b in module_bias if b.bias == "空"]
        bull_strength = sum(b.strength for b in bull_modules)
        bear_strength = sum(b.strength for b in bear_modules)

        # 短期(5天)：更敏感，权重给 VIX、广度、BTC
        short_signals = []
        if vix > 30:
            short_signals.append("panic")
        elif vix < 15:
            short_signals.append("calm")
        if breadth_result:
            if breadth_result.std_phase == "expansion_extreme":
                short_signals.append("mean_reversion")
            elif breadth_result.std_phase == "contraction":
                short_signals.append("breakout_soon")
        if norm_scores.get("btc", 0) < -1:
            short_signals.append("risk_off")

        # 5天预测
        if "panic" in short_signals or bear_strength > bull_strength + 1.0:
            forecast_5d = "空"
            fc_conf_5d = min(0.9, bear_strength / max(bull_strength + bear_strength, 1))
        elif "mean_reversion" in short_signals and score > 7:
            forecast_5d = "空"  # 极端后的回调
            fc_conf_5d = 0.6
        elif bull_strength > bear_strength + 0.5:
            forecast_5d = "多"
            fc_conf_5d = min(0.85, bull_strength / max(bull_strength + bear_strength, 1))
        else:
            forecast_5d = "中性"
            fc_conf_5d = 0.4

        # 长期(30天)：更看趋势、信用、宏观结构
        long_signals = []
        if score >= 7 and norm_scores.get("credit", 0) > 0:
            long_signals.append("strong_bull")
        if score <= 4 and norm_scores.get("credit", 0) < 0:
            long_signals.append("strong_bear")
        if breadth_result and breadth_result.breadth_signal == "weak":
            long_signals.append("breadth_divergence")
        if norm_scores.get("safe_haven", 0) < -0.5:
            long_signals.append("risk_off_building")

        if "strong_bull" in long_signals and "breadth_divergence" not in long_signals:
            forecast_30d = "多"
            fc_conf_30d = 0.75
        elif "strong_bear" in long_signals or "risk_off_building" in long_signals:
            forecast_30d = "空"
            fc_conf_30d = 0.7
        elif "breadth_divergence" in long_signals and score > 6:
            forecast_30d = "空"  # 顶背离
            fc_conf_30d = 0.6
        else:
            forecast_30d = "中性"
            fc_conf_30d = 0.4

        reason = f"短期: {', '.join(short_signals) if short_signals else '信号均衡'}; 长期: {', '.join(long_signals) if long_signals else '趋势延续'}"
        return forecast_5d, forecast_30d, round(fc_conf_5d, 2), round(fc_conf_30d, 2), reason

    def _generate_summary(
        self,
        regime: str,
        score: int,
        confidence: str,
        drivers: List[str],
        warnings: List[str],
        module_bias: List[ModuleBias] = None,
        forecast_5d: str = "中性",
        forecast_30d: str = "中性"
    ) -> str:
        """生成摘要 v2 - 增强版（含模块倾向 + 未来预测）"""
        regime_desc = {
            "Risk-On": "市场风险偏好积极，资金流向风险资产",
            "Risk-Off": "市场风险偏好低迷，资金涌入避险资产",
            "Neutral": "多空力量均衡，市场方向不明"
        }

        lines = [
            f"📊 宏观全景扫描报告 v2",
            f"",
            f"**市场状态**: {regime}",
            f"**环境评分**: {score}/10",
            f"**置信度**: {confidence}",
            f"**主导因子**: {', '.join(drivers[:3])}",
        ]

        # 各模块多空倾向 — Markdown 表格
        if module_bias:
            lines.append("")
            lines.append("**各模块倾向**:")
            lines.append("")
            lines.append("| 模块 | 倾向 | 强度 | 说明 |")
            lines.append("|------|------|------|------|")
            for b in module_bias:
                emoji = {"多": "🟢", "空": "🔴", "中性": "⚪"}.get(b.bias, "⚪")
                lines.append(f"| {emoji} {b.name} | {b.bias} | {b.strength:.0%} | {b.detail} |")

        # 未来预测 — Markdown 表格
        lines.append("")
        lines.append("**未来展望**:")
        lines.append("")
        lines.append("| 时间维度 | 倾向 | 图标 |")
        lines.append("|----------|------|------|")
        fc_emoji_5d = {"多": "📈", "空": "📉", "中性": "➡️"}.get(forecast_5d, "➡️")
        fc_emoji_30d = {"多": "📈", "空": "📉", "中性": "➡️"}.get(forecast_30d, "➡️")
        lines.append(f"| 未来5天  | {forecast_5d} | {fc_emoji_5d} |")
        lines.append(f"| 未来30天 | {forecast_30d} | {fc_emoji_30d} |")

        if warnings:
            lines.append(f"")
            lines.append(f"**风险预警**:")
            for w in warnings[:3]:
                lines.append(f"- {w}")

        lines.extend([
            f"",
            f"{regime_desc.get(regime, '')}",
        ])

        return "\n".join(lines)

    # ---- 便捷方法 ----

    def get_index_summary(self) -> List[Dict]:
        """获取指数分析摘要"""
        result = self.scan()
        return [
            {
                "ticker": r.ticker,
                "name": r.name,
                "trend": r.trend,
                "trend_pct": r.trend_pct,
                "current_price": r.current_price,
                "ret_5d": r.ret_5d,
                "ret_20d": r.ret_20d,
                "ret_60d": r.ret_60d,
                "volume_ratio": r.volume_ratio,
            }
            for r in result.index_results
        ]

    def get_haven_summary(self) -> List[Dict]:
        """获取避险资产摘要"""
        result = self.scan()
        return [
            {
                "ticker": r.ticker,
                "name": r.name,
                "trend": r.trend,
                "trend_pct": r.trend_pct,
                "ret_20d": r.ret_20d,
                "is_abnormal": r.is_abnormal,
            }
            for r in result.haven_results
        ]
