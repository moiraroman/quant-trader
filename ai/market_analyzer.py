# ============================================================
# ai/market_analyzer.py — 市场分析模块
# 自动识别市场状态，推荐策略类型
# ============================================================
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 数据类
# ============================================================

@dataclass
class AnalysisStep:
    """分析步骤"""
    step_name: str
    input_data: str
    calculation: str
    result: str
    reasoning: str


@dataclass
class MarketState:
    """市场状态"""
    trend: str  # "bull" | "bear" | "sideways"
    volatility: str  # "high" | "medium" | "low"
    momentum: str  # "strong" | "weak"
    risk_level: int  # 1-10
    confidence: float  # 0.0-1.0
    recommended_strategies: List[str]
    analysis_text: str
    analysis_steps: List[AnalysisStep]  # 详细分析步骤
    raw_data: Dict  # 原始数据
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# ============================================================
# 市场分析器
# ============================================================

class MarketAnalyzer:
    """
    市场分析器
    
    分析大盘指数（SPY/QQQ），识别市场状态。
    """
    
    def __init__(self, data_fetcher=None):
        """
        Args:
            data_fetcher: 数据获取器（YFinanceFetcher 实例）
        """
        self.fetcher = data_fetcher
        
        # 索引 ETF 列表（用于判断市场状态）
        self.index_tickers = ["SPY", "QQQ", "IWM", "DIA"]
        
        # 策略推荐规则
        self.strategy_rules = {
            "bull_low_vol": ["MACD", "MA_cross", "Breakout"],
            "bull_high_vol": ["RSI", "Bollinger", "Mean_Reversion"],
            "bear_low_vol": ["Bear_MACD", "Short_MA"],
            "bear_high_vol": ["Cash", "Defensive"],
            "sideways_low_vol": ["RSI", "Bollinger", "Grid"],
            "sideways_high_vol": ["Defensive", "Reduce_Size"],
        }
    
    def analyze(self, period_days: int = 100) -> MarketState:
        """
        分析当前市场状态
        
        Args:
            period_days: 分析周期（天数）
        
        Returns:
            MarketState 实例
        """
        # 获取 SPY 数据作为主要参考
        if self.fetcher is None:
            from data.fetcher import YFinanceFetcher
            self.fetcher = YFinanceFetcher()
        
        try:
            df = self.fetcher.download_history("SPY", period=f"{period_days}d")
            if df.empty:
                return self._default_state()
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return self._default_state()
        
        # 记录分析步骤
        analysis_steps = []
        
        # 计算各项指标
        trend, trend_steps = self._analyze_trend(df)
        analysis_steps.extend(trend_steps)
        
        volatility, vol_steps = self._analyze_volatility(df)
        analysis_steps.extend(vol_steps)
        
        momentum, mom_steps = self._analyze_momentum(df)
        analysis_steps.extend(mom_steps)
        
        # 计算风险等级
        risk_level = self._calculate_risk_level(trend, volatility, momentum)
        
        # 推荐策略
        recommended = self._recommend_strategies(trend, volatility)
        
        # 生成分析文本
        analysis_text = self._generate_analysis_text(
            trend, volatility, momentum, risk_level, recommended
        )
        
        # 收集原始数据
        raw_data = {
            "last_price": df["Close"].iloc[-1],
            "ma50": df["Close"].rolling(50).mean().iloc[-1],
            "ma200": df["Close"].rolling(200).mean().iloc[-1],
            "volume": df["Volume"].iloc[-1],
            "avg_volume": df["Volume"].rolling(20).mean().iloc[-1],
            "period_days": period_days,
            "data_points": len(df)
        }
        
        return MarketState(
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            risk_level=risk_level,
            confidence=0.75,  # 可根据历史准确率调整
            recommended_strategies=recommended,
            analysis_text=analysis_text,
            analysis_steps=analysis_steps,
            raw_data=raw_data
        )
    
    def _analyze_trend(self, df: pd.DataFrame) -> Tuple[str, List[AnalysisStep]]:
        """分析趋势"""
        steps = []
        close = df["Close"]
        
        # 步骤1: 计算移动平均线
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        
        current_price = close.iloc[-1]
        current_ma50 = ma50.iloc[-1]
        current_ma200 = ma200.iloc[-1] if not pd.isna(ma200.iloc[-1]) else current_ma50
        
        steps.append(AnalysisStep(
            step_name="计算移动平均线",
            input_data=f"收盘价序列 (n={len(close)})",
            calculation=f"MA50={current_ma50:.2f}, MA200={current_ma200:.2f}, 当前价格={current_price:.2f}",
            result=f"价格 {'>' if current_price > current_ma50 else '<'} MA50 {'>' if current_ma50 > current_ma200 else '<'} MA200",
            reasoning="通过价格与均线的位置关系判断趋势方向"
        ))
        
        # 步骤2: 初步趋势判断
        if current_price > current_ma50 > current_ma200:
            trend = "bull"
            reasoning = "价格 > MA50 > MA200，典型的多头排列，确认上升趋势"
        elif current_price < current_ma50 < current_ma200:
            trend = "bear"
            reasoning = "价格 < MA50 < MA200，典型的空头排列，确认下降趋势"
        else:
            # 步骤3: 使用 ADX 判断趋势强度
            adx = self._calculate_adx(df)
            
            steps.append(AnalysisStep(
                step_name="计算 ADX 趋势强度",
                input_data="最高价、最低价、收盘价",
                calculation=f"ADX = {adx:.1f}",
                result=f"趋势强度: {'强' if adx > 25 else '弱'}",
                reasoning="ADX > 25 表示趋势明显，< 25 表示震荡"
            ))
            
            if adx > 25:
                trend = "bull" if current_price > current_ma50 else "bear"
                reasoning = f"ADX={adx:.1f} > 25，趋势明显，根据价格与MA50关系判断为{trend}"
            else:
                trend = "sideways"
                reasoning = f"ADX={adx:.1f} <= 25，趋势不明显，判定为横盘震荡"
        
        steps.append(AnalysisStep(
            step_name="趋势判定",
            input_data="均线位置 + ADX强度",
            calculation=f"趋势 = {trend}",
            result=trend,
            reasoning=reasoning
        ))
        
        return trend, steps
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Tuple[str, List[AnalysisStep]]:
        """分析波动率"""
        steps = []
        
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        # 计算 ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        atr_pct = (atr / close).iloc[-1] * 100
        
        steps.append(AnalysisStep(
            step_name="计算 ATR 波动率",
            input_data="最高价、最低价、收盘价",
            calculation=f"TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)\nATR(14) = {atr.iloc[-1]:.2f}\nATR% = {atr_pct:.2f}%",
            result=f"ATR百分比 = {atr_pct:.2f}%",
            reasoning="ATR% 反映价格的平均波动幅度"
        ))
        
        # 波动率等级
        if atr_pct > 3:
            volatility = "high"
            reasoning = f"ATR%={atr_pct:.2f}% > 3%，高波动市场"
        elif atr_pct > 1.5:
            volatility = "medium"
            reasoning = f"1.5% < ATR%={atr_pct:.2f}% <= 3%，中等波动"
        else:
            volatility = "low"
            reasoning = f"ATR%={atr_pct:.2f}% <= 1.5%，低波动市场"
        
        steps.append(AnalysisStep(
            step_name="波动率分级",
            input_data=f"ATR% = {atr_pct:.2f}%",
            calculation=f"阈值: 高>3%, 中>1.5%, 低<=1.5%",
            result=volatility,
            reasoning=reasoning
        ))
        
        return volatility, steps
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Tuple[str, List[AnalysisStep]]:
        """分析动量"""
        steps = []
        close = df["Close"]
        
        # 步骤1: 计算 RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        steps.append(AnalysisStep(
            step_name="计算 RSI 动量",
            input_data="收盘价变化",
            calculation=f"RSI(14) = {current_rsi:.1f}",
            result=f"RSI = {current_rsi:.1f} ({'超买' if current_rsi > 70 else '超卖' if current_rsi < 30 else '中性'})",
            reasoning="RSI > 70 超买，< 30 超卖，30-70 中性"
        ))
        
        # 步骤2: 计算 MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_hist = macd - signal
        current_hist = macd_hist.iloc[-1]
        
        steps.append(AnalysisStep(
            step_name="计算 MACD 动量",
            input_data="收盘价",
            calculation=f"EMA12={ema12.iloc[-1]:.2f}, EMA26={ema26.iloc[-1]:.2f}\nMACD={macd.iloc[-1]:.2f}, Signal={signal.iloc[-1]:.2f}\nHistogram={current_hist:.2f}",
            result=f"Histogram = {current_hist:.2f} ({'正' if current_hist > 0 else '负'})",
            reasoning="Histogram > 0 表示上涨动能，< 0 表示下跌动能"
        ))
        
        # 步骤3: 综合判断
        if current_rsi > 60 and current_hist > 0:
            momentum = "strong"
            reasoning = f"RSI={current_rsi:.1f} > 60 且 MACD Histogram={current_hist:.2f} > 0，双重确认强势"
        elif current_rsi < 40 and current_hist < 0:
            momentum = "weak"
            reasoning = f"RSI={current_rsi:.1f} < 40 且 MACD Histogram={current_hist:.2f} < 0，双重确认弱势"
        else:
            momentum = "neutral"
            reasoning = f"RSI={current_rsi:.1f} 和 MACD Histogram={current_hist:.2f} 信号不一致，判定为中性"
        
        steps.append(AnalysisStep(
            step_name="动量综合判定",
            input_data=f"RSI={current_rsi:.1f}, MACD Hist={current_hist:.2f}",
            calculation="RSI>60 & MACD>0 → 强\nRSI<40 & MACD<0 → 弱\n其他 → 中性",
            result=momentum,
            reasoning=reasoning
        ))
        
        return momentum, steps
    
    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """计算 ADX"""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm = plus_dm.where(plus_dm > minus_dm.abs(), 0)
        minus_dm = minus_dm.where(minus_dm.abs() > plus_dm, 0).abs()
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
    
    def _calculate_risk_level(
        self,
        trend: str,
        volatility: str,
        momentum: str
    ) -> int:
        """计算风险等级（1-10）"""
        score = 5  # 基准
        
        # 趋势影响
        if trend == "bull":
            score -= 1
        elif trend == "bear":
            score += 2
        
        # 波动率影响
        if volatility == "high":
            score += 2
        elif volatility == "low":
            score -= 1
        
        # 动量影响
        if momentum == "weak":
            score += 1
        elif momentum == "strong":
            score -= 0.5
        
        return max(1, min(10, int(round(score))))
    
    def _recommend_strategies(
        self,
        trend: str,
        volatility: str
    ) -> List[str]:
        """推荐策略"""
        key = f"{trend}_{volatility}"
        return self.strategy_rules.get(key, ["MACD", "RSI"])
    
    def _generate_analysis_text(
        self,
        trend: str,
        volatility: str,
        momentum: str,
        risk_level: int,
        recommended: List[str]
    ) -> str:
        """生成分析文本"""
        trend_cn = {"bull": "上升趋势", "bear": "下降趋势", "sideways": "横盘震荡"}
        vol_cn = {"high": "高波动", "medium": "中等波动", "low": "低波动"}
        mom_cn = {"strong": "强势", "weak": "弱势", "neutral": "中性"}
        
        text = f"""
当前市场状态：{trend_cn.get(trend, '未知')}
波动率：{vol_cn.get(volatility, '未知')}
动量：{mom_cn.get(momentum, '未知')}
风险等级：{risk_level}/10

推荐策略：{', '.join(recommended)}
"""
        return text.strip()
    
    def _default_state(self) -> MarketState:
        """返回默认状态"""
        return MarketState(
            trend="sideways",
            volatility="medium",
            momentum="neutral",
            risk_level=5,
            confidence=0.0,
            recommended_strategies=["MACD", "RSI"],
            analysis_text="无法获取市场数据，使用默认状态",
            analysis_steps=[],
            raw_data={}
        )
    
    def get_market_summary(self) -> Dict:
        """获取市场摘要（用于 WebUI）"""
        state = self.analyze()
        
        return {
            "trend": state.trend,
            "volatility": state.volatility,
            "momentum": state.momentum,
            "risk_level": state.risk_level,
            "confidence": state.confidence,
            "recommended_strategies": state.recommended_strategies,
            "analysis": state.analysis_text,
            "analysis_steps": [
                {
                    "step_name": s.step_name,
                    "input_data": s.input_data,
                    "calculation": s.calculation,
                    "result": s.result,
                    "reasoning": s.reasoning
                }
                for s in state.analysis_steps
            ],
            "raw_data": state.raw_data,
            "timestamp": state.timestamp.isoformat()
        }
