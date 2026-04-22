# ============================================================
# ai/stock_screener.py — 多因子选股模块
# 从股票池中筛选机会（支持配置化参数）
# ============================================================
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ============================================================
# 数据类
# ============================================================

@dataclass
class FactorAnalysis:
    """因子分析详情"""
    factor_name: str
    weight: float
    raw_value: float
    normalized_score: float  # 0-1
    calculation_steps: str
    interpretation: str


@dataclass
class StockCandidate:
    """股票候选"""
    ticker: str
    score: float  # 综合得分 0-100
    factors: Dict[str, float]  # 各因子得分
    factor_details: List[FactorAnalysis]  # 因子分析详情
    signal: str  # BUY / SELL / HOLD
    price: float
    change_pct: float
    volume_ratio: float  # 量比
    rsi: float
    ma_deviation: float  # MA 偏离度
    atr_pct: float  # ATR 百分比
    reason: str
    signal_reasoning: str  # 信号判断逻辑
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# ============================================================
# 选股器
# ============================================================

class StockScreener:
    """
    多因子选股器（配置化版本）
    
    从股票池中筛选交易机会，支持从配置文件加载参数。
    """
    
    # 预设股票池
    STOCK_POOLS = {
        "sp500": [],  # S&P 500（动态加载）
        "nasdaq100": [],  # NASDAQ 100
        "dow30": [],  # 道琼斯 30
        "popular": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ"],
        "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM"],
        "etf": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "GLD", "TLT"],
    }
    
    # 默认配置
    DEFAULT_CONFIG = {
        "factor_weights": {
            "momentum": 0.25,
            "trend": 0.20,
            "volume": 0.15,
            "volatility": 0.15,
            "rsi": 0.15,
            "ma_deviation": 0.10,
        },
        "thresholds": {
            "rsi_oversold": 35,
            "rsi_overbought": 70,
            "trend_bull": 0.6,
            "trend_bear": 0.4,
            "momentum_strong": 0.7,
            "momentum_weak": 0.3,
            "min_score_buy": 70,
            "max_score_sell": 40,
        },
        "params": {
            "momentum_short_period": 5,
            "momentum_long_period": 20,
            "volume_avg_period": 20,
            "rsi_period": 14,
            "atr_period": 14,
            "ma_short_period": 20,
            "ma_long_period": 50,
        }
    }
    
    def __init__(self, data_fetcher=None, config_path: str = None):
        """
        Args:
            data_fetcher: 数据获取器
            config_path: 配置文件路径，默认使用项目根目录的 config.yaml
        """
        self.fetcher = data_fetcher
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.factor_weights = self.config["factor_weights"]
        self.thresholds = self.config["thresholds"]
        self.params = self.config["params"]
        
        logger.info(f"[Screener] 配置加载完成: 因子权重={self.factor_weights}")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """从配置文件加载选股参数"""
        if config_path is None:
            # 默认路径：项目根目录
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config.yaml"
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            
            screening_config = full_config.get("screening", {})
            
            # 合并配置
            config = {
                "factor_weights": screening_config.get("factor_weights", self.DEFAULT_CONFIG["factor_weights"]),
                "thresholds": screening_config.get("thresholds", self.DEFAULT_CONFIG["thresholds"]),
                "params": screening_config.get("params", self.DEFAULT_CONFIG["params"]),
            }
            
            return config
            
        except Exception as e:
            logger.warning(f"[Screener] 加载配置失败，使用默认配置: {e}")
            return self.DEFAULT_CONFIG.copy()
    
    def update_config(self, **kwargs):
        """
        动态更新配置参数
        
        示例:
            screener.update_config(
                factor_weights={"momentum": 0.3, "trend": 0.3},
                thresholds={"rsi_oversold": 30}
            )
        """
        if "factor_weights" in kwargs:
            self.factor_weights.update(kwargs["factor_weights"])
        if "thresholds" in kwargs:
            self.thresholds.update(kwargs["thresholds"])
        if "params" in kwargs:
            self.params.update(kwargs["params"])
        
        logger.info(f"[Screener] 配置已更新: {kwargs}")
    
    def get_config(self) -> Dict:
        """获取当前配置"""
        return {
            "factor_weights": self.factor_weights.copy(),
            "thresholds": self.thresholds.copy(),
            "params": self.params.copy(),
        }
    
    def screen(
        self,
        tickers: Optional[List[str]] = None,
        pool_name: str = "popular",
        top_n: int = 10,
        min_score: float = 50.0,
        signal_filter: Optional[str] = None
    ) -> List[StockCandidate]:
        """
        筛选股票
        
        Args:
            tickers: 股票列表（优先于 pool_name）
            pool_name: 股票池名称
            top_n: 返回前 N 个
            min_score: 最低得分
            signal_filter: 信号过滤（BUY/SELL/HOLD）
        
        Returns:
            StockCandidate 列表
        """
        # 确定股票列表
        if tickers is None or len(tickers) == 0:
            tickers = self.STOCK_POOLS.get(pool_name, self.STOCK_POOLS["popular"])
        
        if len(tickers) == 0:
            logger.warning("股票列表为空")
            return []
        
        # 初始化数据获取器
        if self.fetcher is None:
            from data.fetcher import YFinanceFetcher
            self.fetcher = YFinanceFetcher()
        
        # 并行处理
        candidates = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._analyze_single, ticker): ticker
                for ticker in tickers
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    candidate = future.result()
                    if candidate and candidate.score >= min_score:
                        if signal_filter is None or candidate.signal == signal_filter:
                            candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"{ticker} 分析失败: {e}")
        
        # 排序
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:top_n]
    
    def _analyze_single(self, ticker: str) -> Optional[StockCandidate]:
        """分析单只股票"""
        try:
            df = self.fetcher.download_history(ticker, period="60d")
            if df.empty or len(df) < 30:
                return None
            
            # 计算各因子（带详细步骤）
            factors, factor_details = self._calculate_factors_with_details(df)
            
            # 计算综合得分
            score = sum(
                factors[f] * self.factor_weights.get(f, 0)
                for f in factors
            )
            
            # 标准化到 0-100
            score = max(0, min(100, score * 100))
            
            # 判断信号
            signal, signal_reasoning = self._determine_signal_with_reasoning(factors, score)
            
            # 生成原因
            reason = self._generate_reason(factors, signal)
            
            # 当前价格和变化
            current_price = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2] if len(df) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # 成交量比
            avg_volume = df["Volume"].rolling(self.params["volume_avg_period"]).mean().iloc[-1]
            current_volume = df["Volume"].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            return StockCandidate(
                ticker=ticker,
                score=score,
                factors=factors,
                factor_details=factor_details,
                signal=signal,
                price=current_price,
                change_pct=change_pct,
                volume_ratio=volume_ratio,
                rsi=factors.get("rsi", 50) * 100,
                ma_deviation=factors.get("ma_deviation", 0),
                atr_pct=factors.get("volatility", 0),
                reason=reason,
                signal_reasoning=signal_reasoning
            )
            
        except Exception as e:
            logger.error(f"{ticker} 分析异常: {e}")
            return None
    
    def _calculate_factors_with_details(self, df: pd.DataFrame) -> Tuple[Dict[str, float], List[FactorAnalysis]]:
        """计算各因子得分（带详细步骤，使用配置参数）"""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        
        factors = {}
        factor_details = []
        
        p = self.params  # 参数简写
        
        # 1. 动量因子（近期涨幅）
        returns_short = close.pct_change(p["momentum_short_period"]).iloc[-1]
        returns_long = close.pct_change(p["momentum_long_period"]).iloc[-1]
        momentum_score = 0.5 * (1 if returns_short > 0 else 0) + 0.5 * (1 if returns_long > 0 else 0)
        factors["momentum"] = momentum_score
        
        factor_details.append(FactorAnalysis(
            factor_name="动量因子",
            weight=self.factor_weights["momentum"],
            raw_value=returns_short,
            normalized_score=momentum_score,
            calculation_steps=f"{p['momentum_short_period']}日涨幅: {returns_short:.2%}\n{p['momentum_long_period']}日涨幅: {returns_long:.2%}\n得分 = 0.5×(短期>0) + 0.5×(长期>0) = {momentum_score:.2f}",
            interpretation="近期上涨得高分，下跌得低分，平衡短期和中期趋势"
        ))
        
        # 2. 趋势因子（MA 位置）
        ma_short = close.rolling(p["ma_short_period"]).mean().iloc[-1]
        ma_long = close.rolling(p["ma_long_period"]).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        trend_score = 0
        if current_price > ma_short:
            trend_score += 0.5
        if ma_short > ma_long:
            trend_score += 0.3
        if current_price > ma_long:
            trend_score += 0.2
        factors["trend"] = trend_score
        
        factor_details.append(FactorAnalysis(
            factor_name="趋势因子",
            weight=self.factor_weights["trend"],
            raw_value=current_price / ma_short - 1,
            normalized_score=trend_score,
            calculation_steps=f"价格={current_price:.2f}, MA{p['ma_short_period']}={ma_short:.2f}, MA{p['ma_long_period']}={ma_long:.2f}\n价格>MA短: {current_price > ma_short} (+0.5)\nMA短>MA长: {ma_short > ma_long} (+0.3)\n价格>MA长: {current_price > ma_long} (+0.2)\n总分 = {trend_score:.2f}",
            interpretation="多头排列得高分，价格站上均线越多得分越高"
        ))
        
        # 3. 成交量因子（放量）
        avg_volume = volume.rolling(p["volume_avg_period"]).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
        volume_score = min(1, volume_ratio / 2)  # 量比 2x = 满分
        factors["volume"] = volume_score
        
        factor_details.append(FactorAnalysis(
            factor_name="成交量因子",
            weight=self.factor_weights["volume"],
            raw_value=volume_ratio,
            normalized_score=volume_score,
            calculation_steps=f"当前成交量: {current_volume:,.0f}\n{p['volume_avg_period']}日均量: {avg_volume.iloc[-1]:,.0f}\n量比 = {volume_ratio:.2f}\n得分 = min(1, 量比/2) = {volume_score:.2f}",
            interpretation="放量表示资金关注，量比2倍以上得满分"
        ))
        
        # 4. 波动因子（ATR 百分比）
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(p["atr_period"]).mean().iloc[-1]
        atr_pct = (atr / current_price) * 100
        volatility_score = 1 - min(1, atr_pct / 5)  # 波动越小越好（稳健）
        factors["volatility"] = volatility_score
        
        factor_details.append(FactorAnalysis(
            factor_name="波动因子",
            weight=self.factor_weights["volatility"],
            raw_value=atr_pct,
            normalized_score=volatility_score,
            calculation_steps=f"ATR({p['atr_period']}) = {atr:.2f}\nATR% = {atr_pct:.2f}%\n得分 = 1 - min(1, ATR%/5) = {volatility_score:.2f}",
            interpretation="波动率越低得分越高，追求稳健性，ATR%5%以上得0分"
        ))
        
        # 5. RSI 因子
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(p["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p["rsi_period"]).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        rsi_score = rsi / 100  # 标准化
        factors["rsi"] = rsi_score
        
        factor_details.append(FactorAnalysis(
            factor_name="RSI因子",
            weight=self.factor_weights["rsi"],
            raw_value=rsi,
            normalized_score=rsi_score,
            calculation_steps=f"RSI({p['rsi_period']}) = {rsi:.1f}\n得分 = RSI/100 = {rsi_score:.2f}",
            interpretation="RSI越高得分越高，但>70超买需警惕，<30超卖可能反弹"
        ))
        
        # 6. MA 偏离因子
        ma_deviation = (current_price - ma_short) / ma_short
        ma_dev_score = 0.5 - min(0.5, max(-0.5, ma_deviation)) + 0.5  # 偏离度适中为好
        factors["ma_deviation"] = ma_dev_score
        
        factor_details.append(FactorAnalysis(
            factor_name="MA偏离因子",
            weight=self.factor_weights["ma_deviation"],
            raw_value=ma_deviation,
            normalized_score=ma_dev_score,
            calculation_steps=f"MA偏离 = (价格-MA{p['ma_short_period']})/MA{p['ma_short_period']} = {ma_deviation:.2%}\n得分 = 0.5 - clamp(偏离, -0.5, 0.5) + 0.5 = {ma_dev_score:.2f}",
            interpretation="价格偏离MA适中最好，偏离过大可能回调，偏离过小可能横盘"
        ))
        
        return factors, factor_details
    
    def _determine_signal_with_reasoning(self, factors: Dict[str, float], score: float) -> Tuple[str, str]:
        """判断信号（带推理过程，使用配置阈值）"""
        rsi = factors.get("rsi", 0.5) * 100
        trend = factors.get("trend", 0.5)
        momentum = factors.get("momentum", 0.5)
        
        t = self.thresholds  # 阈值简写
        
        reasoning_parts = []
        
        # 超卖 + 趋势向上 = BUY
        if rsi < t["rsi_oversold"] and trend > t["trend_bull"]:
            reasoning_parts.append(f"RSI={rsi:.1f} < {t['rsi_oversold']}（超卖）且 趋势得分={trend:.2f} > {t['trend_bull']}（向上）")
            reasoning_parts.append("→ 超卖反弹 + 趋势支撑 = 买入信号")
            return "BUY", "\n".join(reasoning_parts)
        
        # 超买 + 趋势向下 = SELL
        if rsi > t["rsi_overbought"] and trend < t["trend_bear"]:
            reasoning_parts.append(f"RSI={rsi:.1f} > {t['rsi_overbought']}（超买）且 趋势得分={trend:.2f} < {t['trend_bear']}（向下）")
            reasoning_parts.append("→ 超买回调 + 趋势压制 = 卖出信号")
            return "SELL", "\n".join(reasoning_parts)
        
        # 动量强劲 + 得分高 = BUY
        if momentum > t["momentum_strong"] and score > t["min_score_buy"]:
            reasoning_parts.append(f"动量得分={momentum:.2f} > {t['momentum_strong']}（强劲）且 综合得分={score:.1f} > {t['min_score_buy']}（优秀）")
            reasoning_parts.append("→ 强劲动量 + 高评分 = 买入信号")
            return "BUY", "\n".join(reasoning_parts)
        
        # 动量弱 + 得分低 = SELL
        if momentum < t["momentum_weak"] and score < t["max_score_sell"]:
            reasoning_parts.append(f"动量得分={momentum:.2f} < {t['momentum_weak']}（弱势）且 综合得分={score:.1f} < {t['max_score_sell']}（较差）")
            reasoning_parts.append("→ 弱势动量 + 低评分 = 卖出信号")
            return "SELL", "\n".join(reasoning_parts)
        
        # 默认 HOLD
        reasoning_parts.append(f"RSI={rsi:.1f}（{'超卖' if rsi < t['rsi_oversold'] else '超买' if rsi > t['rsi_overbought'] else '中性'}）")
        reasoning_parts.append(f"趋势={trend:.2f}，动量={momentum:.2f}，综合得分={score:.1f}")
        reasoning_parts.append("→ 条件不满足买卖标准 = 持有观望")
        return "HOLD", "\n".join(reasoning_parts)
    
    def _generate_reason(self, factors: Dict[str, float], signal: str) -> str:
        """生成原因"""
        reasons = []
        
        rsi = factors.get("rsi", 0.5) * 100
        trend = factors.get("trend", 0.5)
        momentum = factors.get("momentum", 0.5)
        volume = factors.get("volume", 0.5)
        
        t = self.thresholds
        
        if rsi < t["rsi_oversold"]:
            reasons.append("RSI 超卖")
        elif rsi > t["rsi_overbought"]:
            reasons.append("RSI 超买")
        
        if trend > t["trend_bull"]:
            reasons.append("趋势向上")
        elif trend < t["trend_bear"]:
            reasons.append("趋势向下")
        
        if momentum > t["momentum_strong"]:
            reasons.append("动量强劲")
        elif momentum < t["momentum_weak"]:
            reasons.append("动量减弱")
        
        if volume > 0.8:
            reasons.append("放量")
        
        return " | ".join(reasons) if reasons else "综合指标"
    
    def get_top_buys(self, tickers: Optional[List[str]] = None, top_n: int = 5) -> List[StockCandidate]:
        """获取买入信号最多的股票"""
        return self.screen(tickers=tickers, top_n=top_n, signal_filter="BUY")
    
    def get_top_sells(self, tickers: Optional[List[str]] = None, top_n: int = 5) -> List[StockCandidate]:
        """获取卖出信号最多的股票"""
        return self.screen(tickers=tickers, top_n=top_n, signal_filter="SELL")
    
    def get_screening_summary(self, tickers: List[str]) -> Dict:
        """获取筛选摘要（用于 WebUI）"""
        candidates = self.screen(tickers=tickers, top_n=len(tickers))
        
        return {
            "total_analyzed": len(candidates),
            "buy_signals": len([c for c in candidates if c.signal == "BUY"]),
            "sell_signals": len([c for c in candidates if c.signal == "SELL"]),
            "hold_signals": len([c for c in candidates if c.signal == "HOLD"]),
            "config": self.get_config(),  # 包含当前配置
            "top_candidates": [
                {
                    "ticker": c.ticker,
                    "score": c.score,
                    "signal": c.signal,
                    "price": c.price,
                    "reason": c.reason,
                    "signal_reasoning": c.signal_reasoning,
                    "factor_details": [
                        {
                            "factor_name": f.factor_name,
                            "weight": f.weight,
                            "raw_value": f.raw_value,
                            "normalized_score": f.normalized_score,
                            "calculation_steps": f.calculation_steps,
                            "interpretation": f.interpretation
                        }
                        for f in c.factor_details
                    ]
                }
                for c in candidates[:10]
            ],
            "timestamp": datetime.now().isoformat()
        }
