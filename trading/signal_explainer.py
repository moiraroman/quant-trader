"""
trading/signal_explainer.py — 信号解释器

功能：
    - 记录每笔交易的信号来源和推理链
    - 生成人类可读的交易理由
    - 信号置信度分解
    - 交易决策审计追踪
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ============================================================
# 数据类定义
# ============================================================

@dataclass
class SignalFactor:
    """信号因子"""
    name: str
    weight: float           # 权重 0-1
    score: float            # 得分 -1 ~ +1
    confidence: float       # 置信度 0-1
    description: str        # 人类可读描述
    raw_value: Any = None   # 原始值

@dataclass
class SignalDecision:
    """信号决策"""
    timestamp: datetime
    ticker: str
    
    # 最终决策
    action: str             # BUY/SELL/HOLD
    confidence: float       # 0-1
    
    # 因子分解
    factors: List[SignalFactor] = field(default_factory=list)
    
    # 市场环境
    market_regime: str = "unknown"
    market_score: float = 0.0
    
    # 策略信息
    strategy_name: str = ""
    strategy_params: Dict = field(default_factory=dict)
    
    # 风控检查
    risk_checks: List[Dict] = field(default_factory=list)
    risk_passed: bool = True
    
    # 执行信息
    execution_plan: Dict = field(default_factory=dict)
    
    # 推理链
    reasoning_chain: List[str] = field(default_factory=list)

@dataclass
class TradeExplanation:
    """完整交易解释"""
    trade_id: str
    timestamp: datetime
    ticker: str
    action: str
    
    # 信号决策
    signal_decision: SignalDecision
    
    # 执行结果
    executed_price: float
    expected_price: float
    slippage: float
    
    # 对比分析
    alternative_actions: List[str] = field(default_factory=list)
    missed_opportunity: Optional[str] = None
    
    # 后验分析（交易完成后填充）
    outcome: Optional[str] = None  # win/loss
    pnl: Optional[float] = None
    hindsight_notes: List[str] = field(default_factory=list)

# ============================================================
# 信号解释器
# ============================================================

class SignalExplainer:
    """
    信号解释器：让每笔交易都有迹可循。
    
    功能：
    1. 记录信号因子分解
    2. 生成人类可读的交易理由
    3. 提供决策审计追踪
    4. 支持后验分析
    """
    
    def __init__(self):
        self.decisions: List[SignalDecision] = []
        self.explanations: Dict[str, TradeExplanation] = {}
    
    def record_signal(
        self,
        ticker: str,
        action: str,
        confidence: float,
        factors: List[SignalFactor],
        strategy_name: str = "",
        strategy_params: Dict = None,
        market_regime: str = "unknown",
        market_score: float = 0.0,
        risk_checks: List[Dict] = None,
    ) -> SignalDecision:
        """
        记录信号决策。
        
        示例：
            decision = explainer.record_signal(
                ticker="AAPL",
                action="BUY",
                confidence=0.78,
                factors=[
                    SignalFactor("MA Cross", 0.3, 0.8, 0.9, "50日均线上穿200日均线"),
                    SignalFactor("RSI", 0.2, 0.5, 0.7, "RSI=45，未超买超卖"),
                    SignalFactor("MACD", 0.25, 0.7, 0.8, "MACD柱状图转正"),
                    SignalFactor("Volume", 0.15, 0.6, 0.6, "成交量放大1.5倍"),
                    SignalFactor("Support", 0.1, 0.9, 0.8, "触及支撑位$150"),
                ],
                strategy_name="CompositeStrategy",
                market_regime="bullish",
            )
        """
        # 生成推理链
        reasoning = self._build_reasoning_chain(
            action, confidence, factors, market_regime, market_score
        )
        
        decision = SignalDecision(
            timestamp=datetime.now(),
            ticker=ticker,
            action=action,
            confidence=confidence,
            factors=factors,
            market_regime=market_regime,
            market_score=market_score,
            strategy_name=strategy_name,
            strategy_params=strategy_params or {},
            risk_checks=risk_checks or [],
            reasoning_chain=reasoning,
        )
        
        self.decisions.append(decision)
        
        logger.info(f"[SignalExplainer] {ticker} {action} 置信度={confidence:.2f} 因子数={len(factors)}")
        
        return decision
    
    def _build_reasoning_chain(
        self,
        action: str,
        confidence: float,
        factors: List[SignalFactor],
        market_regime: str,
        market_score: float,
    ) -> List[str]:
        """构建推理链"""
        reasoning = []
        
        # 市场环境
        reasoning.append(f"市场环境: {market_regime} (评分: {market_score:.2f})")
        
        # 因子分析
        bullish_factors = [f for f in factors if f.score > 0.3]
        bearish_factors = [f for f in factors if f.score < -0.3]
        neutral_factors = [f for f in factors if -0.3 <= f.score <= 0.3]
        
        if bullish_factors:
            reasoning.append(f"看涨因子 ({len(bullish_factors)}个):")
            for f in bullish_factors:
                reasoning.append(f"  • {f.name}: {f.description} (得分: {f.score:.2f}, 权重: {f.weight:.0%})")
        
        if bearish_factors:
            reasoning.append(f"看跌因子 ({len(bearish_factors)}个):")
            for f in bearish_factors:
                reasoning.append(f"  • {f.name}: {f.description} (得分: {f.score:.2f}, 权重: {f.weight:.0%})")
        
        if neutral_factors:
            reasoning.append(f"中性因子 ({len(neutral_factors)}个):")
            for f in neutral_factors:
                reasoning.append(f"  • {f.name}: {f.description} (得分: {f.score:.2f})")
        
        # 加权得分计算
        weighted_score = sum(f.score * f.weight for f in factors) / sum(f.weight for f in factors) if factors else 0
        reasoning.append(f"加权得分: {weighted_score:.3f}")
        
        # 决策
        if action == "BUY":
            reasoning.append(f"决策: 买入 (置信度: {confidence:.1%})")
        elif action == "SELL":
            reasoning.append(f"决策: 卖出 (置信度: {confidence:.1%})")
        else:
            reasoning.append(f"决策: 持仓观望 (置信度: {confidence:.1%})")
        
        return reasoning
    
    def explain_trade(self, trade_id: str, decision: SignalDecision,
                      executed_price: float, expected_price: float,
                      slippage: float) -> TradeExplanation:
        """
        生成完整交易解释。
        """
        explanation = TradeExplanation(
            trade_id=trade_id,
            timestamp=datetime.now(),
            ticker=decision.ticker,
            action=decision.action,
            signal_decision=decision,
            executed_price=executed_price,
            expected_price=expected_price,
            slippage=slippage,
        )
        
        self.explanations[trade_id] = explanation
        
        return explanation
    
    def add_post_trade_analysis(self, trade_id: str, pnl: float,
                                 hindsight_notes: List[str] = None):
        """添加后验分析"""
        if trade_id in self.explanations:
            exp = self.explanations[trade_id]
            exp.pnl = pnl
            exp.outcome = "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"
            if hindsight_notes:
                exp.hindsight_notes.extend(hindsight_notes)
    
    # ---- 格式化输出 ----
    
    def format_explanation(self, trade_id: str) -> str:
        """格式化交易解释为可读文本"""
        if trade_id not in self.explanations:
            return f"未找到交易 {trade_id} 的解释"
        
        exp = self.explanations[trade_id]
        dec = exp.signal_decision
        
        lines = [
            f"{'='*60}",
            f"交易解释: {exp.ticker} {exp.action}",
            f"{'='*60}",
            f"",
            f"时间: {exp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"策略: {dec.strategy_name}",
            f"市场环境: {dec.market_regime} (评分: {dec.market_score:.2f})",
            f"",
            f"信号因子分解:",
        ]
        
        for f in dec.factors:
            direction = "📈" if f.score > 0.3 else "📉" if f.score < -0.3 else "➡️"
            lines.append(f"  {direction} {f.name}: {f.description}")
            lines.append(f"     得分: {f.score:+.2f} | 权重: {f.weight:.0%} | 置信度: {f.confidence:.0%}")
        
        lines.extend([
            f"",
            f"决策置信度: {dec.confidence:.1%}",
            f"",
            f"执行结果:",
            f"  预期价格: ${exp.expected_price:.2f}",
            f"  实际成交: ${exp.executed_price:.2f}",
            f"  滑点: ${exp.slippage:.4f}",
        ])
        
        if exp.pnl is not None:
            pnl_emoji = "✅" if exp.pnl > 0 else "❌"
            lines.extend([
                f"",
                f"后验结果: {pnl_emoji} P&L: ${exp.pnl:+.2f}",
            ])
        
        if dec.reasoning_chain:
            lines.extend([
                f"",
                f"推理链:",
            ])
            for r in dec.reasoning_chain:
                lines.append(f"  {r}")
        
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)
    
    def get_decisions(self, days: int = 30) -> List[SignalDecision]:
        """获取最近决策列表"""
        recent = [d for d in self.decisions
                  if (datetime.now() - d.timestamp).days <= days]
        return recent
    
    def get_factor_statistics(self, days: int = 30) -> Dict:
        """获取因子统计（用于优化策略）"""
        recent = [d for d in self.decisions 
                  if (datetime.now() - d.timestamp).days <= days]
        
        if not recent:
            return {}
        
        stats = {}
        for decision in recent:
            for factor in decision.factors:
                if factor.name not in stats:
                    stats[factor.name] = {
                        'appearances': 0,
                        'avg_score': 0,
                        'avg_confidence': 0,
                        'correct_predictions': 0,
                    }
                
                s = stats[factor.name]
                s['appearances'] += 1
                s['avg_score'] += factor.score
                s['avg_confidence'] += factor.confidence
        
        # 计算平均值
        for name, s in stats.items():
            n = s['appearances']
            s['avg_score'] /= n
            s['avg_confidence'] /= n
        
        return stats
    
    def export_decisions(self, filepath: str):
        """导出所有决策到 JSON"""
        data = []
        for d in self.decisions:
            data.append({
                'timestamp': d.timestamp.isoformat(),
                'ticker': d.ticker,
                'action': d.action,
                'confidence': d.confidence,
                'market_regime': d.market_regime,
                'strategy_name': d.strategy_name,
                'factors': [
                    {
                        'name': f.name,
                        'weight': f.weight,
                        'score': f.score,
                        'confidence': f.confidence,
                        'description': f.description,
                    }
                    for f in d.factors
                ],
                'reasoning_chain': d.reasoning_chain,
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
