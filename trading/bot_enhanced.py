"""
trading/bot_enhanced.py — 增强版模拟交易机器人

功能：
    - 集成 OrderManager、EquityTracker、SignalExplainer
    - 多策略并行运行
    - 动态仓位调整
    - 完整信号审计
    - 异常恢复机制
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable

from trading.paper import PaperTrader
from trading.order_manager import OrderManager, OrderType
from trading.equity_tracker import EquityTracker, EquitySnapshot, TradeRecord
from trading.signal_explainer import SignalExplainer, SignalFactor, SignalDecision
from trading.strategy_config import StrategyConfigManager
from risk.manager import RiskManager

logger = logging.getLogger(__name__)

# ============================================================
# 增强版交易机器人
# ============================================================

class EnhancedPaperTradingBot:
    """
    增强版模拟交易机器人。
    
    相比原版改进：
    1. 集成 OrderManager 支持多订单类型
    2. 集成 EquityTracker 实时追踪绩效
    3. 集成 SignalExplainer 记录每笔交易理由
    4. 支持多策略并行运行
    5. 动态仓位调整
    6. 异常恢复机制
    """
    
    def __init__(
        self,
        paper_trader: PaperTrader,
        risk_manager: RiskManager,
        strategy_config: StrategyConfigManager,
        check_interval: int = 300,
    ):
        self.paper = paper_trader
        self.risk = risk_manager
        self.config = strategy_config
        
        # 核心组件
        self.order_manager = OrderManager(paper_trader)
        self.equity_tracker = EquityTracker()
        self.signal_explainer = SignalExplainer()
        
        # 配置
        self.check_interval = check_interval
        self.is_running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # 统计
        self.check_count = 0
        self.trade_count = 0
        
        # 回调
        self.on_trade: Optional[Callable] = None
        self.on_signal: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # 价格缓存（用于订单触发检查）
        self._price_cache: Dict[str, float] = {}
        
        # 初始化权益追踪
        self._record_initial_equity()
    
    def _get_equity_info(self) -> dict:
        """获取权益信息（兼容 PaperTrader API）"""
        prices = self._price_cache.copy()
        if not prices:
            prices = {t: p["avg_cost"] for t, p in self.paper.positions.items()}
        return self.paper.get_equity_snapshot(prices)

    def _record_initial_equity(self):
        """记录初始权益"""
        eq = self._get_equity_info()
        snapshot = EquitySnapshot(
            timestamp=datetime.now(),
            total_equity=eq.get("total_equity", self.paper.cash),
            cash=eq.get("cash", self.paper.cash),
            position_value=eq.get("position_value", 0),
            unrealized_pnl=0,
            realized_pnl_today=0,
            positions=self._get_positions_dict(),
        )
        self.equity_tracker.record_equity(snapshot)
    
    def _get_positions_dict(self) -> Dict:
        """获取持仓字典"""
        positions = {}
        for ticker, pos in self.paper.positions.items():
            positions[ticker] = {
                'qty': pos.qty,
                'avg_cost': pos.avg_cost,
                'current_price': pos.current_price if hasattr(pos, 'current_price') else pos.avg_cost,
                'unrealized_pnl': pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else 0,
            }
        return positions
    
    # ---- 核心运行循环 ----
    
    def start(self):
        """启动机器人"""
        if self.is_running:
            logger.warning("[Bot] 机器人已在运行")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info("[Bot] 机器人已启动")
    
    def stop(self):
        """停止机器人"""
        self.is_running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        # 刷新缓存
        self.equity_tracker._flush_equity_cache()
        
        logger.info("[Bot] 机器人已停止")
    
    def _run_loop(self):
        """主运行循环"""
        while not self._stop_event.is_set():
            try:
                self._check_once()
                self.check_count += 1
            except Exception as e:
                logger.error(f"[Bot] 检查异常: {e}")
                if self.on_error:
                    self.on_error(e)
            
            # 等待下一次检查
            self._stop_event.wait(self.check_interval)
    
    def _check_once(self):
        """执行一次检查"""
        # 1. 检查风控
        if not self._check_risk():
            return
        
        # 2. 获取活跃策略
        active_instances = self.config.get_active_instances()
        
        # 3. 对每个策略实例生成信号
        for instance in active_instances:
            try:
                self._process_strategy_instance(instance)
            except Exception as e:
                logger.error(f"[Bot] 策略 {instance.instance_id} 处理失败: {e}")
        
        # 4. 检查订单触发
        self._check_order_triggers()
        
        # 5. 记录权益快照
        self._record_equity_snapshot()
    
    def _check_risk(self) -> bool:
        """检查风控状态"""
        # 检查日亏损熔断
        if hasattr(self.risk, 'daily_loss_triggered') and self.risk.daily_loss_triggered:
            logger.warning("[Bot] 日亏损熔断已触发，暂停交易")
            return False
        
        # 检查连续亏损
        if hasattr(self.risk, 'consecutive_losses') and hasattr(self.risk, 'max_consecutive_losses'):
            if self.risk.consecutive_losses >= self.risk.max_consecutive_losses:
                logger.warning("[Bot] 连续亏损超限，暂停交易")
                return False
        
        # 检查冷静期
        if hasattr(self.risk, 'cooling_down') and self.risk.cooling_down:
            logger.warning("[Bot] 冷静期中，暂停交易")
            return False
        
        return True
    
    def _process_strategy_instance(self, instance):
        """处理单个策略实例"""
        ticker = instance.ticker
        
        # 获取当前价格（简化，实际应调用数据获取器）
        current_price = self._get_current_price(ticker)
        if not current_price:
            return
        
        # 生成信号（简化示例）
        signal = self._generate_signal(instance, current_price)
        if not signal:
            return
        
        # 记录信号决策
        decision = self.signal_explainer.record_signal(
            ticker=ticker,
            action=signal['action'],
            confidence=signal['confidence'],
            factors=signal['factors'],
            strategy_name=instance.strategy_name,
            strategy_params={name: p.value for name, p in instance.parameters.items()},
            market_regime=signal.get('market_regime', 'unknown'),
            market_score=signal.get('market_score', 0),
        )
        
        # 执行信号
        if signal['action'] in ['BUY', 'SELL']:
            self._execute_signal(instance, decision, current_price)
        
        # 回调
        if self.on_signal:
            self.on_signal(decision)
    
    def _generate_signal(self, instance, current_price) -> Optional[Dict]:
        """
        生成交易信号。
        简化实现，实际应调用策略模块。
        """
        # 这里应该调用实际的策略逻辑
        # 示例：基于参数生成简单信号
        params = {name: p.value for name, p in instance.parameters.items()}
        
        # 模拟信号生成（实际应替换为真实策略）
        import random
        if random.random() > 0.95:  # 5% 概率生成信号
            action = random.choice(['BUY', 'SELL'])
            confidence = random.uniform(0.5, 0.9)
            
            factors = [
                SignalFactor(
                    name="MA_Cross",
                    weight=0.3,
                    score=0.8 if action == 'BUY' else -0.8,
                    confidence=0.9,
                    description=f"{'金叉' if action == 'BUY' else '死叉'}信号",
                ),
                SignalFactor(
                    name="RSI",
                    weight=0.2,
                    score=0.5 if action == 'BUY' else -0.5,
                    confidence=0.7,
                    description="RSI中性",
                ),
            ]
            
            return {
                'action': action,
                'confidence': confidence,
                'factors': factors,
                'market_regime': 'bullish',
                'market_score': 0.7,
            }
        
        return None
    
    def _execute_signal(self, instance, decision: SignalDecision, current_price: float):
        """执行信号"""
        ticker = instance.ticker
        action = decision.action
        
        # 计算仓位（动态调整）
        position_size = self._calculate_position_size(instance, decision, current_price)
        if position_size <= 0:
            return
        
        # 检查风控限制
        if not self._check_position_limits(ticker, action, position_size, current_price):
            return
        
        # 提交订单
        order = self.order_manager.submit_order(
            ticker=ticker,
            action=action,
            qty=position_size,
            order_type="MARKET",
            signal_source=instance.instance_id,
            signal_confidence=decision.confidence,
            signal_reason="; ".join(decision.reasoning_chain[:3]),
        )
        
        # 记录交易
        if order.status.value == "FILLED":
            self.trade_count += 1
            
            # 记录完整交易
            trade_record = TradeRecord(
                trade_id=order.order_id,
                timestamp=datetime.now(),
                ticker=ticker,
                action=action,
                order_type="MARKET",
                qty=position_size,
                price=current_price,
                filled_qty=order.filled_qty,
                filled_price=order.avg_filled_price or current_price,
                signal_source=instance.instance_id,
                signal_confidence=decision.confidence,
                signal_reason="; ".join(decision.reasoning_chain[:3]),
                slippage=order.total_slippage,
                commission=order.total_commission,
                expected_price=current_price,
                execution_delay_ms=0,
            )
            self.equity_tracker.record_trade(trade_record)
            
            # 生成交易解释
            self.signal_explainer.explain_trade(
                trade_id=order.order_id,
                decision=decision,
                executed_price=order.avg_filled_price or current_price,
                expected_price=current_price,
                slippage=order.total_slippage,
            )
            
            # 回调
            if self.on_trade:
                self.on_trade(trade_record)
            
            logger.info(
                f"[Bot] 执行交易 {action} {ticker} qty={position_size} "
                f"price=${current_price:.2f} confidence={decision.confidence:.2f}"
            )
    
    def _calculate_position_size(self, instance, decision, current_price) -> float:
        """
        动态仓位计算。
        
        考虑因素：
        1. 策略权重
        2. 信号置信度
        3. 当前市场波动率
        4. 账户可用资金
        """
        # 基础仓位（账户价值的固定比例）
        base_position_pct = 0.1  # 10%
        
        # 策略权重调整
        weight = instance.weight
        
        # 置信度调整
        confidence_factor = decision.confidence
        
        # 计算目标仓位金额
        eq = self._get_equity_info()
        total_equity = eq.get("total_equity", self.paper.cash)
        target_value = total_equity * base_position_pct * weight * confidence_factor
        
        # 转换为股数
        qty = int(target_value / current_price)
        
        # 确保至少买1股
        return max(qty, 1)
    
    def _check_position_limits(self, ticker, action, qty, price) -> bool:
        """检查持仓限制"""
        # 检查单品种最大仓位
        if hasattr(self.risk, 'max_position_pct'):
            position_value = qty * price
            eq = self._get_equity_info()
            total_equity = eq.get("total_equity", self.paper.cash)
            if total_equity > 0 and position_value / total_equity > self.risk.max_position_pct:
                logger.warning(f"[Bot] 单品种仓位超限: {ticker}")
                return False
        
        # 检查总仓位
        if action == 'BUY':
            eq = self._get_equity_info()
            new_position_value = eq.get("position_value", 0) + qty * price
            total_equity = eq.get("total_equity", self.paper.cash)
            if total_equity > 0 and new_position_value / total_equity > 0.8:  # 80% 总仓位上限
                logger.warning("[Bot] 总仓位超限")
                return False
        
        return True
    
    def _check_order_triggers(self):
        """检查订单触发条件"""
        for ticker in list(self._price_cache.keys()):
            price = self._price_cache[ticker]
            self.order_manager.on_price_update(ticker, price)
    
    def _record_equity_snapshot(self):
        """记录权益快照"""
        eq = self._get_equity_info()
        unrealized = sum(
            p.get("unrealized_pnl", 0) for p in eq.get("positions", [])
        )
        snapshot = EquitySnapshot(
            timestamp=datetime.now(),
            total_equity=eq.get("total_equity", self.paper.cash),
            cash=eq.get("cash", self.paper.cash),
            position_value=eq.get("position_value", 0),
            unrealized_pnl=unrealized,
            realized_pnl_today=0,
            positions=self._get_positions_dict(),
        )
        self.equity_tracker.record_equity(snapshot)
    
    def _get_current_price(self, ticker: str) -> Optional[float]:
        """获取当前价格（简化）"""
        # 实际应调用数据获取器
        # 这里使用缓存或模拟
        return self._price_cache.get(ticker)
    
    def update_price(self, ticker: str, price: float):
        """更新价格（由外部调用）"""
        self._price_cache[ticker] = price
    
    # ---- 手动操作 ----
    
    def run_once(self):
        """手动执行一次检查"""
        self._check_once()
        self.check_count += 1
    
    def get_status(self) -> Dict:
        """获取机器人状态"""
        eq = self._get_equity_info()
        return {
            'is_running': self.is_running,
            'check_count': self.check_count,
            'trade_count': self.trade_count,
            'active_strategies': len(self.config.get_active_instances()),
            'open_orders': len(self.order_manager.get_open_orders()),
            'total_equity': eq.get('total_equity', self.paper.cash),
            'cash': eq.get('cash', self.paper.cash),
            'position_value': eq.get('position_value', 0),
        }
    
    def get_equity_tracker(self) -> EquityTracker:
        """获取权益追踪器"""
        return self.equity_tracker
    
    def get_order_manager(self) -> OrderManager:
        """获取订单管理器"""
        return self.order_manager
    
    def get_signal_explainer(self) -> SignalExplainer:
        """获取信号解释器"""
        return self.signal_explainer
