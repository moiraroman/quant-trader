# -*- coding: utf-8 -*-
# ============================================================
# trading/bot.py — 模拟交易机器人（持续执行）
# ============================================================
import threading
import time
import logging
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class PaperTradingBot:
    """
    模拟交易机器人：后台持续运行，定时检查信号并执行交易。
    
    功能：
    - 可配置监控间隔（分钟）
    - 自动获取数据、生成信号、执行交易
    - 支持止盈止损检查
    - 状态持久化
    """
    
    def __init__(
        self,
        trader,              # PaperTrader 实例
        fetcher,             # 数据获取器
        strategy,            # 策略实例（需有 compute_signals 方法）
        tickers: List[str],  # 监控标的列表
        interval_minutes: float = 15,  # 检查间隔（分钟）
        risk_manager=None,   # 风控管理器
        on_trade: Optional[Callable] = None,  # 交易回调
        on_signal: Optional[Callable] = None,  # 信号回调
        on_status: Optional[Callable] = None,  # 状态回调
    ):
        self.trader = trader
        self.fetcher = fetcher
        self.strategy = strategy
        self.tickers = tickers
        self.interval_minutes = interval_minutes
        self.risk_manager = risk_manager
        
        # 回调函数
        self.on_trade = on_trade
        self.on_signal = on_signal
        self.on_status = on_status
        
        # 运行状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 状态记录
        self.status = {
            "is_running": False,
            "last_check": None,
            "next_check": None,
            "total_checks": 0,
            "total_trades": 0,
            "errors": [],
            "positions": {},
        }
        
        # 止盈止损追踪
        self._stop_losses: Dict[str, float] = {}  # {ticker: stop_price}
        self._take_profits: Dict[str, float] = {}  # {ticker: tp_price}
    
    # ---- 核心方法 ----
    
    def start(self) -> bool:
        """启动后台监控"""
        if self._running:
            logger.warning("[Bot] 已在运行中")
            return False
        
        self._running = True
        self._stop_event.clear()
        self.status["is_running"] = True
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"[Bot] 启动监控，间隔 {self.interval_minutes} 分钟，标的: {self.tickers}")
        self._update_status("started")
        return True
    
    def stop(self) -> bool:
        """停止监控"""
        if not self._running:
            logger.warning("[Bot] 未在运行")
            return False
        
        self._running = False
        self._stop_event.set()
        self.status["is_running"] = False
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        logger.info("[Bot] 停止监控")
        self._update_status("stopped")
        return True
    
    def run_once(self) -> Dict[str, Any]:
        """
        执行一次检查（手动触发）。
        返回检查结果。
        """
        return self._check_and_trade()
    
    # ---- 内部方法 ----
    
    def _run_loop(self):
        """后台运行循环"""
        while self._running and not self._stop_event.is_set():
            try:
                # 执行检查
                result = self._check_and_trade()
                self.status["last_check"] = datetime.now().isoformat()
                self.status["total_checks"] += 1
                
                # 回调
                if self.on_status:
                    self.on_status(self.status)
                
            except Exception as e:
                logger.error(f"[Bot] 检查异常: {e}")
                self.status["errors"].append({
                    "time": datetime.now().isoformat(),
                    "error": str(e),
                })
            
            # 计算下次检查时间
            next_check = datetime.now().timestamp() + self.interval_minutes * 60
            self.status["next_check"] = datetime.fromtimestamp(next_check).isoformat()
            
            # 等待
            self._stop_event.wait(timeout=self.interval_minutes * 60)
    
    def _check_and_trade(self) -> Dict[str, Any]:
        """
        检查信号并执行交易。
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "signals": {},
            "errors": [],
        }
        
        for ticker in self.tickers:
            try:
                # 1. 获取数据
                df = self._fetch_data(ticker)
                if df is None or df.empty:
                    result["errors"].append(f"{ticker}: 无法获取数据")
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # 2. 检查止盈止损
                stop_triggered = self._check_stop_loss_take_profit(ticker, current_price)
                if stop_triggered:
                    result["actions"].append(stop_triggered)
                    continue
                
                # 3. 生成信号
                signals = self._compute_signals(ticker, df)
                result["signals"][ticker] = signals
                
                if self.on_signal:
                    self.on_signal(ticker, signals)
                
                # 4. 执行交易
                action = self._execute_signal(ticker, signals, current_price)
                if action:
                    result["actions"].append(action)
                    self.status["total_trades"] += 1
                    
                    if self.on_trade:
                        self.on_trade(action)
                
            except Exception as e:
                logger.error(f"[Bot] {ticker} 处理失败: {e}")
                result["errors"].append(f"{ticker}: {str(e)}")
        
        # 更新持仓状态
        self.status["positions"] = self.trader.get_all_positions()
        
        return result
    
    def _fetch_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """获取数据"""
        try:
            # 获取最近30天数据用于计算指标
            df = self.fetcher.download_history(ticker, period="1mo", interval="1d")
            return df
        except Exception as e:
            logger.error(f"[Bot] 获取 {ticker} 数据失败: {e}")
            return None
    
    def _compute_signals(self, ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
        """计算信号"""
        try:
            # 策略需要实现 compute_signals 或 _compute_signals
            if hasattr(self.strategy, 'compute_signals'):
                sig_df = self.strategy.compute_signals(df)
            elif hasattr(self.strategy, '_compute_signals'):
                sig_df = self.strategy._compute_signals(df)
            else:
                logger.error(f"[Bot] 策略无 compute_signals 方法")
                return {"signal": "HOLD", "reason": "策略接口错误"}
            
            if sig_df is None or sig_df.empty:
                return {"signal": "HOLD", "reason": "无信号"}
            
            # 取最新信号
            last = sig_df.iloc[-1]
            return {
                "signal": last.get("signal", "HOLD"),
                "confidence": last.get("confidence", 0),
                "strength": last.get("strength", 0),
                "reason": last.get("reason", ""),
            }
        except Exception as e:
            logger.error(f"[Bot] 计算信号失败: {e}")
            return {"signal": "HOLD", "reason": f"计算失败: {e}"}
    
    def _execute_signal(self, ticker: str, signals: Dict, current_price: float) -> Optional[Dict]:
        """执行信号"""
        signal = signals.get("signal", "HOLD")
        reason = signals.get("reason", "")
        confidence = signals.get("confidence", 0)
        
        if signal == "HOLD":
            return None
        
        # 检查当前持仓
        position = self.trader.get_position(ticker)
        
        if signal == "BUY":
            # 已有持仓则跳过
            if position and position["qty"] > 0:
                logger.info(f"[Bot] {ticker} 已有持仓，跳过买入")
                return None
            
            # 计算仓位大小
            if self.risk_manager:
                qty = self.risk_manager.calculate_position_size(
                    self.trader.cash, current_price, 
                    stop_loss=current_price * 0.94  # 默认 6% 止损
                )
                # 记录止盈止损价格
                sl_tp = self.risk_manager.calculate_stop_loss_take_profit(current_price)
                self._stop_losses[ticker] = sl_tp[0]  # tuple: (stop_loss, take_profit)
                self._take_profits[ticker] = sl_tp[1]
            else:
                # 默认使用 20% 资金
                qty = (self.trader.cash * 0.2) / current_price
            
            # 执行买入
            trade_result = self.trader.buy(
                ticker=ticker,
                price=current_price,
                qty=qty,
                signal=reason,
                confidence=confidence,
            )
            
            if trade_result["success"]:
                return {
                    "ticker": ticker,
                    "action": "BUY",
                    "price": current_price,
                    "qty": qty,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
        
        elif signal == "SELL":
            # 无持仓则跳过
            if not position or position["qty"] <= 0:
                logger.info(f"[Bot] {ticker} 无持仓，跳过卖出")
                return None
            
            # 执行卖出（清仓）
            trade_result = self.trader.sell(
                ticker=ticker,
                price=current_price,
                qty=None,  # 清仓
                signal=reason,
                confidence=confidence,
            )
            
            if trade_result["success"]:
                # 清除止盈止损
                self._stop_losses.pop(ticker, None)
                self._take_profits.pop(ticker, None)
                
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "price": current_price,
                    "qty": trade_result["trade"]["quantity"],
                    "reason": reason,
                    "pnl": trade_result.get("realized_pnl", 0),
                    "timestamp": datetime.now().isoformat(),
                }
        
        return None
    
    def _check_stop_loss_take_profit(self, ticker: str, current_price: float) -> Optional[Dict]:
        """检查止盈止损"""
        position = self.trader.get_position(ticker)
        if not position or position["qty"] <= 0:
            return None
        
        stop_loss = self._stop_losses.get(ticker)
        take_profit = self._take_profits.get(ticker)
        
        if not stop_loss and not take_profit:
            return None
        
        triggered = None
        reason = ""
        
        # 止损检查
        if stop_loss and current_price <= stop_loss:
            triggered = "STOP_LOSS"
            reason = f"触发止损 @ ${stop_loss:.2f}"
        
        # 止盈检查
        if take_profit and current_price >= take_profit:
            triggered = "TAKE_PROFIT"
            reason = f"触发止盈 @ ${take_profit:.2f}"
        
        if triggered:
            # 执行卖出
            trade_result = self.trader.sell(
                ticker=ticker,
                price=current_price,
                qty=None,
                signal=reason,
            )
            
            if trade_result["success"]:
                self._stop_losses.pop(ticker, None)
                self._take_profits.pop(ticker, None)
                
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "type": triggered,
                    "price": current_price,
                    "qty": trade_result["trade"]["quantity"],
                    "reason": reason,
                    "pnl": trade_result.get("realized_pnl", 0),
                    "timestamp": datetime.now().isoformat(),
                }
        
        return None
    
    def _update_status(self, event: str):
        """更新状态并回调"""
        self.status["last_event"] = event
        self.status["last_update"] = datetime.now().isoformat()
        
        if self.on_status:
            self.on_status(self.status)
    
    # ---- 配置方法 ----
    
    def set_interval(self, minutes: float):
        """修改监控间隔"""
        self.interval_minutes = minutes
        logger.info(f"[Bot] 监控间隔改为 {minutes} 分钟")
    
    def add_ticker(self, ticker: str):
        """添加监控标的"""
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            logger.info(f"[Bot] 添加标的: {ticker}")
    
    def remove_ticker(self, ticker: str):
        """移除监控标的"""
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            logger.info(f"[Bot] 移除标的: {ticker}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self.status.copy()


# ============================================================
# 单例管理器（用于 WebUI 共享状态）
# ============================================================

_bot_instance: Optional[PaperTradingBot] = None

def get_bot() -> Optional[PaperTradingBot]:
    """获取全局 Bot 实例"""
    return _bot_instance

def create_bot(
    trader,
    fetcher,
    strategy,
    tickers: List[str],
    interval_minutes: float = 15,
    risk_manager=None,
    **callbacks,
) -> PaperTradingBot:
    """创建并注册全局 Bot 实例"""
    global _bot_instance
    
    _bot_instance = PaperTradingBot(
        trader=trader,
        fetcher=fetcher,
        strategy=strategy,
        tickers=tickers,
        interval_minutes=interval_minutes,
        risk_manager=risk_manager,
        **callbacks,
    )
    
    return _bot_instance
