# ============================================================
# risk/manager.py — 风险管理系统
# 实时风控：仓位控制、止损止盈、日亏损熔断、连续亏损限制
# 支持 ATR 动态止盈止损 + 硬止损/硬止盈（固定百分比）
# ============================================================
import os
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


# ============================================================
# 持仓风控记录
# ============================================================

class PositionRisk:
    """单个持仓的风控信息"""
    
    def __init__(self, ticker: str, entry_price: float, qty: float,
                 stop_loss: float = None, take_profit: float = None, 
                 hard_stop_loss: float = None, hard_take_profit: float = None,
                 atr: float = None):
        self.ticker = ticker
        self.entry_price = entry_price
        self.qty = qty
        self.stop_loss = stop_loss  # ATR动态止损
        self.take_profit = take_profit  # ATR动态止盈
        self.hard_stop_loss = hard_stop_loss  # 硬止损（固定百分比）
        self.hard_take_profit = hard_take_profit  # 硬止盈（固定百分比）
        self.atr = atr
        self.highest_price = entry_price
        self.trailing_activated = False
        self.created_at = datetime.now()
    
    def update_high(self, current_price: float):
        """更新最高价（用于移动止损）"""
        if current_price > self.highest_price:
            self.highest_price = current_price
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "entry_price": self.entry_price,
            "qty": self.qty,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "hard_stop_loss": self.hard_stop_loss,
            "hard_take_profit": self.hard_take_profit,
            "atr": self.atr,
            "highest_price": self.highest_price,
            "trailing_activated": self.trailing_activated,
            "created_at": str(self.created_at),
        }


# ============================================================
# 风控管理器
# ============================================================

class RiskManager:
    """
    量化交易风控核心。
    
    风控规则：
    1. 单品种最大仓位比例
    2. 总仓位上限
    3. 单笔最大亏损（基于 ATR）
    4. 日亏损熔断
    5. 连续亏损限制
    6. ATR 动态止盈止损
    7. 硬止损/硬止盈（固定百分比）
    8. RSI > 70 自动止盈
    """

    def __init__(
        self,
        max_position_pct: float = 0.2,          # 单品种最多 20%
        max_total_position_pct: float = 0.8,     # 总仓位最多 80%
        risk_per_trade_pct: float = 0.06,        # 单笔风险 6%
        stop_loss_atr_mult: float = 2.0,         # 止损 ATR 倍数
        take_profit_atr_mult: float = 8.0,       # 止盈 ATR 倍数
        hard_stop_loss_pct: float = 0.05,        # 硬止损 5%（固定百分比）
        hard_take_profit_pct: float = 0.20,      # 硬止盈 20%（固定百分比）
        enable_hard_stop: bool = True,           # 是否启用硬止损
        enable_hard_take_profit: bool = True,    # 是否启用硬止盈
        daily_loss_stop_pct: float = 0.10,       # 日亏损超过 10% 熔断
        max_consecutive_losses: int = 5,         # 最多连续亏损 5 次
        trailing_stop_atr_mult: float = 2.0,     # 移动止损 ATR 倍数
        rsi_take_profit_threshold: float = 70,   # RSI 止盈阈值
        initial_cash: float = 100000.0,
        storage=None,
    ):
        self.max_position_pct = max_position_pct
        self.max_total_position_pct = max_total_position_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.hard_stop_loss_pct = hard_stop_loss_pct
        self.hard_take_profit_pct = hard_take_profit_pct
        self.enable_hard_stop = enable_hard_stop
        self.enable_hard_take_profit = enable_hard_take_profit
        self.daily_loss_stop_pct = daily_loss_stop_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.rsi_take_profit_threshold = rsi_take_profit_threshold
        self.initial_cash = initial_cash
        self.storage = storage
        
        # 持仓风控记录 {ticker: PositionRisk}
        self.position_risks: Dict[str, PositionRisk] = {}
        
        self._reset_daily_state()
        self._load_state()

    def _reset_daily_state(self):
        """重置日内状态"""
        self.today_loss = 0.0
        self.today_trades = 0
        self.today_loss_trades = 0
        self.consecutive_losses = 0
        self.daily_stopped = False
        self.cooldown_until: Optional[datetime] = None
        logger.info("[Risk] 日内状态已重置")

    def _load_state(self):
        """从存储加载跨日状态"""
        if not self.storage:
            return
        try:
            import sqlite3
            conn = sqlite3.connect(self.storage.db_path)
            cur = conn.execute(
                "SELECT name, value FROM risk_state WHERE name IN "
                "('consecutive_losses', 'last_reset_date')"
            )
            for name, value in cur.fetchall():
                if name == "consecutive_losses":
                    self.consecutive_losses = int(value)
            conn.close()
        except Exception:
            pass

    def _save_state(self):
        """保存跨日状态"""
        if not self.storage:
            return
        try:
            import sqlite3
            conn = sqlite3.connect(self.storage.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_state (
                    name TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("INSERT OR REPLACE INTO risk_state VALUES ('consecutive_losses', ?)",
                        (str(self.consecutive_losses),))
            conn.execute("INSERT OR REPLACE INTO risk_state VALUES ('last_reset_date', ?)",
                        (str(date.today()),))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ---- 核心检查方法 ----

    def is_trading_allowed(self) -> bool:
        """是否允许开新仓"""
        if self.daily_stopped:
            logger.warning(f"[Risk] 日亏损熔断触发，禁止交易")
            return False

        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds
            logger.warning(f"[Risk] 冷静期，剩余 {remaining}s")
            return False

        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"[Risk] 连续亏损 {self.consecutive_losses} 次超过上限，暂停交易")
            return False

        return True

    def check_buy(self, ticker: str, price: float, qty: float,
                  total_equity: float, current_positions: dict,
                  atr: float = None) -> Tuple[bool, str]:
        """
        检查是否允许买入。
        
        返回:
            (允许, 原因)
        """
        if not self.is_trading_allowed():
            return False, "风控禁止"

        position_value = qty * price
        position_pct = position_value / (total_equity + 1e-10)

        # 单品种仓位超限
        if position_pct > self.max_position_pct:
            return False, f"单品种仓位 {position_pct:.1%} 超过上限 {self.max_position_pct:.1%}"

        # 总仓位超限
        current_pos_value = sum(
            p.get("qty", 0) * price for t, p in current_positions.items()
        )
        total_pos_pct = (current_pos_value + position_value) / (total_equity + 1e-10)
        if total_pos_pct > self.max_total_position_pct:
            return False, f"总仓位 {total_pos_pct:.1%} 超过上限 {self.max_total_position_pct:.1%}"

        return True, "允许"

    def check_sell(self, ticker: str) -> Tuple[bool, str]:
        """检查是否允许卖出"""
        if self.daily_stopped:
            return False, "日熔断中"
        return True, "允许"

    # ---- ATR 仓位计算 ----

    def calculate_position_size(self, price: float, atr: float, 
                                 total_equity: float = None) -> Tuple[float, float]:
        """
        基于 ATR 计算仓位大小。
        
        公式：仓位 = 总权益 × 风险比例 / (ATR × 止损倍数)
        
        返回:
            (股数, 风险金额)
        """
        if total_equity is None:
            total_equity = self.initial_cash
        
        if atr <= 0 or price <= 0:
            return 0.0, 0.0
        
        # 风险金额
        risk_amount = total_equity * self.risk_per_trade_pct
        
        # 每股风险
        risk_per_share = atr * self.stop_loss_atr_mult
        
        # 仓位大小
        if risk_per_share > 0:
            qty = risk_amount / risk_per_share
            # 向下取整到整数股（A股）或保留小数（美股）
            qty = float(int(qty * 10000)) / 10000  # 保留4位小数
        else:
            qty = 0.0
        
        return max(qty, 0.0), risk_amount

    def calculate_stop_loss_take_profit(self, entry_price: float, atr: float) -> Tuple[float, float]:
        """
        计算ATR动态止损止盈价格。
        
        返回:
            (止损价, 止盈价)
        """
        if atr <= 0:
            return None, None
        
        stop_loss = entry_price - self.stop_loss_atr_mult * atr
        take_profit = entry_price + self.take_profit_atr_mult * atr
        
        return stop_loss, take_profit
    
    def calculate_hard_stop_loss_take_profit(self, entry_price: float) -> Tuple[float, float]:
        """
        计算硬止损/硬止盈价格（固定百分比）。
        
        返回:
            (硬止损价, 硬止盈价)
        """
        hard_stop = entry_price * (1 - self.hard_stop_loss_pct) if self.enable_hard_stop else None
        hard_tp = entry_price * (1 + self.hard_take_profit_pct) if self.enable_hard_take_profit else None
        return hard_stop, hard_tp

    # ---- 持仓风控管理 ----

    def add_position(self, ticker: str, entry_price: float, qty: float,
                     atr: float = None):
        """
        添加持仓并设置止损止盈。
        """
        # ATR动态止损止盈
        stop_loss, take_profit = None, None
        if atr and atr > 0:
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(entry_price, atr)
        
        # 硬止损/硬止盈
        hard_stop, hard_tp = self.calculate_hard_stop_loss_take_profit(entry_price)
        
        pos_risk = PositionRisk(
            ticker=ticker,
            entry_price=entry_price,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            hard_stop_loss=hard_stop,
            hard_take_profit=hard_tp,
            atr=atr,
        )
        self.position_risks[ticker] = pos_risk
        
        sl_str = f"${stop_loss:.2f}" if stop_loss else "N/A"
        tp_str = f"${take_profit:.2f}" if take_profit else "N/A"
        hard_sl_str = f"${hard_stop:.2f}" if hard_stop else "N/A"
        hard_tp_str = f"${hard_tp:.2f}" if hard_tp else "N/A"
        
        logger.info(
            f"[Risk] 添加持仓 {ticker}: 入场=${entry_price:.2f}, "
            f"ATR止损={sl_str}, ATR止盈={tp_str}, "
            f"硬止损={hard_sl_str}({self.hard_stop_loss_pct:.1%}), "
            f"硬止盈={hard_tp_str}({self.hard_take_profit_pct:.1%})"
        )

    def check_position_stop(self, ticker: str, current_price: float,
                            current_rsi: float = None) -> Tuple[bool, str]:
        """
        检查持仓是否触发止损或止盈。
        
        返回:
            (是否触发, 原因)
        """
        if ticker not in self.position_risks:
            return False, ""
        
        pos = self.position_risks[ticker]
        
        # 更新最高价
        pos.update_high(current_price)
        
        # 检查硬止盈（优先级最高）
        if pos.hard_take_profit and current_price >= pos.hard_take_profit:
            return True, f"硬止盈触发: ${current_price:.2f} >= ${pos.hard_take_profit:.2f} ({self.hard_take_profit_pct:.1%})"
        
        # 检查ATR止盈
        if pos.take_profit and current_price >= pos.take_profit:
            return True, f"ATR止盈触发: ${current_price:.2f} >= ${pos.take_profit:.2f}"
        
        # RSI > 70 自动止盈
        if current_rsi is not None and current_rsi > self.rsi_take_profit_threshold:
            return True, f"RSI止盈: RSI={current_rsi:.1f} > {self.rsi_take_profit_threshold}"
        
        # 检查硬止损（优先级高于ATR止损）
        if pos.hard_stop_loss and current_price <= pos.hard_stop_loss:
            return True, f"硬止损触发: ${current_price:.2f} <= ${pos.hard_stop_loss:.2f} ({self.hard_stop_loss_pct:.1%})"
        
        # 检查ATR止损
        if pos.stop_loss and current_price <= pos.stop_loss:
            return True, f"ATR止损触发: ${current_price:.2f} <= ${pos.stop_loss:.2f}"
        
        return False, ""

    def update_trailing_stop(self, ticker: str):
        """
        更新移动止损。
        当价格创新高时，将止损上移。
        """
        if ticker not in self.position_risks:
            return
        
        pos = self.position_risks[ticker]
        atr = pos.atr
        
        if not atr or atr <= 0:
            return
        
        # 新止损 = 最高价 - ATR × 倍数
        new_stop = pos.highest_price - self.trailing_stop_atr_mult * atr
        
        # 只能上移
        if new_stop > pos.stop_loss:
            old_stop = pos.stop_loss
            pos.stop_loss = new_stop
            pos.trailing_activated = True
            logger.info(f"[Risk] {ticker} 移动止损: ${old_stop:.2f} -> ${new_stop:.2f}")

    def remove_position(self, ticker: str):
        """移除持仓"""
        if ticker in self.position_risks:
            del self.position_risks[ticker]

    def get_position_risk(self, ticker: str) -> dict:
        """获取持仓风控信息"""
        if ticker in self.position_risks:
            return self.position_risks[ticker].to_dict()
        return None

    def get_all_position_risks(self) -> Dict[str, dict]:
        """获取所有持仓风控信息"""
        return {t: p.to_dict() for t, p in self.position_risks.items()}

    # ---- 交易记录与风控状态 ----

    def record_trade(self, trade: dict):
        """
        记录交易结果，更新风控状态。
        trade 需含：action, realized_pnl, ticker
        """
        self.today_trades += 1
        action = trade.get("action", "")
        realized = trade.get("realized_pnl", 0)
        ticker = trade.get("ticker", "")

        if action == "SELL" and realized is not None:
            if realized < 0:
                self.today_loss += abs(realized)
                self.consecutive_losses += 1
                self.today_loss_trades += 1
                logger.warning(
                    f"[Risk] 亏损交易 #{self.today_loss_trades} {ticker}: "
                    f"亏损=${realized:.2f}, 连续亏损={self.consecutive_losses}"
                )
            else:
                self.consecutive_losses = 0  # 盈利重置连续亏损
                logger.info(f"[Risk] 盈利交易 {ticker}: +${realized:.2f}")

        # 日亏损熔断检查
        daily_loss_pct = self.today_loss / (self.initial_cash + 1e-10)
        if daily_loss_pct > self.daily_loss_stop_pct:
            self.daily_stopped = True
            logger.critical(
                f"[Risk] 日亏损熔断！已亏 ${self.today_loss:.2f} "
                f"({daily_loss_pct:.2%}) 超过阈值 {self.daily_loss_stop_pct:.2%}"
            )

        # 连续亏损超限 → 冷静期
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.cooldown_until = datetime.now() + timedelta(minutes=30)
            logger.warning(
                f"[Risk] 连续 {self.consecutive_losses} 次亏损，进入冷静期至 {self.cooldown_until}"
            )

        self._save_state()

    def get_status(self) -> dict:
        """获取当前风控状态"""
        return {
            "daily_stopped": self.daily_stopped,
            "consecutive_losses": self.consecutive_losses,
            "today_loss": self.today_loss,
            "today_trades": self.today_trades,
            "cooldown": self.cooldown_until is not None,
            "cooldown_until": str(self.cooldown_until) if self.cooldown_until else None,
            "active_positions": len(self.position_risks),
            "limits": {
                "max_position_pct": self.max_position_pct,
                "risk_per_trade_pct": self.risk_per_trade_pct,
                "stop_loss_atr_mult": self.stop_loss_atr_mult,
                "take_profit_atr_mult": self.take_profit_atr_mult,
                "hard_stop_loss_pct": self.hard_stop_loss_pct,
                "hard_take_profit_pct": self.hard_take_profit_pct,
                "enable_hard_stop": self.enable_hard_stop,
                "enable_hard_take_profit": self.enable_hard_take_profit,
                "daily_loss_stop_pct": self.daily_loss_stop_pct,
                "max_consecutive_losses": self.max_consecutive_losses,
                "rsi_take_profit_threshold": self.rsi_take_profit_threshold,
            }
        }

    def daily_reset(self):
        """每日开盘前调用"""
        self._reset_daily_state()
        logger.info("[Risk] 日风控状态已重置")

    def max_position_size(self, ticker: str, price: float,
                          action: str, total_equity: float = 0) -> float:
        """
        计算最大可买入数量（兼容旧接口）。
        """
        if total_equity <= 0:
            total_equity = self.initial_cash

        max_value = total_equity * self.max_position_pct
        qty = max_value / (price + 1e-10)
        qty = float(int(qty * 10000)) / 10000
        return max(qty, 0.0)
