# ============================================================
# trading/paper.py — 模拟交易执行层
# 对接 MOOMOO OpenAPI 模拟账户
# ============================================================
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 模拟交易引擎（纯本地，无 API 依赖）
# ============================================================

class PaperTrader:
    """
    纯本地模拟交易：
    - 不需要真实券商 API，直接本地撮合
    - 精确模拟资金、持仓、盈亏
    - 交易记录存入 SQLiteStorage
    - 与回测引擎共享 storage 模块
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        storage=None,      # data/storage.py 的 SQLiteStorage 实例
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.storage = storage
        self.positions: dict[str, dict] = {}  # {ticker: {qty, avg_cost}}
        self.mode = "paper"

    # ---- 核心操作 ----

    def buy(self, ticker: str, price: float, qty: float,
            signal: str = "", confidence: float = 0.5) -> dict:
        """买入"""
        cost = price * qty * (1 + self.slippage)
        total_cost = cost * (1 + self.commission)

        if self.cash < total_cost:
            logger.warning(f"[Paper] 资金不足 {ticker}: 需要 ${total_cost:.2f}，剩余 ${self.cash:.2f}")
            return {"success": False, "reason": "资金不足"}

        self.cash -= total_cost

        if ticker in self.positions:
            old = self.positions[ticker]
            total_qty = old["qty"] + qty
            old["avg_cost"] = (old["avg_cost"] * old["qty"] + price * qty) / total_qty
            old["qty"] = total_qty
        else:
            self.positions[ticker] = {"qty": qty, "avg_cost": price}

        trade = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "action": "BUY",
            "quantity": qty,
            "price": price,
            "commission": cost * self.commission,
            "slippage": price * qty * self.slippage,
            "signal": signal,
            "mode": self.mode,
        }
        logger.info(f"[Paper] BUY {ticker} qty={qty:.4f} price={price:.4f} 成本=${total_cost:.2f} 剩余=${self.cash:.2f}")

        if self.storage:
            self.storage.log_trade(**trade)

        return {"success": True, "trade": trade, "cash": self.cash}

    def sell(self, ticker: str, price: float, qty: Optional[float] = None,
             signal: str = "", confidence: float = 0.5) -> dict:
        """卖出（qty=None = 清仓）"""
        if ticker not in self.positions or self.positions[ticker]["qty"] <= 0:
            logger.warning(f"[Paper] 无持仓可卖 {ticker}")
            return {"success": False, "reason": "无持仓"}

        if qty is None:
            qty = self.positions[ticker]["qty"]

        qty = min(qty, self.positions[ticker]["qty"])
        proceeds = price * qty * (1 - self.slippage)
        net_proceeds = proceeds * (1 - self.commission)
        avg_cost = self.positions[ticker]["avg_cost"]
        realized_pnl = (price * (1 - self.slippage) * (1 - self.commission) - avg_cost) * qty

        self.cash += net_proceeds
        self.positions[ticker]["qty"] -= qty
        if self.positions[ticker]["qty"] <= 1e-8:
            del self.positions[ticker]

        trade = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "action": "SELL",
            "quantity": qty,
            "price": price,
            "commission": proceeds * self.commission,
            "slippage": price * qty * self.slippage,
            "signal": signal,
            "realized_pnl": realized_pnl,
            "mode": self.mode,
        }
        logger.info(f"[Paper] SELL {ticker} qty={qty:.4f} price={price:.4f} 净收=${net_proceeds:.2f} 剩余=${self.cash:.2f} 盈亏=${realized_pnl:.2f}")

        if self.storage:
            self.storage.log_trade(**trade)

        return {"success": True, "trade": trade, "cash": self.cash, "realized_pnl": realized_pnl}

    def get_position(self, ticker: str) -> Optional[dict]:
        return self.positions.get(ticker)

    def get_all_positions(self) -> dict:
        return self.positions.copy()

    def get_total_value(self, prices: dict[str, float]) -> float:
        """计算总资产（现金 + 持仓市值）"""
        pos_value = sum(
            self.positions[t]["qty"] * prices.get(t, 0)
            for t in self.positions
        )
        return self.cash + pos_value

    def get_equity_snapshot(self, prices: dict[str, float]) -> dict:
        """当日净值快照"""
        total = self.get_total_value(prices)
        pos_detail = []
        for t, p in self.positions.items():
            cur_price = prices.get(t, p["avg_cost"])
            pos_value = p["qty"] * cur_price
            unrealized = (cur_price - p["avg_cost"]) * p["qty"]
            pos_detail.append({
                "ticker": t,
                "qty": p["qty"],
                "avg_cost": p["avg_cost"],
                "current_price": cur_price,
                "position_value": pos_value,
                "unrealized_pnl": unrealized,
            })
        return {
            "timestamp": datetime.now().isoformat(),
            "cash": self.cash,
            "position_value": sum(p["position_value"] for p in pos_detail),
            "total_equity": total,
            "positions": pos_detail,
        }

    def log_equity(self, prices: dict[str, float] = None):
        """记录净值曲线"""
        if self.storage:
            # 计算持仓市值
            position_value = 0.0
            if prices:
                position_value = sum(
                    self.positions[t]["qty"] * prices.get(t, self.positions[t]["avg_cost"])
                    for t in self.positions
                )
            total_equity = self.cash + position_value
            self.storage.log_equity(
                timestamp=datetime.now().isoformat(),
                total_equity=total_equity,
                cash=self.cash,
                position_value=position_value,
                mode=self.mode,
            )

    def reset(self):
        """重置账户"""
        self.cash = self.initial_cash
        self.positions = {}
        logger.info(f"[Paper] 账户重置，初始资金 ${self.initial_cash:.2f}")


# ============================================================
# MOOMOO OpenAPI 模拟交易（需要安装 moomoo-api SDK）
# ============================================================

class MooMooPaperTrader:
    """
    MOOMOO OpenAPI 模拟账户交易接口。

    使用方法：
    1. 在 MOOMOO OpenAPI 平台申请 API Key（免费）
    2. 设置 config.yaml 中 moomoo.api_key 和 app_secret
    3. pip install moomoo-api（暂未包含在 requirements.txt，按需安装）

    注意：MOOMOO OpenAPI SDK 暂不稳定，此为预留接口，
    实际建议先用 PaperTrader 本地模拟跑通策略逻辑。
    """

    def __init__(self, api_key: str = "", app_secret: str = "",
                 srv_proxy_addr: str = "", storage=None):
        self.api_key = api_key
        self.app_secret = app_secret
        self.srv_proxy_addr = srv_proxy_addr
        self.storage = storage
        self._client = None
        self._paper_mode = True

    def connect(self) -> bool:
        """连接 MOOMOO OpenAPI"""
        try:
            # 预留接口，实际 SDK 接入时实现
            logger.info("[MooMoo] 连接模拟账户...")
            # from moomoo import OpenAPI
            # self._client = OpenAPI(
            #     env=OpenAPI.SANDBOX,   # 模拟环境
            #     api_key=self.api_key,
            #     app_secret=self.app_secret,
            #     proxy=self.srv_proxy_addr,
            # )
            logger.warning("[MooMoo] SDK 未安装，请在 config.yaml 中启用并安装 moomoo-api")
            return False
        except ImportError:
            logger.warning("[MooMoo] moomoo-api 未安装，使用本地 PaperTrader 代替")
            return False

    def get_account_info(self) -> dict:
        """获取账户信息"""
        if not self._client:
            return {}
        # TODO: 调用 SDK 获取账户信息
        return {}

    def get_positions(self) -> list[dict]:
        """获取当前持仓"""
        if not self._client:
            return []
        # TODO: 调用 SDK
        return []

    def order(self, ticker: str, action: str, qty: float,
              order_type: str = "MARKET") -> dict:
        """
        下单。

        参数:
            ticker: 标的代码（如 "AAPL"）
            action: "BUY" / "SELL"
            qty: 数量
            order_type: "MARKET" / "LIMIT"
        """
        if not self._client:
            return {"success": False, "reason": "未连接"}
        # TODO: 调用 SDK
        return {}

    def cancel_order(self, order_id: str) -> dict:
        """撤单"""
        if not self._client:
            return {"success": False, "reason": "未连接"}
        return {}
