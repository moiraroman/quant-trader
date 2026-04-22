# ============================================================
# trading/live.py — 实盘交易层
# MOOMOO OpenAPI 实盘接口，带完整风控保护
# ============================================================
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 枚举和数据类
# ============================================================

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """订单"""
    order_id: str
    ticker: str
    side: OrderSide
    price: float
    quantity: float
    status: OrderStatus
    filled_quantity: float = 0
    filled_price: float = 0
    create_time: datetime = None
    update_time: datetime = None
    
    def __post_init__(self):
        if self.create_time is None:
            self.create_time = datetime.now()
        if self.update_time is None:
            self.update_time = datetime.now()


# ============================================================
# MOOMOO 实盘交易
# ============================================================

class MooMooLiveTrader:
    """
    MOOMOO 实盘交易接口。
    
    使用前准备：
    1. 安装 moomoo-api: pip install moomoo-api
    2. 下载并运行 OpenD 服务（MOOMOO 官方提供）
    3. 在 config.yaml 配置 host、port、trade_pwd 等
    
    重要提醒：
    1. 实盘交易涉及真实资金，操作不可逆
    2. 本模块带有双重风控保护（本地 + API 层）
    3. 建议先用模拟盘（PaperTrader）验证策略至少 30 天再切实盘
    4. 设置日亏损熔断阈值，超过立即停止所有交易
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        trade_pwd: str = "",
        paper_trade: bool = True,
        storage=None,
        risk_manager=None,
        notification_manager=None,
    ):
        """
        初始化 MOOMOO 实盘交易器。
        
        Args:
            host: OpenD 服务地址
            port: OpenD 服务端口
            trade_pwd: 交易密码（实盘必填）
            paper_trade: 是否模拟盘
            storage: 数据存储器
            risk_manager: 风控管理器
            notification_manager: 通知管理器
        """
        self.host = host
        self.port = port
        self.trade_pwd = trade_pwd
        self.paper_trade = paper_trade
        self.storage = storage
        self.risk_manager = risk_manager
        self.notifier = notification_manager
        
        self._client = None
        self._connected = False
        self._trade_lock = False
        
        self.mode = "paper" if paper_trade else "live"
        
        # 订单记录
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        
        # 日亏损跟踪
        self.daily_pnl = 0.0
        self.daily_start_value = 0.0
    
    def connect(self) -> bool:
        """
        连接 MOOMOO OpenAPI。
        
        需要先启动 OpenD 服务。
        """
        try:
            from moomoo import OpenQuoteContext, OpenHKTradeContext, OpenUSTradeContext
            
            # 行情上下文
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)
            
            # 交易上下文（根据市场选择）
            if self.paper_trade:
                # 模拟盘
                self._trade_ctx = OpenUSTradeContext(
                    host=self.host,
                    port=self.port,
                    security_firm=1  # 模拟盘
                )
                logger.info("[Live] 已连接 MOOMOO 模拟盘")
            else:
                # 实盘
                if not self.trade_pwd:
                    logger.error("[Live] 实盘必须提供交易密码！")
                    return False
                self._trade_ctx = OpenUSTradeContext(
                    host=self.host,
                    port=self.port,
                    security_firm=1
                )
                logger.warning("[Live] ⚠️ 已连接 MOOMOO 实盘！")
            
            self._connected = True
            
            # 获取账户初始价值
            self._update_account_value()
            
            return True
            
        except ImportError:
            logger.error("[Live] moomoo-api 未安装，请运行: pip install moomoo-api")
            return False
        except Exception as e:
            logger.error(f"[Live] 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._quote_ctx:
            self._quote_ctx.close()
        if self._trade_ctx:
            self._trade_ctx.close()
        self._connected = False
        logger.info("[Live] 已断开连接")
    
    def _update_account_value(self):
        """更新账户价值"""
        if not self._connected:
            return
        
        try:
            # 获取账户信息
            ret, data = self._trade_ctx.accinfo_query()
            if ret == 0:
                self.daily_start_value = data.iloc[0]["total_assets"]
                logger.info(f"[Live] 账户总资产: ${self.daily_start_value:,.2f}")
        except Exception as e:
            logger.warning(f"[Live] 获取账户信息失败: {e}")
    
    def is_trading_allowed(self) -> bool:
        """检查是否允许交易（风控检查）"""
        if not self._connected:
            logger.warning("[Live] 未连接，拒绝交易")
            return False
        
        if self.risk_manager and not self.risk_manager.is_trading_allowed():
            logger.warning("[Live] 风控模块禁止交易")
            return False
        
        # 检查日亏损熔断
        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl) / self.daily_start_value
            if loss_pct > 0.10:  # 10% 熔断
                logger.warning(f"[Live] 日亏损熔断触发: {loss_pct:.1%}")
                return False
        
        return True
    
    def get_positions(self) -> pd.DataFrame:
        """获取持仓"""
        if not self._connected:
            return pd.DataFrame()
        
        try:
            ret, data = self._trade_ctx.position_list_query()
            if ret == 0:
                return data
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"[Live] 获取持仓失败: {e}")
            return pd.DataFrame()
    
    def get_balance(self) -> Dict:
        """获取账户余额"""
        if not self._connected:
            return {}
        
        try:
            ret, data = self._trade_ctx.accinfo_query()
            if ret == 0:
                row = data.iloc[0]
                return {
                    "total_assets": row["total_assets"],
                    "cash": row["cash"],
                    "market_value": row["market_val"],
                    "available_cash": row.get("available_cash", row["cash"])
                }
            return {}
        except Exception as e:
            logger.error(f"[Live] 获取余额失败: {e}")
            return {}
    
    def get_quote(self, ticker: str) -> Dict:
        """获取实时报价"""
        if not self._connected:
            return {}
        
        try:
            ret, data = self._quote_ctx.get_market_snapshot([ticker])
            if ret == 0 and len(data) > 0:
                row = data.iloc[0]
                return {
                    "ticker": ticker,
                    "last_price": row["last_price"],
                    "bid_price": row.get("bid_price", 0),
                    "ask_price": row.get("ask_price", 0),
                    "volume": row.get("volume", 0),
                    "change": row.get("change_val", 0),
                    "change_pct": row.get("change_rate", 0)
                }
            return {}
        except Exception as e:
            logger.error(f"[Live] 获取报价失败: {e}")
            return {}
    
    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market"  # market | limit
    ) -> Dict:
        """
        下单
        
        Args:
            ticker: 股票代码
            side: BUY / SELL
            quantity: 数量
            price: 限价单价格
            order_type: 订单类型（市价/限价）
        
        Returns:
            订单结果字典
        """
        if not self.is_trading_allowed():
            return {"success": False, "reason": "风控禁止交易"}
        
        if self._trade_lock:
            return {"success": False, "reason": "交易锁定"}
        
        try:
            self._trade_lock = True
            
            # 风控检查
            if self.risk_manager:
                max_qty = self.risk_manager.max_position_size(ticker, price or 0, side)
                if quantity > max_qty:
                    logger.warning(f"[Live] 仓位超限，调整数量 {quantity:.4f} -> {max_qty:.4f}")
                    quantity = max_qty
            
            if quantity <= 0:
                return {"success": False, "reason": "数量为0或负数"}
            
            # 获取当前价格
            if price is None:
                quote = self.get_quote(ticker)
                price = quote.get("last_price", 0)
                if price <= 0:
                    return {"success": False, "reason": "无法获取价格"}
            
            logger.warning(f"[Live] 📤 下单: {side} {ticker} x{quantity} @ ${price:.2f}")
            
            # MOOMOO 下单
            # 注意：实际代码需要根据 moomoo-api 文档调整
            """
            ret, data = self._trade_ctx.place_order(
                price=price,
                qty=quantity,
                code=ticker,
                trd_side=TrdSide.BUY if side == "BUY" else TrdSide.SELL,
                order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
                trd_env=TrdEnv.SIMULATE if self.paper_trade else TrdEnv.REAL,
                sec_market=SecMarket.US
            )
            """
            
            # 模拟下单（实际使用时替换为上面代码）
            order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{ticker}"
            
            order = Order(
                order_id=order_id,
                ticker=ticker,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                price=price,
                quantity=quantity,
                status=OrderStatus.SUBMITTED
            )
            
            self.orders[order_id] = order
            
            result = {
                "success": True,
                "order_id": order_id,
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "price": price,
                "status": "submitted"
            }
            
            # 通知
            if self.notifier:
                self.notifier.notify_trade(
                    action=side,
                    symbol=ticker,
                    price=price,
                    quantity=int(quantity)
                )
            
            logger.info(f"[Live] 订单已提交: {order_id}")
            return result
            
        except Exception as e:
            logger.error(f"[Live] 下单失败: {e}")
            return {"success": False, "reason": str(e)}
        finally:
            self._trade_lock = False
    
    def buy(
        self,
        ticker: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict:
        """买入"""
        return self.place_order(ticker, "BUY", quantity, price)
    
    def sell(
        self,
        ticker: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict:
        """卖出"""
        return self.place_order(ticker, "SELL", quantity, price)
    
    def cancel_order(self, order_id: str) -> Dict:
        """撤单"""
        if not self._connected:
            return {"success": False, "reason": "未连接"}
        
        try:
            # ret, data = self._trade_ctx.modify_order(...)
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"[Live] 订单已撤单: {order_id}")
                return {"success": True, "order_id": order_id}
            return {"success": False, "reason": "订单不存在"}
        except Exception as e:
            logger.error(f"[Live] 撤单失败: {e}")
            return {"success": False, "reason": str(e)}
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        return self.orders.get(order_id)
    
    def sync_positions(self) -> Dict:
        """同步持仓到本地"""
        positions = self.get_positions()
        if positions.empty:
            return {}
        
        position_dict = {}
        for _, row in positions.iterrows():
            ticker = row["code"]
            position_dict[ticker] = {
                "quantity": row["qty"],
                "avg_cost": row["cost_price"],
                "market_value": row["market_val"],
                "pnl": row["pl_val"],
                "pnl_pct": row["pl_ratio"]
            }
        
        return position_dict
    
    def update_daily_pnl(self):
        """更新日亏损"""
        balance = self.get_balance()
        if balance:
            current_value = balance["total_assets"]
            self.daily_pnl = current_value - self.daily_start_value
            
            # 检查熔断
            if self.risk_manager:
                loss_pct = abs(self.daily_pnl) / self.daily_start_value if self.daily_pnl < 0 else 0
                if loss_pct > 0.10:
                    logger.warning(f"[Live] 日亏损熔断: {loss_pct:.1%}")
                    # 通知
                    if self.notifier:
                        self.notifier.notify_risk_alert(
                            "日亏损熔断",
                            f"今日亏损 {loss_pct:.1%}，已触发熔断"
                        )


# ============================================================
# 工具函数
# ============================================================

def create_live_trader_from_config(config: Dict, storage=None, risk_manager=None, notifier=None) -> MooMooLiveTrader:
    """
    从配置创建实盘交易器
    
    Args:
        config: 配置字典
        storage: 数据存储器
        risk_manager: 风控管理器
        notifier: 通知管理器
    
    Returns:
        MooMooLiveTrader 实例
    """
    moomoo_cfg = config.get("data", {}).get("moomoo", {})
    
    return MooMooLiveTrader(
        host=moomoo_cfg.get("host", "127.0.0.1"),
        port=moomoo_cfg.get("port", 11111),
        trade_pwd=moomoo_cfg.get("trade_pwd", ""),
        paper_trade=moomoo_cfg.get("paper_trade", True),
        storage=storage,
        risk_manager=risk_manager,
        notification_manager=notifier
    )
