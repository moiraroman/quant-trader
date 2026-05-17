"""
trading/order_manager.py — 订单管理系统

功能：
    - 多订单类型支持（市价/限价/止损/止损限价）
    - 订单状态生命周期管理
    - 部分成交处理
    - 订单簿模拟
    - 执行质量分析
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 枚举定义
# ============================================================

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"           # 已提交，待处理
    OPEN = "OPEN"                 # 已挂到订单簿
    PARTIAL_FILLED = "PARTIAL"    # 部分成交
    FILLED = "FILLED"             # 完全成交
    CANCELLED = "CANCELLED"       # 已撤销
    REJECTED = "REJECTED"         # 被拒绝
    EXPIRED = "EXPIRED"           # 已过期

# ============================================================
# 数据类定义
# ============================================================

@dataclass
class Order:
    """订单对象"""
    # 基本信息
    order_id: str
    ticker: str
    action: OrderAction
    order_type: OrderType
    
    # 数量
    qty: float
    filled_qty: float = 0.0
    remaining_qty: float = field(init=False)
    
    # 价格
    price: Optional[float] = None           # 限价单价格
    stop_price: Optional[float] = None      # 止损触发价
    trailing_pct: Optional[float] = None    # 移动止损百分比
    
    # 状态
    status: OrderStatus = OrderStatus.PENDING
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # 执行结果
    avg_filled_price: Optional[float] = None
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # 元数据
    signal_source: str = ""
    signal_confidence: float = 0.0
    signal_reason: str = ""
    
    # 条件订单
    expire_at: Optional[datetime] = None
    gtc: bool = False  # Good Till Cancelled
    
    def __post_init__(self):
        self.remaining_qty = self.qty - self.filled_qty
    
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'ticker': self.ticker,
            'action': self.action.value,
            'order_type': self.order_type.value,
            'qty': self.qty,
            'filled_qty': self.filled_qty,
            'remaining_qty': self.remaining_qty,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'avg_filled_price': self.avg_filled_price,
            'total_commission': self.total_commission,
            'signal_source': self.signal_source,
            'signal_confidence': self.signal_confidence,
        }

@dataclass
class Fill:
    """成交记录"""
    fill_id: str
    order_id: str
    ticker: str
    action: str
    qty: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    
    # 执行分析
    expected_price: float
    execution_delay_ms: int

# ============================================================
# 订单簿（模拟交易所订单簿）
# ============================================================

class OrderBook:
    """
    模拟订单簿，处理订单匹配。
    
    简化模型：
    - 市价单：立即以当前价格成交（+滑点）
    - 限价单：价格触达时成交
    - 止损单：价格触达时转为市价单
    """
    
    def __init__(self, slippage_model: str = "fixed"):
        self.bids: Dict[str, List[Order]] = {}   # ticker -> buy orders
        self.asks: Dict[str, List[Order]] = {}   # ticker -> sell orders
        self.slippage_model = slippage_model
        
        # 滑点模型参数
        self.fixed_slippage_pct = 0.0005  # 0.05%
        self.volatility_slippage_factor = 0.1
    
    def add_order(self, order: Order) -> List[Fill]:
        """添加订单到订单簿，返回成交记录"""
        fills = []
        
        if order.order_type == OrderType.MARKET:
            # 市价单立即成交
            fill = self._execute_market_order(order)
            if fill:
                fills.append(fill)
        
        elif order.order_type == OrderType.LIMIT:
            # 限价单挂到订单簿
            self._add_to_book(order)
            order.status = OrderStatus.OPEN
        
        elif order.order_type == OrderType.STOP:
            # 止损单挂到监控列表
            self._add_to_book(order)
            order.status = OrderStatus.OPEN
        
        elif order.order_type == OrderType.TRAILING_STOP:
            # 移动止损
            self._add_to_book(order)
            order.status = OrderStatus.OPEN
        
        return fills
    
    def _execute_market_order(self, order: Order) -> Optional[Fill]:
        """执行市价单"""
        # 模拟市价单成交（实际应查询当前市场价）
        # 这里简化处理，假设以 order.price 或市场价成交
        
        fill_price = order.price or 100.0  # 简化
        slippage = fill_price * self.fixed_slippage_pct
        
        if order.action == OrderAction.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage
        
        commission = fill_price * order.qty * 0.001  # 0.1% 手续费
        
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            ticker=order.ticker,
            action=order.action.value,
            qty=order.qty,
            price=round(fill_price, 4),
            commission=round(commission, 4),
            slippage=round(slippage, 4),
            timestamp=datetime.now(),
            expected_price=order.price or fill_price,
            execution_delay_ms=np.random.randint(50, 500),  # 模拟延迟
        )
        
        order.filled_qty = order.qty
        order.remaining_qty = 0
        order.avg_filled_price = fill_price
        order.total_commission = commission
        order.total_slippage = slippage
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        
        return fill
    
    def _add_to_book(self, order: Order):
        """添加订单到订单簿"""
        ticker = order.ticker
        
        if order.action == OrderAction.BUY:
            if ticker not in self.bids:
                self.bids[ticker] = []
            self.bids[ticker].append(order)
        else:
            if ticker not in self.asks:
                self.asks[ticker] = []
            self.asks[ticker].append(order)
    
    def check_triggers(self, ticker: str, current_price: float) -> List[Fill]:
        """
        检查价格触发（止损单、限价单）。
        返回触发的成交记录。
        """
        fills = []
        
        # 检查止损单（买单：价格 >= stop_price；卖单：价格 <= stop_price）
        for side in [self.bids, self.asks]:
            if ticker not in side:
                continue
            
            triggered = []
            for order in side[ticker]:
                if order.order_type == OrderType.STOP and order.stop_price:
                    if order.action == OrderAction.BUY and current_price >= order.stop_price:
                        triggered.append(order)
                    elif order.action == OrderAction.SELL and current_price <= order.stop_price:
                        triggered.append(order)
                
                elif order.order_type == OrderType.LIMIT and order.price:
                    if order.action == OrderAction.BUY and current_price <= order.price:
                        triggered.append(order)
                    elif order.action == OrderAction.SELL and current_price >= order.price:
                        triggered.append(order)
                
                elif order.order_type == OrderType.TRAILING_STOP:
                    # 移动止损逻辑
                    if order.stop_price:
                        if order.action == OrderAction.SELL and current_price <= order.stop_price:
                            triggered.append(order)
            
            for order in triggered:
                fill = self._execute_market_order(order)
                if fill:
                    fills.append(fill)
                side[ticker].remove(order)
        
        return fills
    
    def update_trailing_stops(self, ticker: str, current_price: float):
        """更新移动止损价格"""
        if ticker not in self.asks:
            return
        
        for order in self.asks[ticker]:
            if order.order_type == OrderType.TRAILING_STOP and order.trailing_pct:
                # 计算新的止损价：最高价 × (1 - trailing_pct)
                # 简化：假设 order.price 存储了最高价
                highest = order.price or current_price
                if current_price > highest:
                    order.price = current_price
                    new_stop = current_price * (1 - order.trailing_pct)
                    if new_stop > order.stop_price:
                        order.stop_price = new_stop
                        logger.info(f"[OrderBook] {ticker} 移动止损上调至 ${new_stop:.2f}")

# ============================================================
# 订单管理器
# ============================================================

class OrderManager:
    """
    订单管理器：统一管理所有订单生命周期。
    """
    
    def __init__(self, paper_trader=None):
        self.order_book = OrderBook()
        self.paper_trader = paper_trader
        
        # 订单存储
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        
        # 回调
        self.on_fill: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None
    
    # ---- 下单接口 ----
    
    def submit_order(
        self,
        ticker: str,
        action: str,  # "BUY" or "SELL"
        qty: float,
        order_type: str = "MARKET",
        price: float = None,
        stop_price: float = None,
        trailing_pct: float = None,
        signal_source: str = "",
        signal_confidence: float = 0.0,
        signal_reason: str = "",
    ) -> Order:
        """
        提交订单。
        
        示例：
            # 市价买入
            order = om.submit_order("AAPL", "BUY", 100)
            
            # 限价买入
            order = om.submit_order("AAPL", "BUY", 100, "LIMIT", price=150.0)
            
            # 止损卖出
            order = om.submit_order("AAPL", "SELL", 100, "STOP", stop_price=140.0)
            
            # 移动止损（5% trailing）
            order = om.submit_order("AAPL", "SELL", 100, "TRAILING_STOP", trailing_pct=0.05)
        """
        # Handle both string and OrderType enum inputs
        if isinstance(order_type, OrderType):
            ot = order_type
        else:
            ot = OrderType(order_type.upper())
        
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            action=OrderAction(action.upper()),
            order_type=ot,
            qty=qty,
            price=price,
            stop_price=stop_price,
            trailing_pct=trailing_pct,
            signal_source=signal_source,
            signal_confidence=signal_confidence,
            signal_reason=signal_reason,
        )
        
        order.submitted_at = datetime.now()
        self.orders[order.order_id] = order
        
        # 提交到订单簿
        fills = self.order_book.add_order(order)
        
        # 处理成交
        for fill in fills:
            self._process_fill(fill)
        
        logger.info(
            f"[OrderManager] 提交订单 {order.order_id}: {action} {ticker} "
            f"qty={qty} type={order_type} status={order.status.value}"
        )
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        if order_id not in self.orders:
            logger.warning(f"[OrderManager] 订单不存在: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            logger.warning(f"[OrderManager] 订单无法撤销，状态: {order.status.value}")
            return False
        
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()
        
        # 从订单簿移除
        self._remove_from_book(order)
        
        logger.info(f"[OrderManager] 撤销订单 {order_id}")
        
        if self.on_order_update:
            self.on_order_update(order)
        
        return True
    
    def cancel_all_orders(self, ticker: str = None) -> int:
        """撤销所有订单（或指定标的）"""
        cancelled = 0
        for order in list(self.orders.values()):
            if order.status in [OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
                if ticker is None or order.ticker == ticker:
                    if self.cancel_order(order.order_id):
                        cancelled += 1
        return cancelled
    
    # ---- 价格更新 ----
    
    def on_price_update(self, ticker: str, current_price: float):
        """
        接收价格更新，检查触发订单。
        应由 Bot 定时调用。
        """
        # 检查止损/限价触发
        fills = self.order_book.check_triggers(ticker, current_price)
        for fill in fills:
            self._process_fill(fill)
        
        # 更新移动止损
        self.order_book.update_trailing_stops(ticker, current_price)
    
    # ---- 内部方法 ----
    
    def _process_fill(self, fill: Fill):
        """处理成交"""
        self.fills.append(fill)
        
        # 更新订单状态
        if fill.order_id in self.orders:
            order = self.orders[fill.order_id]
            order.filled_qty += fill.qty
            order.remaining_qty -= fill.qty
            
            if order.remaining_qty <= 0:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
            else:
                order.status = OrderStatus.PARTIAL_FILLED
        
        # 执行 PaperTrader 交易
        if self.paper_trader:
            try:
                if fill.action == "BUY":
                    self.paper_trader.buy(
                        ticker=fill.ticker,
                        price=fill.price,
                        qty=fill.qty,
                    )
                else:
                    self.paper_trader.sell(
                        ticker=fill.ticker,
                        price=fill.price,
                        qty=fill.qty,
                    )
            except Exception as e:
                logger.error(f"[OrderManager] PaperTrader 执行失败: {e}")
        
        # 回调
        if self.on_fill:
            self.on_fill(fill)
        
        logger.info(
            f"[OrderManager] 成交 {fill.fill_id}: {fill.action} {fill.ticker} "
            f"qty={fill.qty} price=${fill.price:.2f}"
        )
    
    def _remove_from_book(self, order: Order):
        """从订单簿移除订单"""
        side = self.order_book.bids if order.action == OrderAction.BUY else self.order_book.asks
        if order.ticker in side and order in side[order.ticker]:
            side[order.ticker].remove(order)
    
    # ---- 查询接口 ----
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)
    
    def get_orders(self, ticker: str = None, status: str = None) -> List[Order]:
        """查询订单"""
        result = list(self.orders.values())
        if ticker:
            result = [o for o in result if o.ticker == ticker]
        if status:
            result = [o for o in result if o.status.value == status]
        return result
    
    def get_open_orders(self, ticker: str = None) -> List[Order]:
        """获取未完成订单"""
        return self.get_orders(ticker, status="OPEN")
    
    def get_fills(self, order_id: str = None, ticker: str = None) -> List[Fill]:
        """获取成交记录"""
        result = self.fills
        if order_id:
            result = [f for f in result if f.order_id == order_id]
        if ticker:
            result = [f for f in result if f.ticker == ticker]
        return result
    
    def get_position_summary(self) -> Dict:
        """获取持仓汇总（基于成交记录）"""
        position_qty = {}
        position_cost = {}
        
        for fill in self.fills:
            ticker = fill.ticker
            if fill.action == "BUY":
                position_qty[ticker] = position_qty.get(ticker, 0) + fill.qty
                position_cost[ticker] = position_cost.get(ticker, 0) + fill.qty * fill.price
            else:
                position_qty[ticker] = position_qty.get(ticker, 0) - fill.qty
        
        summary = {}
        for ticker, qty in position_qty.items():
            if qty > 0:
                avg_cost = position_cost.get(ticker, 0) / qty if qty > 0 else 0
                summary[ticker] = {
                    'qty': qty,
                    'avg_cost': round(avg_cost, 4),
                }
        
        return summary
