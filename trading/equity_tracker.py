"""
trading/equity_tracker.py — 实时净值追踪系统

功能：
    - 分钟级权益曲线记录
    - 实时回撤监控
    - 滚动绩效指标（夏普、胜率、盈亏比）
    - 多维度收益归因
    - 与基准对比（SPY/QQQ）
"""

import logging
import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 数据类定义
# ============================================================

@dataclass
class EquitySnapshot:
    """净值快照"""
    timestamp: datetime
    total_equity: float
    cash: float
    position_value: float
    unrealized_pnl: float
    realized_pnl_today: float
    
    # 持仓明细
    positions: Dict[str, dict]
    
    # 市场状态
    market_regime: str = "unknown"  # bull/bear/sideways
    vix_level: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """绩效指标"""
    timestamp: datetime
    
    # 收益指标
    total_return_pct: float
    annualized_return_pct: float
    
    # 风险指标
    volatility_annual: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    
    # 风险调整收益
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 交易统计
    win_rate_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # 效率指标
    trades_per_day: float
    avg_holding_hours: float

@dataclass
class TradeRecord:
    """完整交易记录"""
    trade_id: str
    timestamp: datetime
    ticker: str
    action: str  # BUY/SELL
    
    # 订单信息
    order_type: str  # MARKET/LIMIT/STOP
    qty: float
    price: float
    filled_qty: float
    filled_price: float
    
    # 信号信息
    signal_source: str  # 策略名称
    signal_confidence: float
    signal_reason: str
    
    # 执行分析
    slippage: float
    commission: float
    expected_price: float
    execution_delay_ms: int
    
    # 盈亏
    realized_pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

# ============================================================
# 净值追踪器
# ============================================================

class EquityTracker:
    """
    实时净值追踪与绩效分析。
    
    功能：
    1. 记录分钟级权益曲线
    2. 实时计算回撤
    3. 滚动更新绩效指标
    4. 收益归因分析
    5. 基准对比
    """
    
    def __init__(self, storage_path: str = "data/equity.db"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        self._equity_cache: List[EquitySnapshot] = []
        self._trade_cache: List[TradeRecord] = []
        
        # 绩效缓存（避免重复计算）
        self._last_metrics: Optional[PerformanceMetrics] = None
        self._last_metrics_time: Optional[datetime] = None
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.storage_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                timestamp TEXT PRIMARY KEY,
                total_equity REAL,
                cash REAL,
                position_value REAL,
                unrealized_pnl REAL,
                realized_pnl_today REAL,
                positions TEXT,
                market_regime TEXT,
                vix_level REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_records (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT,
                ticker TEXT,
                action TEXT,
                order_type TEXT,
                qty REAL,
                price REAL,
                filled_qty REAL,
                filled_price REAL,
                signal_source TEXT,
                signal_confidence REAL,
                signal_reason TEXT,
                slippage REAL,
                commission REAL,
                expected_price REAL,
                execution_delay_ms INTEGER,
                realized_pnl REAL,
                pnl_pct REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TEXT PRIMARY KEY,
                metrics TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    # ---- 权益记录 ----
    
    def record_equity(self, snapshot: EquitySnapshot):
        """记录净值快照"""
        self._equity_cache.append(snapshot)
        
        # 每10条写入一次数据库
        if len(self._equity_cache) >= 10:
            self._flush_equity_cache()
    
    def _flush_equity_cache(self):
        """刷新权益缓存到数据库"""
        if not self._equity_cache:
            return
        
        conn = sqlite3.connect(self.storage_path)
        for snap in self._equity_cache:
            conn.execute("""
                INSERT OR REPLACE INTO equity_curve VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snap.timestamp.isoformat(),
                snap.total_equity,
                snap.cash,
                snap.position_value,
                snap.unrealized_pnl,
                snap.realized_pnl_today,
                json.dumps(snap.positions),
                snap.market_regime,
                snap.vix_level,
            ))
        conn.commit()
        conn.close()
        
        self._equity_cache = []
    
    # ---- 交易记录 ----
    
    def record_trade(self, trade: TradeRecord):
        """记录交易"""
        self._trade_cache.append(trade)
        
        # 立即写入（交易数据重要）
        conn = sqlite3.connect(self.storage_path)
        conn.execute("""
            INSERT OR REPLACE INTO trade_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id,
            trade.timestamp.isoformat(),
            trade.ticker,
            trade.action,
            trade.order_type,
            trade.qty,
            trade.price,
            trade.filled_qty,
            trade.filled_price,
            trade.signal_source,
            trade.signal_confidence,
            trade.signal_reason,
            trade.slippage,
            trade.commission,
            trade.expected_price,
            trade.execution_delay_ms,
            trade.realized_pnl,
            trade.pnl_pct,
        ))
        conn.commit()
        conn.close()
    
    # ---- 查询方法 ----

    def get_equity_curve(self, days: int = 30) -> pd.DataFrame:
        """获取权益曲线"""
        self._flush_equity_cache()
        
        start_time = datetime.now() - timedelta(days=days)
        conn = sqlite3.connect(self.storage_path)
        df = pd.read_sql_query("""
            SELECT * FROM equity_curve 
            WHERE timestamp > ? 
            ORDER BY timestamp
        """, conn, params=(start_time.isoformat(),))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_trade_history(self, days: int = 30, ticker: str = None) -> pd.DataFrame:
        """获取交易历史"""
        start_time = datetime.now() - timedelta(days=days)
        conn = sqlite3.connect(self.storage_path)
        
        if ticker:
            df = pd.read_sql_query("""
                SELECT * FROM trade_records 
                WHERE timestamp > ? AND ticker = ?
                ORDER BY timestamp
            """, conn, params=(start_time.isoformat(), ticker))
        else:
            df = pd.read_sql_query("""
                SELECT * FROM trade_records 
                WHERE timestamp > ?
                ORDER BY timestamp
            """, conn, params=(start_time.isoformat(),))
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    # ---- 绩效计算 ----
    
    def calculate_metrics(self, days: int = 30) -> PerformanceMetrics:
        """计算绩效指标"""
        # 检查缓存
        cache_valid = (
            self._last_metrics and 
            self._last_metrics_time and 
            (datetime.now() - self._last_metrics_time).seconds < 60
        )
        if cache_valid:
            return self._last_metrics
        
        # 获取数据
        equity_df = self.get_equity_curve(days)
        trades_df = self.get_trade_history(days)
        
        if equity_df.empty:
            return self._empty_metrics()
        
        # 计算收益指标
        total_return = (equity_df['total_equity'].iloc[-1] / equity_df['total_equity'].iloc[0]) - 1
        n_days = len(equity_df)
        years = n_days / 252
        annualized = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        
        # 计算波动率
        returns = equity_df['total_equity'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if not returns.empty else 0
        
        # 计算回撤
        cummax = equity_df['total_equity'].cummax()
        drawdown = (equity_df['total_equity'] - cummax) / cummax
        max_dd = drawdown.min()
        current_dd = drawdown.iloc[-1]
        
        # 风险调整收益
        risk_free = 0.045
        sharpe = (annualized - risk_free) / volatility if volatility > 0 else 0
        
        # Sortino（只考虑下行波动）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino = (annualized - risk_free) / downside_std if downside_std > 0 else 0
        
        # Calmar
        calmar = annualized / abs(max_dd) if max_dd != 0 else 0
        
        # 交易统计
        if not trades_df.empty:
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            win_trades = sell_trades[sell_trades['realized_pnl'] > 0]
            
            win_rate = len(win_trades) / len(sell_trades) if len(sell_trades) > 0 else 0
            
            avg_win = win_trades['realized_pnl'].mean() if not win_trades.empty else 0
            loss_trades = sell_trades[sell_trades['realized_pnl'] <= 0]
            avg_loss = abs(loss_trades['realized_pnl'].mean()) if not loss_trades.empty else 0
            
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            trades_per_day = len(trades_df) / max(days, 1)
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            trades_per_day = 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            total_return_pct=total_return * 100,
            annualized_return_pct=annualized * 100,
            volatility_annual=volatility * 100,
            max_drawdown_pct=max_dd * 100,
            current_drawdown_pct=current_dd * 100,
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            win_rate_pct=win_rate * 100,
            profit_factor=round(profit_factor, 3),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            trades_per_day=round(trades_per_day, 2),
            avg_holding_hours=0.0,  # 需要持仓记录计算
        )
        
        # 更新缓存
        self._last_metrics = metrics
        self._last_metrics_time = datetime.now()
        
        # 保存到数据库
        self._save_metrics(metrics)
        
        return metrics
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """空指标"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            total_return_pct=0,
            annualized_return_pct=0,
            volatility_annual=0,
            max_drawdown_pct=0,
            current_drawdown_pct=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            win_rate_pct=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            trades_per_day=0,
            avg_holding_hours=0,
        )
    
    def _save_metrics(self, metrics: PerformanceMetrics):
        """保存指标到数据库"""
        conn = sqlite3.connect(self.storage_path)
        conn.execute("""
            INSERT OR REPLACE INTO performance_metrics VALUES (?, ?)
        """, (
            metrics.timestamp.isoformat(),
            json.dumps(asdict(metrics), default=str),
        ))
        conn.commit()
        conn.close()
    
    # ---- 归因分析 ----
    
    def analyze_attribution(self, days: int = 30) -> Dict:
        """收益归因分析"""
        trades_df = self.get_trade_history(days)
        
        if trades_df.empty:
            return {}
        
        # 按信号源归因
        source_pnl = trades_df.groupby('signal_source')['realized_pnl'].sum().to_dict()
        
        # 按标的归因
        ticker_pnl = trades_df.groupby('ticker')['realized_pnl'].sum().to_dict()
        
        # 按置信度区间归因
        trades_df['confidence_bucket'] = pd.cut(
            trades_df['signal_confidence'], 
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['low', 'medium', 'high', 'very_high']
        )
        confidence_pnl = trades_df.groupby('confidence_bucket')['realized_pnl'].sum().to_dict()
        
        # 滑点分析
        avg_slippage = trades_df['slippage'].mean()
        total_slippage_cost = trades_df['slippage'].sum()
        
        return {
            'by_signal_source': source_pnl,
            'by_ticker': ticker_pnl,
            'by_confidence': confidence_pnl,
            'slippage_analysis': {
                'avg_slippage_per_trade': round(avg_slippage, 4),
                'total_slippage_cost': round(total_slippage_cost, 2),
                'slippage_as_pct_of_pnl': round(total_slippage_cost / (trades_df['realized_pnl'].sum() + 1e-10) * 100, 2),
            },
            'execution_quality': {
                'avg_execution_delay_ms': trades_df['execution_delay_ms'].mean(),
                'filled_vs_expected_ratio': (trades_df['filled_price'] / trades_df['expected_price']).mean(),
            }
        }
    
    # ---- 基准对比 ----
    
    def compare_benchmark(self, benchmark_ticker: str = "SPY", days: int = 30) -> Dict:
        """与基准对比"""
        # 这里需要接入数据获取器
        # 简化实现，返回结构
        return {
            'benchmark_ticker': benchmark_ticker,
            'strategy_return_pct': None,  # 需要计算
            'benchmark_return_pct': None,  # 需要获取基准数据
            'alpha': None,
            'beta': None,
            'information_ratio': None,
            'tracking_error': None,
        }
    
    # ---- 实时监控 ----
    
    def get_realtime_status(self) -> Dict:
        """获取实时状态（用于Dashboard）"""
        equity_df = self.get_equity_curve(days=1)
        metrics = self.calculate_metrics(days=7)
        
        if equity_df.empty:
            return {'status': 'no_data'}
        
        latest = equity_df.iloc[-1]
        
        return {
            'current_equity': latest['total_equity'],
            'cash': latest['cash'],
            'position_value': latest['position_value'],
            'unrealized_pnl': latest['unrealized_pnl'],
            'today_pnl': latest['realized_pnl_today'],
            'current_drawdown_pct': metrics.current_drawdown_pct,
            'sharpe_ratio': metrics.sharpe_ratio,
            'win_rate': metrics.win_rate_pct,
            'trades_today': metrics.trades_per_day,
            'market_regime': latest['market_regime'],
        }
