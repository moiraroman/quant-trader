# ============================================================
# data/storage.py — 本地数据存储模块
# 支持：Parquet（推荐）/ SQLite / CSV
# 自动去重、自动分区、按 ticker 管理
# ============================================================
import os
import time
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# Parquet 存储（主存储，推荐）
# ============================================================

class ParquetStorage:
    """
    按 ticker/interval 组织目录结构：
      data_cache/
      ├── AAPL_1d.parquet
      ├── AAPL_1h.parquet
      ├── TSLA_1d.parquet
      └── ...

    自动按日期追加（append），不重复覆盖。
    """

    def __init__(self, cache_dir: str = "data_cache/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, interval: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}_{interval}.parquet"

    def load(self, ticker: str, interval: str = "1d") -> pd.DataFrame:
        """加载已有数据（若缓存未过期则直接返回）"""
        p = self._path(ticker, interval)
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            return df
        return pd.DataFrame()

    def save(self, df: pd.DataFrame, ticker: str, interval: str = "1d"):
        """追加保存数据（合并已有数据 + 新数据，自动去重）"""
        if df.empty:
            return
        p = self._path(ticker, interval)

        # 读取旧数据并合并去重
        if p.exists():
            old = pd.read_parquet(p)
            old.index = pd.to_datetime(old.index)
            df = pd.concat([old, df]).drop_duplicates().sort_index()
            df = df[~df.index.duplicated(keep="last")]

        df.to_parquet(p, index=True)
        logger.info(f"[ParquetStorage] 已保存 {ticker}_{interval} {len(df)} 条 -> {p}")

    def clear(self, ticker: Optional[str] = None, interval: Optional[str] = None):
        """清除缓存"""
        if ticker and interval:
            p = self._path(ticker, interval)
            if p.exists():
                p.unlink()
                logger.info(f"[ParquetStorage] 已删除 {p}")
        elif ticker:
            for p in self.cache_dir.glob(f"{ticker.upper()}_*.parquet"):
                p.unlink()
                logger.info(f"[ParquetStorage] 已删除 {p}")
        else:
            for p in self.cache_dir.glob("*.parquet"):
                p.unlink()
            logger.info(f"[ParquetStorage] 已清空全部缓存")

    def is_expired(self, ticker: str, interval: str, expire_hours: int = 24) -> bool:
        """检查缓存是否过期"""
        p = self._path(ticker, interval)
        if not p.exists():
            return True
        age = time.time() - p.stat().st_mtime
        return age > expire_hours * 3600

    def list_cached(self) -> list[str]:
        """列出所有缓存的 ticker"""
        files = self.cache_dir.glob("*_*.parquet")
        tickers = sorted({f.stem.rsplit("_", 1)[0] for f in files})
        return tickers


# ============================================================
# SQLite 持仓/交易记录存储
# ============================================================

class SQLiteStorage:
    """
    存储交易记录、持仓快照、信号日志。
    表结构：
      - trades: 每笔成交记录
      - positions: 当前持仓快照
      - signals: 每日策略信号
      - equity_curve: 每日净值记录
    """

    def __init__(self, db_path: str = "data_cache/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,   -- BUY / SELL
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    signal TEXT,
                    mode TEXT DEFAULT 'backtest'  -- backtest / paper / live
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL UNIQUE,
                    quantity REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    mode TEXT DEFAULT 'backtest'
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal TEXT NOT NULL,   -- BUY / SELL / HOLD
                    confidence REAL,
                    price REAL,
                    reason TEXT
                );

                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL UNIQUE,
                    total_equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    position_value REAL DEFAULT 0,
                    mode TEXT DEFAULT 'backtest'
                );
            """)
            conn.commit()
        logger.info(f"[SQLiteStorage] 数据库初始化完成: {self.db_path}")

    # ---- 交易记录 ----

    def log_trade(self, **kwargs):
        cols = ["timestamp", "ticker", "action", "quantity", "price",
                "commission", "slippage", "signal", "mode"]
        vals = {k: kwargs.get(k, 0 if k in ("commission", "slippage", "quantity", "price") else "") for k in cols}
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join(['?' for _ in cols])})",
                [vals[c] for c in cols]
            )
            conn.commit()
        logger.info(f"[SQLite] 记录交易: {vals['timestamp']} {vals['action']} {vals['ticker']} qty={vals['quantity']} price={vals['price']}")

    def get_trades(self, mode: str = "backtest") -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql("SELECT * FROM trades WHERE mode=? ORDER BY timestamp", conn, params=(mode,))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ---- 信号记录 ----

    def log_signal(self, **kwargs):
        cols = ["timestamp", "ticker", "strategy", "signal", "confidence", "price", "reason"]
        vals = {k: kwargs.get(k, None) for k in cols}
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO signals ({','.join(cols)}) VALUES ({','.join(['?' for _ in cols])})",
                [vals[c] for c in cols]
            )
            conn.commit()

    def get_signals(self, ticker: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        sql = "SELECT * FROM signals"
        params = []
        if ticker:
            sql += " WHERE ticker=?"
            params.append(ticker)
        sql += f" ORDER BY timestamp DESC LIMIT {limit}"
        with self._conn() as conn:
            df = pd.read_sql(sql, conn, params=params if params else None)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ---- 净值曲线 ----

    def log_equity(self, **kwargs):
        cols = ["timestamp", "total_equity", "cash", "position_value", "mode"]
        vals = {}
        for k in cols:
            if k in kwargs:
                vals[k] = kwargs[k]
            elif k in ("total_equity", "cash", "position_value"):
                vals[k] = 0.0
            else:
                vals[k] = ""
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO equity_curve ({','.join(cols)}) VALUES ({','.join(['?' for _ in cols])})",
                [vals[c] for c in cols]
            )
            conn.commit()

    def get_equity_curve(self, mode: str = "backtest") -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql(
                "SELECT * FROM equity_curve WHERE mode=? ORDER BY timestamp",
                conn, params=(mode,)
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ---- 持仓快照 ----

    def update_position(self, **kwargs):
        cols = ["timestamp", "ticker", "quantity", "avg_cost", "current_price", "unrealized_pnl", "mode"]
        vals = {k: kwargs.get(k, 0) for k in cols}
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO positions ({','.join(cols)}) VALUES ({','.join(['?' for _ in cols])})",
                [vals[c] for c in cols]
            )
            conn.commit()

    def get_positions(self, mode: str = "backtest") -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql("SELECT * FROM positions WHERE mode=? ORDER BY ticker", conn, params=(mode,))
        return df

    def get_position(self, ticker: str, mode: str = "backtest") -> Optional[dict]:
        with self._conn() as conn:
            df = pd.read_sql(
                "SELECT * FROM positions WHERE ticker=? AND mode=?",
                conn, params=(ticker, mode)
            )
        return df.iloc[0].to_dict() if not df.empty else None

    def clear_table(self, table: str):
        # 白名单验证防止SQL注入
        valid_tables = {"trades", "positions", "signals", "equity_curve", "risk_state"}
        if table not in valid_tables:
            logger.error(f"[SQLite] 清空表失败: '{table}' 不是合法的表名")
            raise ValueError(f"非法表名: {table}，合法表名: {valid_tables}")
        with self._conn() as conn:
            conn.execute(f"DELETE FROM {table}")
            conn.commit()
        logger.info(f"[SQLite] 清空表: {table}")
