# ============================================================
# data/fetcher_base.py — 数据获取器抽象基类
# 所有数据源实现此接口，确保统一调用方式
# ============================================================
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """实时报价数据结构"""
    ticker: str
    last_price: float
    previous_close: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    market_cap: Optional[float] = None
    currency: str = "USD"
    exchange: str = "US"
    timestamp: str = ""
    source: str = ""


@dataclass
class DataSourceStatus:
    """数据源状态"""
    name: str
    is_available: bool
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_reason: str = ""
    requests_today: int = 0
    requests_limit: int = 0


class BaseDataFetcher(ABC):
    """
    数据获取器抽象基类。

    所有数据源必须实现：
        - fetch_history(): 获取历史K线
        - fetch_quote(): 获取实时报价
        - get_status(): 返回数据源状态
    """

    def __init__(self, name: str, requests_per_day: int = 0):
        """
        参数:
            name: 数据源名称
            requests_per_day: 每日请求限制（0=无限制）
        """
        self.name = name
        self.requests_per_day = requests_per_day
        self._requests_today = 0
        self._last_success = None
        self._last_failure = None
        self._failure_reason = ""
        self._is_available = True

    @abstractmethod
    def fetch_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        获取历史K线数据。

        返回 DataFrame 必须包含以下列（标准化）:
            - Open: 开盘价
            - High: 最高价
            - Low: 最低价
            - Close: 收盘价
            - Volume: 成交量

        索引: DatetimeIndex（无时区）
        """
        pass

    @abstractmethod
    def fetch_quote(self, ticker: str) -> Optional[Quote]:
        """获取实时报价"""
        pass

    def get_status(self) -> DataSourceStatus:
        """返回数据源当前状态"""
        return DataSourceStatus(
            name=self.name,
            is_available=self._is_available,
            last_success=self._last_success,
            last_failure=self._last_failure,
            failure_reason=self._failure_reason,
            requests_today=self._requests_today,
            requests_limit=self.requests_per_day,
        )

    def _record_success(self):
        """记录成功请求"""
        self._last_success = datetime.now()
        self._requests_today += 1

    def _record_failure(self, reason: str):
        """记录失败"""
        self._last_failure = datetime.now()
        self._failure_reason = reason
        # 连续失败则临时标记不可用
        # 下次请求会重试

    def is_rate_limited(self) -> bool:
        """检查是否超出每日限制"""
        if self.requests_per_day == 0:
            return False
        return self._requests_today >= self.requests_per_day

    def reset_daily_counter(self):
        """重置每日计数器（供定时任务调用）"""
        self._requests_today = 0
        logger.info(f"[{self.name}] 每日请求计数已重置")


# ============================================================
# 工具函数
# ============================================================

def normalize_ohlc(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    标准化 OHLCV DataFrame。
    确保列名为大写，索引为无时区的 DatetimeIndex。
    """
    if df.empty:
        return df

    # 列名标准化
    column_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj close": "Adj Close",
        "adj_close": "Adj Close",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns.str.lower()})

    # 确保必需列存在
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            if col == "Volume":
                df[col] = 0  # 无成交量则填0
            else:
                logger.warning(f"[normalize_ohlc] 缺少列 {col}，来源: {source}")
                return pd.DataFrame()

    # 索引处理
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.warning(f"[normalize_ohlc] 索引转换失败: {e}")
            return pd.DataFrame()

    # 去时区
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 排序
    df = df.sort_index()

    return df[required]
