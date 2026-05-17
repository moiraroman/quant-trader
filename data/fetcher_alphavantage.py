# ============================================================
# data/fetcher_alphavantage.py — Alpha Vantage 数据源
# 特点：免费 25次/天、支持分钟级数据、需 API Key
# 文档：https://www.alphavantage.co/documentation/
# ============================================================
import logging
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from .fetcher_base import BaseDataFetcher, Quote, normalize_ohlc

logger = logging.getLogger(__name__)


class AlphaVantageFetcher(BaseDataFetcher):
    """
    Alpha Vantage 数据获取器。

    免费套餐限制:
        - 25 次/天
        - 5 次/分钟

    优点:
        - 支持分钟级数据（1min/5min/15min/30min/60min）
        - 支持加密货币、外汇
        - 技术指标 API（内置 RSI/MACD 等）

    缺点:
        - 免费额度有限
        - 请求频率受限

    适用场景:
        - yfinance/Stooq 失败时的备用
        - 需要分钟级数据时
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        """
        参数:
            api_key: Alpha Vantage API Key
                - 可从环境变量 ALPHAVANTAGE_API_KEY 读取
                - 免费申请：https://www.alphavantage.co/support/#api-key
        """
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not self.api_key:
            logger.warning("[AlphaVantage] 未设置 API Key，请设置 ALPHAVANTAGE_API_KEY 环境变量")

        super().__init__(name="AlphaVantage", requests_per_day=25)
        self._session = requests.Session()
        self._last_request_time = 0
        self._min_interval = 12  # 5次/分钟 = 每12秒1次

    def fetch_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        获取历史K线。

        参数:
            ticker: 股票代码
            period: 时间范围（Alpha Vantage 用 outputsize 控制）
            interval: "1d"/"1w"/"1mo" 或 "1min"/"5min"/"15min"/"30min"/"60min"
        """
        if not self.api_key:
            logger.warning("[AlphaVantage] 无 API Key，跳过")
            return pd.DataFrame()

        if self.is_rate_limited():
            logger.warning(f"[AlphaVantage] 已达每日限制 ({self.requests_per_day}/天)")
            return pd.DataFrame()

        self._rate_limit_wait()

        function, av_interval, outputsize = self._map_params(interval, period)

        params = {
            "function": function,
            "symbol": ticker.upper(),
            "apikey": self.api_key,
            "outputsize": outputsize,
        }
        if av_interval:
            params["interval"] = av_interval

        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # 检查错误
            if "Error Message" in data:
                logger.warning(f"[AlphaVantage] {ticker} 错误: {data['Error Message']}")
                self._record_failure(data["Error Message"])
                return pd.DataFrame()

            if "Note" in data:
                # 频率限制提示
                logger.warning(f"[AlphaVantage] 频率限制: {data['Note']}")
                self._record_failure("Rate limit")
                return pd.DataFrame()

            # 解析数据
            df = self._parse_response(data, interval)
            if df.empty:
                return df

            df = normalize_ohlc(df, source="AlphaVantage")
            self._record_success()
            logger.info(f"[AlphaVantage] {ticker} 获取 {len(df)} 条记录")
            return df

        except Exception as e:
            self._record_failure(str(e))
            logger.error(f"[AlphaVantage] {ticker} 失败: {e}")
            return pd.DataFrame()

    def fetch_quote(self, ticker: str) -> Optional[Quote]:
        """获取实时报价（GLOBAL_QUOTE）"""
        if not self.api_key:
            return None

        if self.is_rate_limited():
            return None

        self._rate_limit_wait()

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker.upper(),
            "apikey": self.api_key,
        }

        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "Global Quote" not in data or not data["Global Quote"]:
                logger.warning(f"[AlphaVantage] {ticker} 无报价数据")
                return None

            q = data["Global Quote"]

            return Quote(
                ticker=ticker,
                last_price=float(q.get("05. price", 0)),
                previous_close=float(q.get("08. previous close", 0)),
                open_price=float(q.get("02. open", 0)),
                high_price=float(q.get("03. high", 0)),
                low_price=float(q.get("04. low", 0)),
                volume=int(q.get("06. volume", 0)),
                currency="USD",
                exchange="US",
                timestamp=datetime.now().isoformat(),
                source="AlphaVantage",
            )
        except Exception as e:
            logger.error(f"[AlphaVantage] {ticker} 报价失败: {e}")
            return None

    def _map_params(self, interval: str, period: str) -> tuple[str, Optional[str], str]:
        """映射参数到 Alpha Vantage 格式"""
        # function, interval, outputsize
        outputsize = "full" if period in ["5y", "10y"] else "compact"

        if interval in ["1min", "5min", "15min", "30min", "60min"]:
            return ("TIME_SERIES_INTRADAY", interval, outputsize)
        elif interval in ["1w", "weekly"]:
            return ("TIME_SERIES_WEEKLY", None, outputsize)
        elif interval in ["1mo", "monthly"]:
            return ("TIME_SERIES_MONTHLY", None, outputsize)
        else:
            return ("TIME_SERIES_DAILY", None, outputsize)

    def _parse_response(self, data: dict, interval: str) -> pd.DataFrame:
        """解析 API 响应"""
        # 确定时间序列键名
        ts_key = None
        for key in data:
            if "Time Series" in key:
                ts_key = key
                break

        if not ts_key:
            logger.warning("[AlphaVantage] 未找到时间序列数据")
            return pd.DataFrame()

        ts = data[ts_key]
        if not ts:
            return pd.DataFrame()

        # 转为 DataFrame
        df = pd.DataFrame.from_dict(ts, orient="index")

        # 列名清理（去除数字前缀）
        df.columns = [c.split(". ")[-1].lower() for c in df.columns]

        # 重命名
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        # 转换类型
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 索引处理
        df.index = pd.to_datetime(df.index)

        return df

    def _rate_limit_wait(self):
        """等待满足请求间隔"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
