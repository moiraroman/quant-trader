# ============================================================
# data/fetcher_stooq.py — Stooq 数据源
# 特点：免费、无API Key限制、CSV直下、数据干净
# 限制：仅日线及以上，无实时报价（延迟数据）
# ============================================================
import io
import logging
import zipfile
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from .fetcher_base import BaseDataFetcher, Quote, normalize_ohlc

logger = logging.getLogger(__name__)


class StooqFetcher(BaseDataFetcher):
    """
    Stooq 数据获取器。

    优点:
        - 完全免费，无需 API Key
        - 无请求频率限制
        - 数据质量高（来自交易所）

    缺点:
        - 仅支持日线及以上周期（无 intraday）
        - 无实时报价（返回前一日收盘）
        - 部分小盘股可能缺失

    适用场景:
        - yfinance 失败时的备用源
        - 批量下载历史数据
    """

    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(self):
        super().__init__(name="Stooq", requests_per_day=0)  # 无限制
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def fetch_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        获取历史K线。

        参数:
            ticker: 股票代码（如 "AAPL"、"SPY"）
            period: 时间范围（"1mo"/"3mo"/"6mo"/"1y"/"2y"/"5y"）
            interval: 仅支持 "1d"、"1w"、"1m"（月线）
        """
        # Stooq 使用美国股票加 .US 后缀
        symbol = f"{ticker.upper()}.US"

        # 计算日期范围
        end_date = datetime.now()
        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
        }
        days = period_days.get(period, 365)
        start_date = end_date - timedelta(days=days)

        params = {
            "s": symbol,
            "i": self._map_interval(interval),
            "d1": start_date.strftime("%Y%m%d"),
            "d2": end_date.strftime("%Y%m%d"),
        }

        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()

            # 解析 CSV
            df = pd.read_csv(io.StringIO(resp.text))

            if df.empty or "Date" not in df.columns:
                logger.warning(f"[Stooq] {ticker} 无数据")
                return pd.DataFrame()

            # 标准化
            df = df.rename(columns={"Date": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            # 列名映射
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })

            df = normalize_ohlc(df, source="Stooq")
            self._record_success()
            logger.info(f"[Stooq] {ticker} 获取 {len(df)} 条记录")
            return df

        except Exception as e:
            self._record_failure(str(e))
            logger.error(f"[Stooq] {ticker} 失败: {e}")
            return pd.DataFrame()

    def fetch_quote(self, ticker: str) -> Optional[Quote]:
        """
        Stooq 不提供实时报价。
        返回最新历史收盘价作为"报价"。
        """
        try:
            df = self.fetch_history(ticker, period="1mo", interval="1d")
            if df.empty:
                return None

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            return Quote(
                ticker=ticker,
                last_price=float(last["Close"]),
                previous_close=float(prev["Close"]),
                open_price=float(last["Open"]),
                high_price=float(last["High"]),
                low_price=float(last["Low"]),
                volume=int(last["Volume"]),
                currency="USD",
                exchange="US",
                timestamp=datetime.now().isoformat(),
                source="Stooq（延迟）",
            )
        except Exception as e:
            logger.error(f"[Stooq] {ticker} 报价失败: {e}")
            return None

    def _map_interval(self, interval: str) -> str:
        """映射 interval 到 Stooq 格式"""
        mapping = {
            "1d": "d",   # 日线
            "1w": "w",   # 周线
            "1m": "m",   # 月线
            "d": "d",
            "w": "w",
            "m": "m",
        }
        return mapping.get(interval, "d")

    # ---- 批量下载（高级功能）----

    def download_index_components(self, index: str = "SP500") -> dict[str, pd.DataFrame]:
        """
        下载指数成分股历史数据。
        注意：Stooq 提供 S&P 500 成分股打包下载。

        参数:
            index: "SP500" / "NASDAQ100" / "DJI"
        """
        # Stooq 提供打包 ZIP 下载
        # URL: https://stooq.com/db/h/{index}_us.zip
        # 但需要手动解压处理，这里仅作占位
        logger.warning("[Stooq] 批量下载功能待实现")
        return {}
