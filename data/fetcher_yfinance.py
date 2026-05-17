# ============================================================
# data/fetcher_yfinance.py — yfinance 数据源
# 特点：免费、无限制、数据全面、无 API Key
# 这是主数据源，优先级最高
# ============================================================
import logging
import os
from datetime import datetime
from typing import Optional

import yfinance as yf
import pandas as pd

from .fetcher_base import BaseDataFetcher, Quote, normalize_ohlc

logger = logging.getLogger(__name__)


class YFinanceFetcher(BaseDataFetcher):
    """
    yfinance 数据获取器（主数据源）。

    优点:
        - 完全免费，无请求限制
        - 无需 API Key
        - 数据全面（历史、财务、新闻、分析师）
        - 支持多时间框架（含分钟级）

    缺点:
        - 无官方支持，可能随时变动
        - 实时数据有 ~15分钟延迟
        - 部分小盘股数据可能缺失

    适用场景:
        - 主数据源，所有请求优先走这里
    """

    def __init__(self, cache_dir: str = "data_cache/"):
        super().__init__(name="yfinance", requests_per_day=0)  # 无限制
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ---- K线数据 ----

    def fetch_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        下载历史K线数据。

        参数:
            ticker: 股票代码，如 "AAPL"、"MSFT"
            period: 时间范围，如 "1y"、"2y"、"1mo"
            interval: K线周期，如 "1d"、"1h"、"5m"
        """
        logger.info(f"[yfinance] 下载 {ticker} {interval} {period}")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                actions=False,
            )
            if df.empty:
                logger.warning(f"[yfinance] {ticker} 无数据")
                return df

            df.index = df.index.tz_localize(None)  # 去时区
            df = normalize_ohlc(df, source="yfinance")
            self._record_success()
            logger.info(f"[yfinance] {ticker} 获取 {len(df)} 条记录 {df.index[0].date()} ~ {df.index[-1].date()}")
            return df

        except Exception as e:
            self._record_failure(str(e))
            logger.error(f"[yfinance] {ticker} 下载失败: {e}")
            return pd.DataFrame()

    # ---- 实时报价 ----

    def fetch_quote(self, ticker: str) -> Optional[Quote]:
        """获取实时报价（~15分钟延迟）"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info

            last_price = info.last_price or 0
            prev_close = info.previous_close or 0

            # 尝试获取当日开高低
            df = stock.history(period="1d", interval="1m")
            if not df.empty:
                open_price = df["Open"].iloc[0]
                high_price = df["High"].max()
                low_price = df["Low"].min()
                volume = int(df["Volume"].sum())
            else:
                open_price = last_price
                high_price = last_price
                low_price = last_price
                volume = 0

            return Quote(
                ticker=ticker,
                last_price=float(last_price),
                previous_close=float(prev_close),
                open_price=float(open_price),
                high_price=float(high_price),
                low_price=float(low_price),
                volume=volume,
                market_cap=info.market_cap if hasattr(info, "market_cap") else None,
                currency=info.currency or "USD",
                exchange=info.exchange or "US",
                timestamp=datetime.now().isoformat(),
                source="yfinance（~15min延迟）",
            )
        except Exception as e:
            self._record_failure(str(e))
            logger.error(f"[yfinance] {ticker} 实时报价失败: {e}")
            return None

    # ---- 兼容旧接口（保留）----

    def download_multi(
        self,
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """批量下载多个标的"""
        logger.info(f"[yfinance] 批量下载 {tickers}")
        result = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by="ticker",
            progress=False,
        )
        out = {}
        for t in tickers:
            if len(tickers) == 1:
                out[t] = result.copy()
            else:
                try:
                    out[t] = result[t].dropna(how="all")
                except KeyError:
                    out[t] = pd.DataFrame()
        return out

    def get_financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        """获取财务数据"""
        try:
            stock = yf.Ticker(ticker)
            return {
                "income_stmt": stock.income_stmt,
                "balance_sheet": stock.balance_sheet,
                "cash_flow": stock.cashflow,
            }
        except Exception as e:
            logger.error(f"[yfinance] {ticker} 财务数据失败: {e}")
            return {}

    def get_info(self, ticker: str) -> dict:
        """获取标的摘要信息"""
        try:
            return yf.Ticker(ticker).info
        except Exception as e:
            logger.error(f"[yfinance] {ticker} info 失败: {e}")
            return {}

    def get_analyst_info(self, ticker: str) -> dict:
        """获取分析师评级、目标价"""
        try:
            stock = yf.Ticker(ticker)
            return {
                "recommendations": stock.recommendations,
                "earnings_dates": stock.earnings_dates,
                "analyst_price_targets": stock.analyst_price_targets,
            }
        except Exception as e:
            logger.error(f"[yfinance] {ticker} 分析师数据失败: {e}")
            return {}

    def get_news(self, ticker: str, max_news: int = 10) -> list[dict]:
        """获取近期新闻标题"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            return [
                {
                    "title": n.get("title", ""),
                    "publisher": n.get("publisher", ""),
                    "link": n.get("link", ""),
                    "published": n.get("published", ""),
                }
                for n in (news or [])[:max_news]
            ]
        except Exception as e:
            logger.error(f"[yfinance] {ticker} 新闻失败: {e}")
            return []

    def get_splits(self, ticker: str) -> pd.DataFrame:
        """获取拆股历史"""
        try:
            return yf.Ticker(ticker).splits
        except Exception:
            return pd.DataFrame()

    def get_dividends(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """获取分红历史"""
        try:
            stock = yf.Ticker(ticker)
            return stock.dividends
        except Exception:
            return pd.DataFrame()

    # ---- 兼容旧接口 ----

    def download_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        auto_adjust: bool = True,
        progress: bool = False,
    ) -> pd.DataFrame:
        """
        兼容旧接口：下载历史K线数据。
        实际调用 fetch_history()。
        """
        return self.fetch_history(ticker, period=period, interval=interval)

    def get_quote(self, ticker: str) -> dict:
        """获取单个标的实时报价（兼容旧接口，返回 dict）"""
        quote = self.fetch_quote(ticker)
        if quote is None:
            return {}
        return {
            "ticker": quote.ticker,
            "last_price": quote.last_price,
            "previous_close": quote.previous_close,
            "market_cap": quote.market_cap,
            "currency": quote.currency,
            "exchange": quote.exchange,
            "timestamp": quote.timestamp,
            "source": quote.source,
        }
