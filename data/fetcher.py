# ============================================================
# data/fetcher.py — 市场数据获取模块
# 多数据源架构：yfinance(主) → Stooq(备) → AlphaVantage/FMP(备)
# ============================================================
#
# 使用方式：
#   from data.fetcher import YFinanceFetcher, MultiSourceFetcher
#   fetcher = MultiSourceFetcher()  # 推荐：自动降级
#   # 或
#   fetcher = YFinanceFetcher()     # 仅用 yfinance
#
# 数据源优先级（MultiSourceFetcher）：
#   1. yfinance（主源，免费无限制）
#   2. Stooq（备用，免费无限制，仅日线+）
#   3. Alpha Vantage（备用，25次/天，支持分钟级）
#   4. FMP（备用，250次/天，财务数据丰富）
#
# ============================================================

# 导出主要类（保持向后兼容）
from .fetcher_base import BaseDataFetcher, Quote, DataSourceStatus
from .fetcher_yfinance import YFinanceFetcher
from .fetcher_stooq import StooqFetcher
from .fetcher_multi import MultiSourceFetcher

# 兼容旧代码：YFinanceFetcher 直接导出
# 但推荐使用 MultiSourceFetcher 获得自动降级能力

__all__ = [
    "YFinanceFetcher",
    "MultiSourceFetcher",
    "StooqFetcher",
    "BaseDataFetcher",
    "Quote",
    "DataSourceStatus",
]


# ============================================================
# 旧版 YFinanceFetcher 代码（保留但标记废弃）
# 新代码应使用 fetcher_yfinance.py 中的实现
# ============================================================

# 下方的旧实现已迁移到 fetcher_yfinance.py
# 保留此类定义仅用于向后兼容检查

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_period(period: str) -> tuple[str, str]:
    """把 period 转为 start/end 日期字符串"""
    mapping = {
        "1mo": ("1mo", "now"),
        "3mo": ("3mo", "now"),
        "6mo": ("6mo", "now"),
        "1y": ("1y", "now"),
        "2y": ("2y", "now"),
        "5y": ("5y", "now"),
        "10y": ("10y", "now"),
    }
    return mapping.get(period, ("2y", "now"))


# 旧版类定义（向后兼容，实际使用上方导入的新实现）
class _YFinanceFetcherLegacy:
    """
    通过 yfinance 拉取美股历史行情、财务数据、实时报价。
    全部免费，无需 API Key。
    """

    def __init__(self, cache_dir: str = "data_cache/"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ---- K线数据 ----

    def download_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        auto_adjust: bool = True,
        progress: bool = False,
    ) -> pd.DataFrame:
        """
        下载历史K线数据。

        参数:
            ticker: 股票代码，如 "AAPL"、"MSFT"
            period: 时间范围，如 "1y"、"2y"
            interval: K线周期，如 "1d"、"1h"、"5m"
            auto_adjust: 是否自动复权（推荐 True）
            progress: 是否显示下载进度条

        返回:
            DataFrame，列: Open/High/Low/Close/Volume（复权后）
        """
        logger.info(f"[YFinance] 下载 {ticker} {interval} {period}")
        try:
            stock = yf.Ticker(ticker)
            # yfinance >= 1.0 移除了 progress 参数，改用 actions=False 过滤分红
            df = stock.history(
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=False,
            )
            if df.empty:
                logger.warning(f"[YFinance] {ticker} 无数据")
                return df
            df.index = df.index.tz_localize(None)  # 去掉时区，方便后续处理
            logger.info(f"[YFinance] {ticker} 获取 {len(df)} 条记录 {df.index[0].date()} ~ {df.index[-1].date()}")
            return df
        except Exception as e:
            logger.error(f"[YFinance] {ticker} 下载失败: {e}")
            return pd.DataFrame()

    def download_multi(
        self,
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """批量下载多个标的"""
        logger.info(f"[YFinance] 批量下载 {tickers}")
        result = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by="ticker",
        )
        # 转为 dict[ticker, DataFrame]
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

    # ---- 实时报价 ----

    def get_quote(self, ticker: str) -> dict:
        """获取单个标的实时报价"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            return {
                "ticker": ticker,
                "last_price": info.last_price or 0,
                "previous_close": info.previous_close or 0,
                "market_cap": info.market_cap or 0,
                "currency": info.currency or "USD",
                "exchange": info.exchange or "US",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"[YFinance] {ticker} 实时报价失败: {e}")
            return {}

    def get_quotes(self, tickers: list[str]) -> list[dict]:
        """批量实时报价"""
        return [self.get_quote(t) for t in tickers]

    # ---- 财务数据 ----

    def get_financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        """获取财务数据：income_stmt / balance_sheet / cash_flow"""
        try:
            stock = yf.Ticker(ticker)
            return {
                "income_stmt": stock.income_stmt,
                "balance_sheet": stock.balance_sheet,
                "cash_flow": stock.cashflow,
            }
        except Exception as e:
            logger.error(f"[YFinance] {ticker} 财务数据失败: {e}")
            return {}

    def get_info(self, ticker: str) -> dict:
        """获取标的摘要信息（PE、EPS、派息等）"""
        try:
            return yf.Ticker(ticker).info
        except Exception as e:
            logger.error(f"[YFinance] {ticker} info 失败: {e}")
            return {}

    # ---- 分析师预期 ----

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
            logger.error(f"[YFinance] {ticker} 分析师数据失败: {e}")
            return {}

    # ---- 新闻情绪 ----

    def get_news(self, ticker: str, max_news: int = 10) -> list[dict]:
        """获取近期新闻标题（用于情绪分析）"""
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
            logger.error(f"[YFinance] {ticker} 新闻失败: {e}")
            return []

    # ---- 分红/拆股 ----

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


# ============================================================
# 主入口（独立测试）
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetcher = YFinanceFetcher(cache_dir="data_cache/")

    # 测试下载 AAPL 历史
    df = fetcher.download_history("AAPL", period="3mo", interval="1d")
    print(df.tail())

    # 测试实时报价
    quote = fetcher.get_quote("AAPL")
    print("报价:", quote)

    # 测试批量
    quotes = fetcher.get_quotes(["AAPL", "TSLA", "NVDA"])
    for q in quotes:
        print(q)
