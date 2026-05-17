# ============================================================
# data/fetcher_fmp.py — Financial Modeling Prep 数据源
# 特点：免费 250次/天、数据质量高、财务数据丰富
# 文档：https://site.financialmodelingprep.com/developer/docs/
# ============================================================
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from .fetcher_base import BaseDataFetcher, Quote, normalize_ohlc

logger = logging.getLogger(__name__)


class FMPFetcher(BaseDataFetcher):
    """
    Financial Modeling Prep 数据获取器。

    免费套餐限制:
        - 250 次/天
        
    优点:
        - 数据质量高（官方数据源）
        - 财务数据丰富（财报、估值、评级）
        - 支持实时报价（较准确）
        - 文档清晰，API 稳定

    缺点:
        - 免费额度有限（但比 Alpha Vantage 宽松）
        - 部分高级功能需付费

    适用场景:
        - yfinance 失败时的主要备用源
        - 需要实时报价时
        - 财务数据分析
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        参数:
            api_key: FMP API Key
                - 可从环境变量 FMP_API_KEY 读取
                - 免费申请：https://site.financialmodelingprep.com/developer/docs/
        """
        self.api_key = api_key or os.getenv("FMP_API_KEY", "")
        if not self.api_key:
            logger.warning("[FMP] 未设置 API Key，请设置 FMP_API_KEY 环境变量")

        super().__init__(name="FMP", requests_per_day=250)
        self._session = requests.Session()

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
            period: 时间范围
            interval: "1d"/"1h"/"15min"/"5min"/"1min"
        """
        if not self.api_key:
            logger.warning("[FMP] 无 API Key，跳过")
            return pd.DataFrame()

        if self.is_rate_limited():
            logger.warning(f"[FMP] 已达每日限制 ({self.requests_per_day}/天)")
            return pd.DataFrame()

        endpoint, params = self._build_history_params(ticker, period, interval)

        try:
            resp = self._session.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and "Error Message" in data:
                logger.warning(f"[FMP] {ticker} 错误: {data['Error Message']}")
                self._record_failure(data["Error Message"])
                return pd.DataFrame()

            if not data:
                logger.warning(f"[FMP] {ticker} 无数据")
                return pd.DataFrame()

            df = self._parse_historical(data)
            df = normalize_ohlc(df, source="FMP")
            self._record_success()
            logger.info(f"[FMP] {ticker} 获取 {len(df)} 条记录")
            return df

        except Exception as e:
            self._record_failure(str(e))
            logger.error(f"[FMP] {ticker} 失败: {e}")
            return pd.DataFrame()

    def fetch_quote(self, ticker: str) -> Optional[Quote]:
        """获取实时报价（quote-short 接口）"""
        if not self.api_key:
            return None

        if self.is_rate_limited():
            return None

        try:
            resp = self._session.get(
                f"{self.BASE_URL}/quote-short/{ticker.upper()}",
                params={"apikey": self.api_key},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return None

            q = data[0] if isinstance(data, list) else data

            # 获取更多信息（previous close）
            resp2 = self._session.get(
                f"{self.BASE_URL}/quote/{ticker.upper()}",
                params={"apikey": self.api_key},
                timeout=30,
            )
            full_data = resp2.json()
            full_q = full_data[0] if isinstance(full_data, list) and full_data else {}

            return Quote(
                ticker=ticker,
                last_price=float(q.get("price", 0)),
                previous_close=float(full_q.get("previousClose", q.get("price", 0))),
                open_price=float(full_q.get("open", q.get("price", 0))),
                high_price=float(full_q.get("dayHigh", q.get("price", 0))),
                low_price=float(full_q.get("dayLow", q.get("price", 0))),
                volume=int(full_q.get("volume", 0)),
                market_cap=float(full_q.get("marketCap", 0)) if full_q.get("marketCap") else None,
                currency="USD",
                exchange="NASDAQ" if "nasdaq" in full_q.get("exchange", "").lower() else "US",
                timestamp=datetime.now().isoformat(),
                source="FMP",
            )
        except Exception as e:
            logger.error(f"[FMP] {ticker} 报价失败: {e}")
            return None

    def get_financials(self, ticker: str) -> dict:
        """获取财务数据（FMP 特有优势）"""
        if not self.api_key:
            return {}

        try:
            # Income Statement
            resp = self._session.get(
                f"{self.BASE_URL}/income-statement/{ticker}",
                params={"apikey": self.api_key, "limit": 4},
                timeout=30,
            )
            income = resp.json()

            # Balance Sheet
            resp = self._session.get(
                f"{self.BASE_URL}/balance-sheet-statement/{ticker}",
                params={"apikey": self.api_key, "limit": 4},
                timeout=30,
            )
            balance = resp.json()

            # Cash Flow
            resp = self._session.get(
                f"{self.BASE_URL}/cash-flow-statement/{ticker}",
                params={"apikey": self.api_key, "limit": 4},
                timeout=30,
            )
            cashflow = resp.json()

            return {
                "income_statement": income,
                "balance_sheet": balance,
                "cash_flow": cashflow,
            }
        except Exception as e:
            logger.error(f"[FMP] {ticker} 财务数据失败: {e}")
            return {}

    def _build_history_params(self, ticker: str, period: str, interval: str) -> tuple[str, dict]:
        """构建历史数据请求参数"""
        params = {"apikey": self.api_key}

        # 时间范围
        end_date = datetime.now()
        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825,
        }
        days = period_days.get(period, 365)
        start_date = end_date - timedelta(days=days)

        params["from"] = start_date.strftime("%Y-%m-%d")
        params["to"] = end_date.strftime("%Y-%m-%d")

        # 映射 interval
        if interval in ["1min", "5min", "15min", "30min", "1h"]:
            endpoint = f"/historical-chart/{interval}/{ticker.upper()}"
        elif interval in ["1d", "daily"]:
            endpoint = f"/historical-price-full/{ticker.upper()}"
            # 可选：serietype=line 获取最新价
        elif interval in ["1w", "weekly"]:
            # FMP 不直接支持周线，需要从日线聚合或使用历史数据
            endpoint = f"/historical-price-full/{ticker.upper()}"
        else:
            endpoint = f"/historical-price-full/{ticker.upper()}"

        return endpoint, params

    def _parse_historical(self, data) -> pd.DataFrame:
        """解析历史数据"""
        if isinstance(data, list):
            # historical-chart 返回列表
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # historical-price-full 返回 {"historical": [...]}
            if "historical" in data:
                df = pd.DataFrame(data["historical"])
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        if df.empty:
            return df

        # 日期处理
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

        # 列名标准化
        df = df.rename(columns={
            "open": "Open", "Open": "Open",
            "high": "High", "High": "High",
            "low": "Low", "Low": "Low",
            "close": "Close", "Close": "Close",
            "volume": "Volume", "Volume": "Volume",
        })

        # 类型转换
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
