# ============================================================
# data/fetcher_multi.py — 多数据源编排器
# 策略：主源失败 → 自动切换备用源 → 缓存兜底
# ============================================================
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .fetcher_base import BaseDataFetcher, Quote, normalize_ohlc
from .fetcher_stooq import StooqFetcher

logger = logging.getLogger(__name__)


class MultiSourceFetcher:
    """
    多数据源编排器。

    优先级链（自动降级）:
        1. yfinance（主源，免费，无限制）
        2. Stooq（备用，免费，无限制，日线+）
        3. Alpha Vantage（备用，25次/天，支持分钟级）
        4. FMP（备用，250次/天，财务数据丰富）

    特性:
        - 自动降级：主源失败自动切换下一个
        - 缓存：历史数据本地缓存（Parquet格式）
        - 请求优化：同类请求合并，减少API调用
        - 数据源状态追踪：每个源的成功/失败记录
        - 每日计数器自动重置
    """

    def __init__(
        self,
        cache_dir: str = "data_cache/",
        use_yfinance: bool = True,
        use_stooq: bool = True,
        use_alphavantage: bool = True,
        use_fmp: bool = True,
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._sources: list[BaseDataFetcher] = []
        self._source_map: dict[str, BaseDataFetcher] = {}

        # 按优先级初始化
        if use_yfinance:
            try:
                from .fetcher_yfinance import YFinanceFetcher
                yf = YFinanceFetcher(cache_dir=cache_dir)
                self._sources.append(yf)
                self._source_map["yfinance"] = yf
                logger.info("[MultiSource] yfinance 已加载")
            except Exception as e:
                logger.warning(f"[MultiSource] yfinance 加载失败: {e}")

        if use_stooq:
            try:
                stooq = StooqFetcher()
                self._sources.append(stooq)
                self._source_map["stooq"] = stooq
                logger.info("[MultiSource] Stooq 已加载")
            except Exception as e:
                logger.warning(f"[MultiSource] Stooq 加载失败: {e}")

        if use_alphavantage:
            try:
                from .fetcher_alphavantage import AlphaVantageFetcher
                av = AlphaVantageFetcher()
                if av.api_key:  # 有 API Key 才启用
                    self._sources.append(av)
                    self._source_map["alphavantage"] = av
                    logger.info("[MultiSource] Alpha Vantage 已加载")
                else:
                    logger.info("[MultiSource] Alpha Vantage 无 API Key，跳过")
            except Exception as e:
                logger.warning(f"[MultiSource] Alpha Vantage 加载失败: {e}")

        if use_fmp:
            try:
                from .fetcher_fmp import FMPFetcher
                fmp = FMPFetcher()
                if fmp.api_key:  # 有 API Key 才启用
                    self._sources.append(fmp)
                    self._source_map["fmp"] = fmp
                    logger.info("[MultiSource] FMP 已加载")
                else:
                    logger.info("[MultiSource] FMP 无 API Key，跳过")
            except Exception as e:
                logger.warning(f"[MultiSource] FMP 加载失败: {e}")

        if not self._sources:
            logger.error("[MultiSource] 无可用数据源！")

        logger.info(f"[MultiSource] 已加载 {len(self._sources)} 个数据源: {list(self._source_map.keys())}")

    # ============================================================
    # 兼容旧接口（YFinanceFetcher 方法签名）
    # ============================================================

    def download_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        auto_adjust: bool = True,
        progress: bool = False,
    ) -> pd.DataFrame:
        """
        下载历史K线（兼容旧接口）。
        自动尝试缓存 → 主源 → 备用源。
        """
        return self.fetch_history(ticker, period, interval)

    def get_quote(self, ticker: str) -> dict:
        """获取实时报价（兼容旧接口）"""
        quote = self.fetch_quote(ticker)
        if quote:
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
        return {}

    def get_quotes(self, tickers: list[str]) -> list[dict]:
        """批量实时报价（兼容旧接口）"""
        return [self.get_quote(t) for t in tickers]

    def get_financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        """获取财务数据"""
        # 优先用 FMP（财务数据最丰富）
        fmp = self._source_map.get("fmp")
        if fmp and hasattr(fmp, "get_financials"):
            result = fmp.get_financials(ticker)
            if result:
                return result

        # 回退到 yfinance
        yf_src = self._source_map.get("yfinance")
        if yf_src and hasattr(yf_src, "get_financials"):
            return yf_src.get_financials(ticker)

        return {}

    def get_info(self, ticker: str) -> dict:
        """获取标的摘要信息"""
        yf_src = self._source_map.get("yfinance")
        if yf_src and hasattr(yf_src, "get_info"):
            return yf_src.get_info(ticker)
        return {}

    def get_analyst_info(self, ticker: str) -> dict:
        """获取分析师评级"""
        yf_src = self._source_map.get("yfinance")
        if yf_src and hasattr(yf_src, "get_analyst_info"):
            return yf_src.get_analyst_info(ticker)
        return {}

    def get_news(self, ticker: str, max_news: int = 10) -> list[dict]:
        """获取新闻"""
        yf_src = self._source_map.get("yfinance")
        if yf_src and hasattr(yf_src, "get_news"):
            return yf_src.get_news(ticker, max_news)
        return []

    # ============================================================
    # 核心方法：带降级的数据获取
    # ============================================================

    def fetch_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        带自动降级的历史数据获取。

        流程:
            1. 检查缓存（历史数据不变，直接复用）
            2. yfinance（主源）
            3. Stooq（备用，仅日线+）
            4. Alpha Vantage（备用，支持分钟级）
            5. FMP（备用）
            6. 全部失败 → 返回空 DataFrame
        """
        # 1. 缓存检查
        cached = self._load_cache(ticker, period, interval)
        if cached is not None:
            logger.info(f"[MultiSource] {ticker} 命中缓存")
            return cached

        # 2. 逐源尝试
        for source in self._sources:
            if source.is_rate_limited():
                logger.info(f"[MultiSource] {source.name} 已达限制，跳过")
                continue

            # Stooq 不支持分钟级数据
            if isinstance(source, StooqFetcher) and interval not in ["1d", "1w", "1m", "d", "w", "m"]:
                continue

            try:
                df = source.fetch_history(ticker, period, interval)
                if not df.empty and len(df) >= 3:
                    logger.info(f"[MultiSource] {ticker} 从 {source.name} 获取成功 ({len(df)}条)")
                    # 缓存成功数据
                    self._save_cache(ticker, period, interval, df)
                    return df
            except Exception as e:
                logger.warning(f"[MultiSource] {source.name} 异常: {e}")
                continue

        logger.error(f"[MultiSource] {ticker} 所有数据源均失败")
        return pd.DataFrame()

    def fetch_quote(self, ticker: str) -> Optional[Quote]:
        """带自动降级的实时报价获取"""
        for source in self._sources:
            if source.is_rate_limited():
                continue

            try:
                quote = source.fetch_quote(ticker)
                if quote and quote.last_price > 0:
                    logger.info(f"[MultiSource] {ticker} 报价来自 {source.name}")
                    return quote
            except Exception:
                continue

        logger.warning(f"[MultiSource] {ticker} 报价获取失败（所有源）")
        return None

    # ============================================================
    # 批量获取
    # ============================================================

    def fetch_history_batch(
        self,
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """批量获取历史数据"""
        results = {}
        for ticker in tickers:
            results[ticker] = self.fetch_history(ticker, period, interval)
        return results

    # ============================================================
    # 缓存管理
    # ============================================================

    def _cache_key(self, ticker: str, period: str, interval: str) -> str:
        """生成缓存文件名"""
        return f"{ticker}_{interval}_{period}.parquet"

    def _cache_path(self, ticker: str, period: str, interval: str) -> str:
        """返回缓存完整路径"""
        return os.path.join(self.cache_dir, self._cache_key(ticker, period, interval))

    def _load_cache(
        self,
        ticker: str,
        period: str,
        interval: str,
        max_age_hours: int = 24,
    ) -> Optional[pd.DataFrame]:
        """
        加载缓存。

        缓存策略:
            - 历史数据（日线+）: 24小时有效（历史不会变）
            - 分钟级数据: 不缓存（实时性强）
            - 当日数据: 缓存到收盘后
        """
        if interval in ["1min", "5min", "15min", "30min", "1h", "60min"]:
            return None  # 分钟级不缓存

        path = self._cache_path(ticker, period, interval)
        if not os.path.exists(path):
            return None

        # 检查缓存时效
        mtime = os.path.getmtime(path)
        age_hours = (time.time() - mtime) / 3600

        # 历史数据缓存24小时
        if age_hours > max_age_hours:
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None
            return df
        except Exception as e:
            logger.warning(f"[MultiSource] 缓存读取失败 {path}: {e}")
            return None

    def _save_cache(
        self,
        ticker: str,
        period: str,
        interval: str,
        df: pd.DataFrame,
    ) -> bool:
        """保存缓存"""
        if interval in ["1min", "5min", "15min", "30min", "1h", "60min"]:
            return False  # 分钟级不缓存

        path = self._cache_path(ticker, period, interval)
        try:
            df.to_parquet(path, engine="pyarrow")
            logger.info(f"[MultiSource] 缓存已保存: {path}")
            return True
        except Exception as e:
            logger.warning(f"[MultiSource] 缓存保存失败: {e}")
            return False

    def clear_cache(self, ticker: Optional[str] = None):
        """清理缓存"""
        if ticker:
            # 清理特定标的
            for f in os.listdir(self.cache_dir):
                if f.startswith(ticker.upper()) and f.endswith(".parquet"):
                    os.remove(os.path.join(self.cache_dir, f))
                    logger.info(f"[MultiSource] 删除缓存: {f}")
        else:
            # 清理所有
            for f in os.listdir(self.cache_dir):
                if f.endswith(".parquet"):
                    os.remove(os.path.join(self.cache_dir, f))
            logger.info("[MultiSource] 全部缓存已清理")

    # ============================================================
    # 状态监控
    # ============================================================

    def get_sources_status(self) -> dict[str, dict]:
        """获取所有数据源状态"""
        return {
            name: {
                "available": src._is_available,
                "requests_today": src._requests_today,
                "requests_limit": src.requests_per_day,
                "last_success": src._last_success.isoformat() if src._last_success else None,
                "last_failure": src._last_failure.isoformat() if src._last_failure else None,
                "failure_reason": src._failure_reason,
            }
            for name, src in self._source_map.items()
        }

    def reset_daily_counters(self):
        """重置所有源每日计数器（供定时任务调用）"""
        for src in self._sources:
            src.reset_daily_counter()

    # ============================================================
    # 独立测试
    # ============================================================

    def test_sources(self, ticker: str = "AAPL") -> dict:
        """测试所有数据源连通性"""
        results = {}

        for name, src in self._source_map.items():
            try:
                # 测试历史数据
                df = src.fetch_history(ticker, period="1mo", interval="1d")
                hist_ok = not df.empty

                # 测试报价
                quote = src.fetch_quote(ticker)
                quote_ok = quote is not None and quote.last_price > 0

                results[name] = {
                    "history": "✅" if hist_ok else "❌",
                    "quote": "✅" if quote_ok else "❌",
                    "records": len(df) if hist_ok else 0,
                    "price": quote.last_price if quote_ok else 0,
                }
            except Exception as e:
                results[name] = {
                    "history": "❌",
                    "quote": "❌",
                    "error": str(e),
                }

        return results


# ============================================================
# 独立测试
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fetcher = MultiSourceFetcher(cache_dir="data_cache/")
    print("数据源状态:", fetcher.get_sources_status())

    # 测试连通性
    print("\n--- 连通性测试 ---")
    test_results = fetcher.test_sources("AAPL")
    for name, result in test_results.items():
        print(f"  {name}: {result}")

    # 测试历史数据
    print("\n--- 历史数据测试 ---")
    df = fetcher.fetch_history("SPY", period="3mo", interval="1d")
    if not df.empty:
        print(f"  SPY: {len(df)} 条, 最新 {df.index[-1].date()} Close={df['Close'].iloc[-1]:.2f}")
    else:
        print("  SPY: 获取失败")

    # 测试报价
    print("\n--- 报价测试 ---")
    quote = fetcher.fetch_quote("SPY")
    if quote:
        print(f"  SPY: {quote.last_price:.2f} (来自 {quote.source})")
    else:
        print("  SPY: 报价失败")
