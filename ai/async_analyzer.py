# ============================================================
# ai/async_analyzer.py — 异步并行分析 + 缓存 + 懒加载
# ============================================================
import asyncio
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 1. 缓存系统
# ============================================================

class AnalysisCache:
    """
    基于文件系统的分析结果缓存。
    缓存键: (ticker, analysis_type, date)
    TTL: 盘中数据15分钟，日终数据当天有效
    """
    
    def __init__(self, cache_dir: str = None, ttl_intraday: int = 900, ttl_eod: int = 86400):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache" / "analysis"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_intraday = ttl_intraday  # 15分钟
        self.ttl_eod = ttl_eod            # 24小时
        self._memory_cache = {}           # L1: 内存缓存
        self._memory_max_size = 50        # 最多50个结果
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.pkl"
    
    def _make_key(self, ticker: str, analysis_type: str, date_str: str = None) -> str:
        """生成缓存键"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        return f"{ticker}_{analysis_type}_{date_str}"
    
    def get(self, ticker: str, analysis_type: str, date_str: str = None) -> Optional[dict]:
        """
        获取缓存结果。
        返回None表示缓存未命中或已过期。
        """
        key = self._make_key(ticker, analysis_type, date_str)
        
        # L1: 内存缓存
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() - entry["timestamp"] < self.ttl_intraday:
                logger.debug(f"[Cache] L1 hit: {key}")
                return entry["data"]
            else:
                del self._memory_cache[key]
        
        # L2: 文件缓存
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        try:
            mtime = cache_path.stat().st_mtime
            age = time.time() - mtime
            
            # 判断数据类型决定TTL
            is_eod = analysis_type in ["backtest", "monte_carlo", "portfolio"]
            ttl = self.ttl_eod if is_eod else self.ttl_intraday
            
            if age > ttl:
                logger.debug(f"[Cache] L2 expired: {key} (age={age:.0f}s)")
                cache_path.unlink(missing_ok=True)
                return None
            
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            
            # 回填L1
            self._set_memory(key, data)
            logger.debug(f"[Cache] L2 hit: {key}")
            return data
            
        except Exception as e:
            logger.warning(f"[Cache] L2 read error: {e}")
            return None
    
    def _set_memory(self, key: str, data: dict):
        """设置内存缓存（LRU）"""
        if len(self._memory_cache) >= self._memory_max_size:
            # 移除最旧的
            oldest = min(self._memory_cache, key=lambda k: self._memory_cache[k]["timestamp"])
            del self._memory_cache[oldest]
        
        self._memory_cache[key] = {
            "timestamp": time.time(),
            "data": data,
        }
    
    def set(self, ticker: str, analysis_type: str, data: dict, date_str: str = None):
        """保存缓存结果"""
        key = self._make_key(ticker, analysis_type, date_str)
        
        # L1
        self._set_memory(key, data)
        
        # L2
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"[Cache] Saved: {key}")
        except Exception as e:
            logger.warning(f"[Cache] L2 write error: {e}")
    
    def invalidate(self, ticker: str = None, analysis_type: str = None):
        """使缓存失效"""
        pattern = ""
        if ticker:
            pattern += f"{ticker}_"
        if analysis_type:
            pattern += f"{analysis_type}_"
        
        # L1
        keys_to_remove = [k for k in self._memory_cache if k.startswith(pattern)]
        for k in keys_to_remove:
            del self._memory_cache[k]
        
        # L2
        for f in self.cache_dir.glob(f"{pattern}*.pkl"):
            f.unlink(missing_ok=True)
        
        logger.info(f"[Cache] Invalidated: {pattern}*")
    
    def clear_all(self):
        """清空所有缓存"""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)
        logger.info("[Cache] All cleared")


# 全局缓存实例
_global_cache = None

def get_cache() -> AnalysisCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = AnalysisCache()
    return _global_cache


# ============================================================
# 2. 懒加载装饰器
# ============================================================

def lazy_load(cache_key_func: Callable = None, ttl: int = 900):
    """
    懒加载装饰器：先检查缓存，未命中才执行函数。
    
    用法:
        @lazy_load(cache_key_func=lambda ticker, **kw: f"mtf_{ticker}")
        def analyze_multi_timeframe(ticker, fetcher):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            if cache_key_func:
                try:
                    key = cache_key_func(*args, **kwargs)
                    ticker = args[0] if args else kwargs.get("ticker", "unknown")
                    cached = cache.get(ticker, key)
                    if cached is not None:
                        return cached
                except Exception:
                    pass
            
            result = func(*args, **kwargs)
            
            if cache_key_func and result is not None:
                try:
                    key = cache_key_func(*args, **kwargs)
                    ticker = args[0] if args else kwargs.get("ticker", "unknown")
                    cache.set(ticker, key, result)
                except Exception:
                    pass
            
            return result
        return wrapper
    return decorator


# ============================================================
# 3. 异步并行分析
# ============================================================

class AsyncAnalyzer:
    """
    异步并行分析器：在ThreadPoolExecutor中并行执行多个分析任务。
    适用于IO密集型操作（yfinance数据获取）。
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = get_cache()
    
    def run_parallel(self, tasks: list[Callable]) -> list:
        """
        并行执行多个分析任务。
        
        参数:
            tasks: 任务函数列表，每个函数无参数（使用闭包捕获参数）
        
        返回:
            结果列表（与tasks顺序一致）
        """
        from concurrent.futures import as_completed
        
        futures = []
        for task in tasks:
            future = self.executor.submit(task)
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                results.append(future.result(timeout=60))
            except Exception as e:
                logger.warning(f"[Async] Task failed: {e}")
                results.append(None)
        
        return results
    
    def run_full_analysis_parallel(
        self,
        ticker: str,
        fetcher,
        modules: list[str] = None,
    ) -> dict:
        """
        对单个标的并行执行多个分析模块。
        
        参数:
            ticker: 标的代码
            fetcher: YFinanceFetcher实例
            modules: 要执行的模块列表，默认全部
        
        返回:
            {"module_name": result, ...}
        """
        if modules is None:
            modules = [
                "multi_timeframe", "support_resistance", "pattern_recognition",
                "derivatives", "volume_profile", "classical_theory",
                "sentiment", "institutional_flows", "macro_policy",
                "institutional_detector",
            ]
        
        results = {"ticker": ticker, "modules": {}}
        
        # 定义任务
        def make_task(module_name):
            def task():
                try:
                    if module_name == "multi_timeframe":
                        from ai.multi_timeframe import analyze_multi_timeframe
                        return analyze_multi_timeframe(ticker, 0, fetcher)
                    elif module_name == "support_resistance":
                        from ai.support_resistance import analyze_support_resistance
                        return analyze_support_resistance(ticker, 0, fetcher)
                    elif module_name == "pattern_recognition":
                        from ai.pattern_recognition import analyze_patterns
                        return analyze_patterns(ticker, fetcher)
                    elif module_name == "derivatives":
                        from ai.derivatives import analyze_derivatives
                        return analyze_derivatives(ticker, fetcher)
                    elif module_name == "volume_profile":
                        from ai.volume_profile import analyze_volume_profile
                        return analyze_volume_profile(ticker, 0, fetcher)
                    elif module_name == "classical_theory":
                        from ai.classical_theory import analyze_classical_theory
                        return analyze_classical_theory(ticker, 0, fetcher)
                    elif module_name == "sentiment":
                        from ai.sentiment import analyze_market_sentiment
                        return analyze_market_sentiment(ticker, fetcher)
                    elif module_name == "institutional_flows":
                        from ai.institutional_flows import analyze_institutional_flows
                        return analyze_institutional_flows(ticker, 0, fetcher)
                    elif module_name == "macro_policy":
                        from ai.macro_policy import analyze_macro_policy
                        return analyze_macro_policy(ticker, 0, fetcher)
                    elif module_name == "institutional_detector":
                        from ai.institutional_detector import analyze_institutional_activity
                        return analyze_institutional_activity(ticker, 0, fetcher)
                    else:
                        return None
                except Exception as e:
                    logger.warning(f"[Async] {module_name} failed: {e}")
                    return None
            return task
        
        tasks = [make_task(m) for m in modules]
        parallel_results = self.run_parallel(tasks)
        
        for module_name, result in zip(modules, parallel_results):
            if result is not None:
                results["modules"][module_name] = result
                # 缓存结果
                self.cache.set(ticker, module_name, {"result": result})
        
        return results
    
    def run_multi_ticker_analysis(
        self,
        tickers: list[str],
        fetcher,
        modules: list[str] = None,
    ) -> dict:
        """
        对多个标的并行执行分析。
        
        返回:
            {ticker: {"modules": {...}}, ...}
        """
        def make_ticker_task(ticker):
            def task():
                return self.run_full_analysis_parallel(ticker, fetcher, modules)
            return task
        
        tasks = [make_ticker_task(t) for t in tickers]
        results_list = self.run_parallel(tasks)
        
        return {ticker: result for ticker, result in zip(tickers, results_list) if result}
    
    def close(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)


# ============================================================
# 4. 性能监控
# ============================================================

class PerformanceMonitor:
    """简单性能监控"""
    
    def __init__(self):
        self.timings = {}
    
    def timeit(self, name: str):
        """上下文管理器计时"""
        class TimerContext:
            def __init__(ctx, monitor, name):
                ctx.monitor = monitor
                ctx.name = name
                ctx.start = None
            
            def __enter__(ctx):
                ctx.start = time.time()
                return ctx
            
            def __exit__(ctx, *args):
                elapsed = time.time() - ctx.start
                ctx.monitor.timings[ctx.name] = elapsed
                logger.info(f"[Perf] {ctx.name}: {elapsed:.2f}s")
        
        return TimerContext(self, name)
    
    def get_report(self) -> dict:
        """获取性能报告"""
        total = sum(self.timings.values())
        return {
            "total_time": round(total, 2),
            "breakdown": {k: round(v, 2) for k, v in self.timings.items()},
            "slowest": max(self.timings, key=self.timings.get) if self.timings else None,
        }


# ============================================================
# 5. 便捷函数
# ============================================================

def run_cached_analysis(
    ticker: str,
    analysis_type: str,
    fetcher,
    force_refresh: bool = False,
) -> Optional[dict]:
    """
    带缓存的分析执行入口。
    
    参数:
        ticker: 标的代码
        analysis_type: 分析类型 (e.g. "multi_timeframe", "sentiment")
        fetcher: YFinanceFetcher
        force_refresh: 强制刷新缓存
    
    返回:
        分析结果字典或None
    """
    cache = get_cache()
    
    if not force_refresh:
        cached = cache.get(ticker, analysis_type)
        if cached is not None:
            return cached.get("result")
    
    # 执行分析
    try:
        if analysis_type == "multi_timeframe":
            from ai.multi_timeframe import analyze_multi_timeframe, format_mtf_result
            r = analyze_multi_timeframe(ticker, 0, fetcher)
            result = format_mtf_result(r)
        elif analysis_type == "support_resistance":
            from ai.support_resistance import analyze_support_resistance, format_sr_result
            r = analyze_support_resistance(ticker, 0, fetcher)
            result = format_sr_result(r)
        elif analysis_type == "sentiment":
            from ai.sentiment import analyze_market_sentiment, format_sentiment_result
            r = analyze_market_sentiment(ticker, fetcher)
            result = format_sentiment_result(r)
        elif analysis_type == "derivatives":
            from ai.derivatives import analyze_derivatives, format_derivatives_result
            r = analyze_derivatives(ticker, fetcher)
            result = format_derivatives_result(r)
        elif analysis_type == "volume_profile":
            from ai.volume_profile import analyze_volume_profile, format_vp_result
            r = analyze_volume_profile(ticker, 0, fetcher)
            result = format_vp_result(r)
        elif analysis_type == "classical_theory":
            from ai.classical_theory import analyze_classical_theory, format_classical_result
            r = analyze_classical_theory(ticker, 0, fetcher)
            result = format_classical_result(r)
        elif analysis_type == "pattern_recognition":
            from ai.pattern_recognition import analyze_patterns
            result = analyze_patterns(ticker, fetcher)
        elif analysis_type == "institutional_detector":
            from ai.institutional_detector import analyze_institutional_activity, format_institutional_result
            r = analyze_institutional_activity(ticker, 0, fetcher)
            result = format_institutional_result(r)
        elif analysis_type == "institutional_flows":
            from ai.institutional_flows import analyze_institutional_flows, format_flows_result
            r = analyze_institutional_flows(ticker, 0, fetcher)
            result = format_flows_result(r)
        elif analysis_type == "macro_policy":
            from ai.macro_policy import analyze_macro_policy, format_macro_policy_result
            r = analyze_macro_policy(ticker, 0, fetcher)
            result = format_macro_policy_result(r)
        else:
            logger.warning(f"[CachedAnalysis] Unknown type: {analysis_type}")
            return None
        
        cache.set(ticker, analysis_type, {"result": result})
        return result
        
    except Exception as e:
        logger.warning(f"[CachedAnalysis] {analysis_type} failed: {e}")
        return None


def batch_run_analysis(
    tickers: list[str],
    analysis_types: list[str],
    fetcher,
    max_workers: int = 4,
) -> dict:
    """
    批量并行分析多个标的的多个模块。
    
    返回:
        {ticker: {analysis_type: result, ...}, ...}
    """
    analyzer = AsyncAnalyzer(max_workers=max_workers)
    
    # 构建所有任务
    all_tasks = []
    task_index = []
    
    for ticker in tickers:
        for atype in analysis_types:
            def make_task(t, a):
                return lambda: (t, a, run_cached_analysis(t, a, fetcher))
            all_tasks.append(make_task(ticker, atype))
            task_index.append((ticker, atype))
    
    # 并行执行
    results = analyzer.run_parallel(all_tasks)
    
    # 组织结果
    output = {t: {} for t in tickers}
    for (ticker, atype), result in zip(task_index, results):
        if result and result[2] is not None:
            output[ticker][atype] = result[2]
    
    analyzer.close()
    return output
