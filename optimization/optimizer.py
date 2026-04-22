# ============================================================
# optimization/optimizer.py — 策略参数优化模块
# 支持：网格搜索 / 随机搜索 / 贝叶斯优化
# ============================================================
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import itertools
import random

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 数据类
# ============================================================

@dataclass
class OptimizationResult:
    """优化结果"""
    strategy_name: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    optimization_method: str
    n_trials: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# ============================================================
# 策略参数优化器
# ============================================================

class StrategyOptimizer:
    """
    策略参数优化器
    
    支持多种优化方法：
    - grid: 网格搜索（穷举）
    - random: 随机搜索
    - bayesian: 贝叶斯优化（需 optuna）
    """
    
    def __init__(
        self,
        data_fetcher=None,
        backtester=None,
        scoring: str = "sharpe"
    ):
        """
        Args:
            data_fetcher: 数据获取器
            backtester: 回测器
            scoring: 优化目标（sharpe/return/max_drawdown）
        """
        self.fetcher = data_fetcher
        self.backtester = backtester
        self.scoring = scoring
        
        # 参数空间定义
        self.param_spaces = {
            "MA": {
                "short_window": [5, 10, 15, 20],
                "long_window": [30, 50, 100, 200]
            },
            "RSI": {
                "period": [7, 14, 21],
                "oversold": [20, 25, 30, 35],
                "overbought": [65, 70, 75, 80]
            },
            "MACD": {
                "fast_period": [8, 10, 12, 14],
                "slow_period": [20, 24, 26, 30],
                "signal_period": [7, 9, 11]
            },
            "Bollinger": {
                "period": [15, 20, 25],
                "std_dev": [1.5, 2.0, 2.5, 3.0]
            },
            "Composite": {
                "rsi_period": [10, 14, 21],
                "rsi_oversold": [25, 30, 35, 40],
                "rsi_overbought": [60, 65, 70, 75],
                "atr_period": [10, 14, 20],
                "atr_stop_multiplier": [1.5, 2.0, 2.5, 3.0]
            }
        }
    
    def optimize(
        self,
        strategy_name: str,
        ticker: str,
        method: str = "grid",
        n_trials: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        custom_param_space: Optional[Dict] = None
    ) -> OptimizationResult:
        """
        优化策略参数
        
        Args:
            strategy_name: 策略名称
            ticker: 股票代码
            method: 优化方法（grid/random/bayesian）
            n_trials: 试验次数（随机/贝叶斯）
            start_date: 回测开始日期
            end_date: 回测结束日期
            custom_param_space: 自定义参数空间
        
        Returns:
            OptimizationResult 实例
        """
        logger.info(f"开始优化 {strategy_name} 参数，方法: {method}")
        
        # 获取参数空间
        param_space = custom_param_space or self.param_spaces.get(strategy_name, {})
        if not param_space:
            raise ValueError(f"未定义策略 {strategy_name} 的参数空间")
        
        # 获取数据
        if self.fetcher is None:
            from data.fetcher import YFinanceFetcher
            self.fetcher = YFinanceFetcher()
        
        period = "2y"
        df = self.fetcher.download_history(ticker, period=period)
        if df.empty:
            raise ValueError(f"无法获取 {ticker} 数据")
        
        # 执行优化
        if method == "grid":
            results = self._grid_search(strategy_name, df, param_space)
        elif method == "random":
            results = self._random_search(strategy_name, df, param_space, n_trials)
        elif method == "bayesian":
            results = self._bayesian_optimization(strategy_name, df, param_space, n_trials)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
        
        # 找出最佳参数
        best_result = max(results, key=lambda x: x["score"])
        
        logger.info(f"优化完成，最佳得分: {best_result['score']:.4f}")
        
        return OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_result["params"],
            best_score=best_result["score"],
            all_results=results,
            optimization_method=method,
            n_trials=len(results)
        )
    
    def _grid_search(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        param_space: Dict
    ) -> List[Dict]:
        """网格搜索"""
        results = []
        
        # 生成所有参数组合
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        total_combinations = 1
        for v in param_values:
            total_combinations *= len(v)
        
        logger.info(f"网格搜索: {total_combinations} 种组合")
        
        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))
            
            # 回测评估
            score = self._evaluate_params(strategy_name, df, params)
            
            results.append({
                "params": params,
                "score": score
            })
        
        return results
    
    def _random_search(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        param_space: Dict,
        n_trials: int
    ) -> List[Dict]:
        """随机搜索"""
        results = []
        
        logger.info(f"随机搜索: {n_trials} 次试验")
        
        for _ in range(n_trials):
            # 随机采样参数
            params = {}
            for name, values in param_space.items():
                params[name] = random.choice(values)
            
            # 回测评估
            score = self._evaluate_params(strategy_name, df, params)
            
            results.append({
                "params": params,
                "score": score
            })
        
        return results
    
    def _bayesian_optimization(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        param_space: Dict,
        n_trials: int
    ) -> List[Dict]:
        """贝叶斯优化（使用 Optuna）"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna 未安装，回退到随机搜索")
            return self._random_search(strategy_name, df, param_space, n_trials)
        
        results = []
        
        def objective(trial):
            # 采样参数
            params = {}
            for name, values in param_space.items():
                if isinstance(values[0], int):
                    params[name] = trial.suggest_int(name, min(values), max(values))
                elif isinstance(values[0], float):
                    params[name] = trial.suggest_float(name, min(values), max(values))
                else:
                    params[name] = trial.suggest_categorical(name, values)
            
            # 回测评估
            score = self._evaluate_params(strategy_name, df, params)
            
            results.append({
                "params": params,
                "score": score
            })
            
            return score
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return results
    
    def _evaluate_params(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        params: Dict
    ) -> float:
        """评估参数组合"""
        try:
            # 创建策略实例
            strategy = self._create_strategy(strategy_name, params)
            
            # 生成信号（使用 _compute_signals 而非不存在的 generate_signals）
            sig_df = strategy._compute_signals(df)
            if sig_df.empty or "signal" not in sig_df.columns:
                return 0.0
            
            # 将信号DataFrame与原始数据合并，用于回测
            df_with_signals = df.copy()
            # 对齐索引
            common_idx = df.index.intersection(sig_df.index)
            df_with_signals = df_with_signals.loc[common_idx]
            for col in sig_df.columns:
                df_with_signals[col] = sig_df.loc[common_idx, col]
            
            # 简单回测
            metrics = self._simple_backtest(df_with_signals)
            
            # 返回目标得分
            if self.scoring == "sharpe":
                return metrics.get("sharpe_ratio", 0)
            elif self.scoring == "return":
                return metrics.get("total_return", 0)
            elif self.scoring == "max_drawdown":
                return -metrics.get("max_drawdown", 1)  # 负数，最小化回撤
            else:
                return metrics.get("sharpe_ratio", 0)
                
        except Exception as e:
            logger.debug(f"参数评估失败: {e}")
            return 0.0
    
    def _create_strategy(self, strategy_name: str, params: Dict):
        """创建策略实例"""
        from strategy.technical import MAStrategy, RSIStrategy, MACDStrategy, BollingerStrategy
        from strategy.composite import CompositeStrategy
        
        strategy_map = {
            "MA": MAStrategy,
            "RSI": RSIStrategy,
            "MACD": MACDStrategy,
            "Bollinger": BollingerStrategy,
            "Composite": CompositeStrategy
        }
        
        strategy_class = strategy_map.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"未知策略: {strategy_name}")
        
        return strategy_class(**params)
    
    def _simple_backtest(self, df: pd.DataFrame) -> Dict:
        """简单回测"""
        initial_cash = 100000
        cash = initial_cash
        position = 0
        trades = []
        
        close = df["Close"]
        signal = df["signal"]
        
        for i in range(1, len(df)):
            if signal.iloc[i] == "BUY" and position == 0:  # 买入信号
                shares = int(cash / close.iloc[i])
                if shares > 0:
                    cash -= shares * close.iloc[i]
                    position = shares
                    trades.append({"type": "BUY", "price": close.iloc[i], "shares": shares})
            
            elif signal.iloc[i] == "SELL" and position > 0:  # 卖出信号
                cash += position * close.iloc[i]
                trades.append({"type": "SELL", "price": close.iloc[i], "shares": position})
                position = 0
        
        # 最终价值
        final_value = cash + position * close.iloc[-1]
        total_return = (final_value - initial_cash) / initial_cash
        
        # 计算夏普比率
        if len(trades) > 1:
            returns = pd.Series([t["price"] for t in trades]).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "final_value": final_value,
            "num_trades": len(trades)
        }
    
    def save_results(
        self,
        result: OptimizationResult,
        output_dir: str = "optimization"
    ):
        """保存优化结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{result.strategy_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_path / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "strategy_name": result.strategy_name,
                "best_params": result.best_params,
                "best_score": result.best_score,
                "optimization_method": result.optimization_method,
                "n_trials": result.n_trials,
                "all_results": result.all_results[:100],  # 只保存前100个
                "timestamp": result.timestamp.isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"优化结果已保存: {filepath}")
    
    def get_optimization_summary(self, result: OptimizationResult) -> str:
        """获取优化摘要"""
        return f"""
## 策略优化结果

- **策略**: {result.strategy_name}
- **方法**: {result.optimization_method}
- **试验次数**: {result.n_trials}
- **最佳得分**: {result.best_score:.4f}

### 最佳参数
```json
{json.dumps(result.best_params, indent=2)}
```

### Top 5 参数组合
| 排名 | 得分 | 参数 |
|------|------|------|
""" + "\n".join([
            f"| {i+1} | {r['score']:.4f} | {r['params']} |"
            for i, r in enumerate(sorted(result.all_results, key=lambda x: x["score"], reverse=True)[:5])
        ])
