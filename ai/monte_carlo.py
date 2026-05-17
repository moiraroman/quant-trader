# ============================================================
# ai/monte_carlo.py — 蒙特卡洛模拟
# 基于历史波动率生成价格路径概率分布
# ============================================================
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """蒙特卡洛模拟结果"""
    ticker: str
    current_price: float
    simulation_days: int = 20
    num_simulations: int = 1000
    # 价格统计
    mean_price: float = 0.0
    median_price: float = 0.0
    std_price: float = 0.0
    # 分位数
    p5_price: float = 0.0   # 5%分位（悲观）
    p25_price: float = 0.0  # 25%分位
    p75_price: float = 0.0  # 75%分位
    p95_price: float = 0.0  # 95%分位（乐观）
    # 收益率统计
    mean_return: float = 0.0
    median_return: float = 0.0
    p5_return: float = 0.0
    p95_return: float = 0.0
    # 概率
    prob_up: float = 0.0      # 上涨概率
    prob_down: float = 0.0    # 下跌概率
    prob_flat: float = 0.0    # 横盘概率（±1%）
    prob_target_hit: float = 0.0  # 达到目标价概率
    prob_stop_hit: float = 0.0    # 触及止损概率
    # 路径数据（仅保存部分用于可视化）
    sample_paths: list = field(default_factory=list)  # 50条样本路径
    final_prices: list = field(default_factory=list)
    # 风险指标
    var_95: float = 0.0  # 95% VaR（单日）
    cvar_95: float = 0.0  # 条件VaR
    max_simulated_drawdown: float = 0.0
    # 情景标签
    scenario_label: str = ""
    # 缺失数据
    missing_data: list = field(default_factory=list)
    # 免责声明
    disclaimer: str = "蒙特卡洛模拟基于历史波动率，假设收益正态分布，实际市场可能存在肥尾风险"


def _calculate_historical_params(df: pd.DataFrame, lookback: int = 60) -> tuple[float, float]:
    """计算历史收益率的均值和标准差"""
    returns = df["Close"].pct_change().dropna()
    if len(returns) < lookback:
        lookback = len(returns)
    recent_returns = returns.iloc[-lookback:]
    mu = recent_returns.mean()
    sigma = recent_returns.std()
    return mu, sigma


def run_monte_carlo(
    ticker: str,
    current_price: float,
    fetcher,
    simulation_days: int = 20,
    num_simulations: int = 1000,
    target_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    use_gbm: bool = True,  # True=几何布朗运动, False=简单随机游走
) -> MonteCarloResult:
    """
    执行蒙特卡洛价格路径模拟。

    参数:
        ticker: 标的代码
        current_price: 当前价格
        fetcher: YFinanceFetcher实例
        simulation_days: 模拟天数
        num_simulations: 模拟路径数
        target_price: 目标价（用于计算达到概率）
        stop_price: 止损价（用于计算触及概率）
        use_gbm: 使用几何布朗运动（更 realistic）

    返回:
        MonteCarloResult
    """
    result = MonteCarloResult(
        ticker=ticker,
        current_price=current_price,
        simulation_days=simulation_days,
        num_simulations=num_simulations,
    )

    # 获取历史数据计算波动率
    try:
        df = fetcher.download_history(ticker, period="1y", interval="1d")
        if df.empty or len(df) < 30:
            result.missing_data.append("历史数据不足（需至少30天）")
            # 使用默认参数
            mu = 0.0005
            sigma = 0.015
        else:
            mu, sigma = _calculate_historical_params(df, lookback=60)
    except Exception as e:
        logger.warning(f"[MonteCarlo] {ticker} 获取历史数据失败: {e}")
        result.missing_data.append(f"历史数据获取失败，使用默认波动率")
        mu = 0.0005
        sigma = 0.015

    # 生成随机路径
    np.random.seed(42)  # 可复现
    dt = 1.0  # 1天步长

    if use_gbm:
        # 几何布朗运动: dS/S = mu*dt + sigma*dW
        # S(t) = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
        random_shocks = np.random.standard_normal((simulation_days, num_simulations))
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks

        log_returns = drift + diffusion
        cum_log_returns = np.cumsum(log_returns, axis=0)
        price_paths = current_price * np.exp(cum_log_returns)
    else:
        # 简单随机游走
        random_shocks = np.random.standard_normal((simulation_days, num_simulations))
        daily_returns = mu * dt + sigma * np.sqrt(dt) * random_shocks
        cum_returns = np.cumsum(daily_returns, axis=0)
        price_paths = current_price * (1 + cum_returns)

    # 确保价格不为负
    price_paths = np.maximum(price_paths, 0.01)

    final_prices = price_paths[-1, :]
    result.final_prices = [round(p, 2) for p in final_prices]

    # 统计计算
    result.mean_price = round(float(np.mean(final_prices)), 2)
    result.median_price = round(float(np.median(final_prices)), 2)
    result.std_price = round(float(np.std(final_prices)), 2)
    result.p5_price = round(float(np.percentile(final_prices, 5)), 2)
    result.p25_price = round(float(np.percentile(final_prices, 25)), 2)
    result.p75_price = round(float(np.percentile(final_prices, 75)), 2)
    result.p95_price = round(float(np.percentile(final_prices, 95)), 2)

    # 收益率统计
    final_returns = (final_prices - current_price) / current_price * 100
    result.mean_return = round(float(np.mean(final_returns)), 2)
    result.median_return = round(float(np.median(final_returns)), 2)
    result.p5_return = round(float(np.percentile(final_returns, 5)), 2)
    result.p95_return = round(float(np.percentile(final_returns, 95)), 2)

    # 概率计算
    result.prob_up = round(float(np.mean(final_returns > 0)) * 100, 1)
    result.prob_down = round(float(np.mean(final_returns < 0)) * 100, 1)
    result.prob_flat = round(float(np.mean(np.abs(final_returns) < 1)) * 100, 1)

    if target_price:
        result.prob_target_hit = round(float(np.mean(final_prices >= target_price)) * 100, 1)
    if stop_price:
        # 检查路径中是否触及止损（任意一天）
        hit_stop = np.any(price_paths <= stop_price, axis=0)
        result.prob_stop_hit = round(float(np.mean(hit_stop)) * 100, 1)

    # 样本路径（保存50条用于可视化）
    sample_indices = np.random.choice(num_simulations, min(50, num_simulations), replace=False)
    for idx in sample_indices:
        path = price_paths[:, idx]
        result.sample_paths.append([round(p, 2) for p in path])

    # VaR计算（基于历史收益率）
    historical_returns = df["Close"].pct_change().dropna() if 'df' in dir() and not df.empty else pd.Series([0])
    if len(historical_returns) > 10:
        result.var_95 = round(float(np.percentile(historical_returns, 5)) * 100, 2)
        result.cvar_95 = round(float(historical_returns[historical_returns <= np.percentile(historical_returns, 5)].mean()) * 100, 2)

    # 最大模拟回撤
    max_dd_list = []
    for i in range(num_simulations):
        path = price_paths[:, i]
        peak = path[0]
        max_dd = 0
        for price in path:
            if price > peak:
                peak = price
            dd = (peak - price) / peak * 100
            if dd > max_dd:
                max_dd = dd
        max_dd_list.append(max_dd)
    result.max_simulated_drawdown = round(float(np.mean(max_dd_list)), 2)

    # 情景标签
    if result.prob_up > 60:
        result.scenario_label = "偏多"
    elif result.prob_down > 60:
        result.scenario_label = "偏空"
    else:
        result.scenario_label = "中性震荡"

    logger.info(f"[MonteCarlo] {ticker} 模拟完成: 均值{result.mean_price}, 5%分位{result.p5_price}, 95%分位{result.p95_price}")
    return result


def format_monte_carlo_result(result: MonteCarloResult) -> dict:
    """格式化蒙特卡洛结果为UI展示格式"""
    return {
        "标的": result.ticker,
        "当前价格": result.current_price,
        "模拟参数": {
            "模拟天数": result.simulation_days,
            "模拟路径数": result.num_simulations,
        },
        "价格预测": {
            "均值": result.mean_price,
            "中位数": result.median_price,
            "标准差": result.std_price,
            "5%分位(悲观)": result.p5_price,
            "25%分位": result.p25_price,
            "75%分位": result.p75_price,
            "95%分位(乐观)": result.p95_price,
        },
        "收益率预测": {
            "均值": f"{result.mean_return}%",
            "中位数": f"{result.median_return}%",
            "5%分位": f"{result.p5_return}%",
            "95%分位": f"{result.p95_return}%",
        },
        "概率分布": {
            "上涨概率": f"{result.prob_up}%",
            "下跌概率": f"{result.prob_down}%",
            "横盘概率(±1%)": f"{result.prob_flat}%",
            "达到目标价概率": f"{result.prob_target_hit}%" if result.prob_target_hit > 0 else "未设置目标价",
            "触及止损概率": f"{result.prob_stop_hit}%" if result.prob_stop_hit > 0 else "未设置止损价",
        },
        "风险指标": {
            "95% VaR(单日)": f"{result.var_95}%",
            "条件VaR": f"{result.cvar_95}%",
            "平均最大回撤": f"{result.max_simulated_drawdown}%",
        },
        "情景判断": result.scenario_label,
        "样本路径数": len(result.sample_paths),
        "免责声明": result.disclaimer,
        "缺少数据": result.missing_data,
    }


def run_multi_horizon_monte_carlo(
    ticker: str,
    current_price: float,
    fetcher,
    horizons: list[int] = None,
    num_simulations: int = 1000,
) -> dict:
    """
    多时间维度蒙特卡洛模拟（5天/20天/60天）。
    返回: {horizon_days: formatted_result}
    """
    if horizons is None:
        horizons = [5, 20, 60]

    results = {}
    for days in horizons:
        try:
            mc = run_monte_carlo(ticker, current_price, fetcher, simulation_days=days, num_simulations=num_simulations)
            results[f"{days}天"] = format_monte_carlo_result(mc)
        except Exception as e:
            logger.warning(f"[MonteCarlo] {ticker} {days}天模拟失败: {e}")
            results[f"{days}天"] = {"error": str(e)}

    return results
