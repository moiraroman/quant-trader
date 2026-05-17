# ============================================================
# ai/portfolio_optimizer.py — 组合优化器
# 马科维茨均值-方差优化 + 风险平价
# ============================================================
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入scipy优化器
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("[Portfolio] scipy未安装，使用解析近似方案")


@dataclass
class PortfolioAsset:
    """组合中的资产"""
    ticker: str
    expected_return: float = 0.0  # 年化预期收益率
    volatility: float = 0.0       # 年化波动率
    weight: float = 0.0           # 优化后权重
    # 约束
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class PortfolioResult:
    """组合优化结果"""
    tickers: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)       # {ticker: weight}
    expected_return: float = 0.0   # 年化预期收益
    volatility: float = 0.0        # 年化波动率
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    # 风险贡献
    risk_contributions: dict = field(default_factory=dict)
    # 有效前沿
    efficient_frontier: list = field(default_factory=list)  # [(return, vol, sharpe)]
    # 优化方法
    method: str = ""
    # 相关性矩阵
    correlation_matrix: dict = field(default_factory=dict)
    # 缺失数据
    missing_data: list = field(default_factory=list)
    # 免责声明
    disclaimer: str = "组合优化基于历史数据，未来收益和风险可能与历史表现显著不同"


def _calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """计算日收益率"""
    return df.pct_change().dropna()


def _portfolio_performance(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.04) -> tuple:
    """计算组合收益、波动率、Sharpe"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe


def _negative_sharpe(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.04) -> float:
    """负Sharpe（用于最小化）"""
    p_ret, p_vol, sharpe = _portfolio_performance(weights, returns, risk_free_rate)
    return -sharpe


def _portfolio_volatility(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """组合波动率"""
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))


def _risk_parity_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """风险平价目标函数：最小化风险贡献差异"""
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    if portfolio_var <= 0:
        return 1e10
    marginal_risk = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_risk / portfolio_var
    target_risk = 1.0 / len(weights)
    return np.sum((risk_contrib - target_risk) ** 2)


def optimize_markowitz(
    tickers: list[str],
    fetcher,
    risk_free_rate: float = 0.04,
    target_return: Optional[float] = None,
    max_volatility: Optional[float] = None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    lookback_days: int = 252,
) -> PortfolioResult:
    """
    马科维茨均值-方差优化。

    参数:
        tickers: 标的列表
        fetcher: YFinanceFetcher
        risk_free_rate: 无风险利率（年化）
        target_return: 目标收益约束（可选）
        max_volatility: 最大波动率约束（可选）
        min_weight: 最小权重
        max_weight: 最大权重
        lookback_days: 历史回看天数

    返回:
        PortfolioResult
    """
    result = PortfolioResult(tickers=tickers, method="markowitz")

    # 获取历史数据
    returns_data = {}
    for ticker in tickers:
        try:
            df = fetcher.download_history(ticker, period=f"{lookback_days + 20}d", interval="1d")
            if df.empty or len(df) < 30:
                result.missing_data.append(f"{ticker} 历史数据不足")
                continue
            returns_data[ticker] = df["Close"].pct_change().dropna()
        except Exception as e:
            logger.warning(f"[Portfolio] {ticker} 获取失败: {e}")
            result.missing_data.append(f"{ticker} 数据获取失败")

    if len(returns_data) < 2:
        result.missing_data.append("有效标的不足2个，无法优化")
        return result

    # 构建收益率矩阵
    returns_df = pd.DataFrame(returns_data).dropna()
    if len(returns_df) < 30:
        result.missing_data.append("共同交易日不足")
        return result

    n = len(tickers)
    returns_matrix = returns_df.values
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values * 252  # 年化

    # 相关性矩阵
    corr = returns_df.corr()
    result.correlation_matrix = {t: {c: round(corr.loc[t, c], 2) for c in corr.columns} for t in corr.index}

    if not SCIPY_AVAILABLE:
        # 解析近似：等权重
        weights = np.array([1.0 / n] * n)
        result.weights = {t: round(w, 3) for t, w in zip(tickers, weights)}
        result.expected_return, result.volatility, result.sharpe_ratio = _portfolio_performance(weights, returns_df, risk_free_rate)
        result.missing_data.append("scipy未安装，使用等权重近似")
        return result

    # 约束
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w: np.sum(mean_returns * w) * 252 - target_return})
    if max_volatility is not None:
        constraints.append({"type": "ineq", "fun": lambda w: max_volatility - _portfolio_volatility(w, returns_df)})

    bounds = tuple((min_weight, max_weight) for _ in range(n))
    init_guess = np.array([1.0 / n] * n)

    # 1. 最大Sharpe组合
    try:
        opt_sharpe = minimize(
            _negative_sharpe,
            init_guess,
            args=(returns_df, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if opt_sharpe.success:
            weights_sharpe = opt_sharpe.x
            result.weights = {t: round(w, 4) for t, w in zip(tickers, weights_sharpe)}
            result.expected_return, result.volatility, result.sharpe_ratio = _portfolio_performance(weights_sharpe, returns_df, risk_free_rate)
        else:
            result.missing_data.append("最大Sharpe优化未收敛，使用等权重")
            weights_sharpe = init_guess
            result.weights = {t: round(w, 4) for t, w in zip(tickers, weights_sharpe)}
    except Exception as e:
        logger.warning(f"[Portfolio] 优化失败: {e}")
        result.missing_data.append(f"优化失败: {e}")
        weights_sharpe = init_guess
        result.weights = {t: round(w, 4) for t, w in zip(tickers, weights_sharpe)}

    # 2. 有效前沿
    try:
        target_returns = np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, 20)
        for tr in target_returns:
            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w: np.sum(mean_returns * w) * 252 - tr},
            ]
            opt = minimize(
                _portfolio_volatility,
                init_guess,
                args=(returns_df,),
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )
            if opt.success:
                vol = opt.fun
                ret = tr
                sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
                result.efficient_frontier.append({
                    "return": round(ret * 100, 2),
                    "volatility": round(vol * 100, 2),
                    "sharpe": round(sharpe, 2),
                })
    except Exception as e:
        logger.warning(f"[Portfolio] 有效前沿计算失败: {e}")

    # 3. 风险贡献
    try:
        w = np.array([result.weights.get(t, 0) for t in tickers])
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        if port_var > 0:
            marginal = np.dot(cov_matrix, w)
            risk_contrib = w * marginal / port_var
            result.risk_contributions = {t: round(rc * 100, 2) for t, rc in zip(tickers, risk_contrib)}
    except Exception as e:
        logger.warning(f"[Portfolio] 风险贡献计算失败: {e}")

    # 4. 最大回撤估算
    try:
        portfolio_returns = (returns_df * np.array([result.weights.get(t, 0) for t in tickers])).sum(axis=1)
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        result.max_drawdown = round(float(drawdowns.min()) * 100, 2)
    except Exception as e:
        logger.warning(f"[Portfolio] 最大回撤计算失败: {e}")

    logger.info(f"[Portfolio] 马科维茨优化完成: 预期收益{result.expected_return*100:.1f}%, 波动{result.volatility*100:.1f}%, Sharpe{result.sharpe_ratio:.2f}")
    return result


def optimize_risk_parity(
    tickers: list[str],
    fetcher,
    lookback_days: int = 252,
) -> PortfolioResult:
    """
    风险平价优化：各资产对组合风险贡献相等。
    """
    result = PortfolioResult(tickers=tickers, method="risk_parity")

    returns_data = {}
    for ticker in tickers:
        try:
            df = fetcher.download_history(ticker, period=f"{lookback_days + 20}d", interval="1d")
            if df.empty or len(df) < 30:
                result.missing_data.append(f"{ticker} 历史数据不足")
                continue
            returns_data[ticker] = df["Close"].pct_change().dropna()
        except Exception as e:
            result.missing_data.append(f"{ticker} 数据获取失败")

    if len(returns_data) < 2:
        result.missing_data.append("有效标的不足")
        return result

    returns_df = pd.DataFrame(returns_data).dropna()
    n = len(tickers)
    cov_matrix = returns_df.cov().values * 252

    if not SCIPY_AVAILABLE:
        # 近似：波动率倒数权重
        vols = returns_df.std().values * np.sqrt(252)
        inv_vols = 1.0 / (vols + 1e-10)
        weights = inv_vols / inv_vols.sum()
        result.weights = {t: round(w, 4) for t, w in zip(tickers, weights)}
        result.expected_return, result.volatility, result.sharpe_ratio = _portfolio_performance(weights, returns_df)
        result.missing_data.append("scipy未安装，使用波动率倒数近似")
        return result

    init_guess = np.array([1.0 / n] * n)
    bounds = tuple((0.01, 0.99) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    try:
        opt = minimize(
            _risk_parity_objective,
            init_guess,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if opt.success:
            weights = opt.x
            result.weights = {t: round(w, 4) for t, w in zip(tickers, weights)}
            result.expected_return, result.volatility, result.sharpe_ratio = _portfolio_performance(weights, returns_df)

            # 风险贡献
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal / port_var
            result.risk_contributions = {t: round(rc * 100, 2) for t, rc in zip(tickers, risk_contrib)}
        else:
            result.missing_data.append("风险平价优化未收敛")
    except Exception as e:
        logger.warning(f"[Portfolio] 风险平价优化失败: {e}")
        result.missing_data.append(f"优化失败: {e}")

    return result


def format_portfolio_result(result: PortfolioResult) -> dict:
    """格式化组合优化结果为UI展示格式"""
    return {
        "标的": result.tickers,
        "优化方法": result.method,
        "权重分配": result.weights,
        "预期年化收益": f"{result.expected_return * 100:.2f}%",
        "预期年化波动": f"{result.volatility * 100:.2f}%",
        "Sharpe比率": round(result.sharpe_ratio, 2),
        "最大回撤": f"{result.max_drawdown}%",
        "风险贡献": result.risk_contributions,
        "相关性矩阵": result.correlation_matrix,
        "有效前沿(部分)": result.efficient_frontier[:5],
        "免责声明": result.disclaimer,
        "缺少数据": result.missing_data,
    }


def run_portfolio_analysis(
    tickers: list[str],
    fetcher,
    methods: list[str] = None,
) -> dict:
    """
    执行多种组合优化方法对比。

    返回:
        {"markowitz": ..., "risk_parity": ..., "equal_weight": ...}
    """
    if methods is None:
        methods = ["markowitz", "risk_parity", "equal_weight"]

    results = {}

    if "markowitz" in methods:
        results["markowitz"] = format_portfolio_result(optimize_markowitz(tickers, fetcher))

    if "risk_parity" in methods:
        results["risk_parity"] = format_portfolio_result(optimize_risk_parity(tickers, fetcher))

    if "equal_weight" in methods:
        # 等权重基准
        n = len(tickers)
        eq_result = PortfolioResult(tickers=tickers, method="equal_weight")
        eq_result.weights = {t: round(1.0 / n, 4) for t in tickers}
        try:
            returns_data = {}
            for t in tickers:
                df = fetcher.download_history(t, period="252d", interval="1d")
                if not df.empty:
                    returns_data[t] = df["Close"].pct_change().dropna()
            if len(returns_data) >= 2:
                returns_df = pd.DataFrame(returns_data).dropna()
                w = np.array([1.0 / n] * n)
                eq_result.expected_return, eq_result.volatility, eq_result.sharpe_ratio = _portfolio_performance(w, returns_df)
                eq_result.max_drawdown = 0.0  # 简化
        except Exception as e:
            eq_result.missing_data.append(f"等权重计算失败: {e}")
        results["equal_weight"] = format_portfolio_result(eq_result)

    return results
