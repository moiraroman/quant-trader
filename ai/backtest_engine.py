# ============================================================
# ai/backtest_engine.py — 策略回测引擎
# 基于历史数据验证策略表现，提供胜率/收益/回撤统计
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测结果"""
    ticker: str
    strategy_name: str = ""  # 策略名称
    # 基本统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    median_return: float = 0.0
    max_return: float = 0.0
    min_return: float = 0.0
    # 风险指标
    max_drawdown: float = 0.0  # 最大回撤 %
    avg_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    # 收益分布
    returns: list = field(default_factory=list)  # 每笔收益 %
    cumulative_returns: list = field(default_factory=list)
    trade_dates: list = field(default_factory=list)
    # 分年度统计
    annual_stats: dict = field(default_factory=dict)
    # 信号质量
    signal_accuracy: float = 0.0  # 信号方向准确率
    early_exit_rate: float = 0.0  # 提前止损率
    # 缺失数据
    missing_data: list = field(default_factory=list)
    # 免责声明
    disclaimer: str = "历史回测不代表未来表现，结果仅供研究参考"


@dataclass
class TradeRecord:
    """单笔交易记录"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    direction: str  # "long" / "short"
    return_pct: float
    exit_reason: str  # "target" / "stop" / "timeout" / "signal_flip"
    holding_days: int


def _get_signal_from_analysis(
    ticker: str,
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
) -> tuple[str, float]:
    """
    基于技术分析生成交易信号（简化版）。
    使用RSI+MACD+均线交叉综合判断。
    返回: (signal, confidence)
    """
    if idx < lookback + 26:
        return "neutral", 0.0

    window = df.iloc[idx - lookback:idx]
    close = window["Close"].values
    high = window["High"].values
    low = window["Low"].values
    volume = window["Volume"].values if "Volume" in window.columns else np.ones(len(close))

    # RSI
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
    macd_line = ema12 - ema26
    signal_line = pd.Series(close).ewm(span=9, adjust=False).mean().iloc[-1]
    macd_hist = macd_line - signal_line

    # 均线
    sma20 = np.mean(close[-20:])
    sma50 = np.mean(close[-50:]) if len(close) >= 50 else sma20

    # 信号评分
    bull_score = 0
    bear_score = 0

    if rsi < 30:
        bull_score += 2
    elif rsi > 70:
        bear_score += 2
    elif rsi < 45:
        bull_score += 1
    elif rsi > 55:
        bear_score += 1

    if macd_hist > 0 and macd_hist > (macd_hist * 0.9 if macd_hist != 0 else 0):
        bull_score += 1
    elif macd_hist < 0:
        bear_score += 1

    if close[-1] > sma20 > sma50:
        bull_score += 2
    elif close[-1] < sma20 < sma50:
        bear_score += 2

    # 成交量确认
    vol_avg = np.mean(volume[-10:])
    if volume[-1] > vol_avg * 1.2:
        if bull_score > bear_score:
            bull_score += 1
        elif bear_score > bull_score:
            bear_score += 1

    if bull_score >= 4 and bull_score > bear_score + 1:
        confidence = min(85, 50 + bull_score * 5)
        return "long", confidence
    elif bear_score >= 4 and bear_score > bull_score + 1:
        confidence = min(85, 50 + bear_score * 5)
        return "short", confidence
    else:
        return "neutral", 30.0


def backtest_strategy(
    ticker: str,
    fetcher,
    strategy_type: str = "combined",  # "combined" / "rsi_only" / "macd_only" / "trend_only"
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    holding_period: int = 5,  # 持仓天数（5天或20天）
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 6.0,
    initial_capital: float = 10000.0,
) -> BacktestResult:
    """
    对指定策略进行历史回测。

    参数:
        ticker: 标的代码
        fetcher: YFinanceFetcher实例
        strategy_type: 策略类型
        start_date: 回测开始日期（默认2年前）
        end_date: 回测结束日期（默认今天）
        holding_period: 持仓周期（5=短线，20=中线）
        stop_loss_pct: 止损百分比
        take_profit_pct: 止盈百分比
        initial_capital: 初始资金

    返回:
        BacktestResult
    """
    result = BacktestResult(ticker=ticker, strategy_name=strategy_type)

    # 获取历史数据（至少3年）
    try:
        df = fetcher.download_history(ticker, period="3y", interval="1d")
        if df.empty or len(df) < 252:
            result.missing_data.append("历史数据不足（需至少1年）")
            return result
    except Exception as e:
        logger.warning(f"[Backtest] {ticker} 获取历史数据失败: {e}")
        result.missing_data.append(f"历史数据获取失败: {e}")
        return result

    # 日期过滤
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    if len(df) < 100:
        result.missing_data.append("过滤后数据不足100天")
        return result

    trades: list[TradeRecord] = []
    equity_curve = [initial_capital]
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    entry_date = ""
    position_direction = ""

    for i in range(60, len(df) - holding_period):
        if in_position:
            # 检查退出条件
            current_price = float(df["Close"].iloc[i])
            holding_days = i - entry_idx

            # 计算当前收益
            if position_direction == "long":
                current_return = (current_price - entry_price) / entry_price * 100
            else:
                current_return = (entry_price - current_price) / entry_price * 100

            exit_reason = ""
            if current_return <= -stop_loss_pct:
                exit_reason = "stop"
            elif current_return >= take_profit_pct:
                exit_reason = "target"
            elif holding_days >= holding_period:
                exit_reason = "timeout"
            else:
                # 检查信号反转
                new_signal, _ = _get_signal_from_analysis(ticker, df, i)
                if (position_direction == "long" and new_signal == "short") or \
                   (position_direction == "short" and new_signal == "long"):
                    exit_reason = "signal_flip"

            if exit_reason:
                trades.append(TradeRecord(
                    entry_date=entry_date,
                    exit_date=str(df.index[i]),
                    entry_price=entry_price,
                    exit_price=current_price,
                    direction=position_direction,
                    return_pct=current_return,
                    exit_reason=exit_reason,
                    holding_days=holding_days,
                ))
                in_position = False

        else:
            # 检查入场信号
            signal, confidence = _get_signal_from_analysis(ticker, df, i)
            if signal in ("long", "short") and confidence >= 50:
                in_position = True
                entry_price = float(df["Close"].iloc[i])
                entry_idx = i
                entry_date = str(df.index[i])
                position_direction = signal

    # 统计计算
    if not trades:
        result.missing_data.append("回测期间无交易信号")
        return result

    returns = [t.return_pct for t in trades]
    result.total_trades = len(trades)
    result.winning_trades = sum(1 for r in returns if r > 0)
    result.losing_trades = sum(1 for r in returns if r <= 0)
    result.win_rate = round(result.winning_trades / len(trades) * 100, 1)
    result.avg_return = round(np.mean(returns), 2)
    result.median_return = round(np.median(returns), 2)
    result.max_return = round(max(returns), 2)
    result.min_return = round(min(returns), 2)
    result.returns = [round(r, 2) for r in returns]

    # 资金曲线
    capital = initial_capital
    cum_returns = []
    for r in returns:
        capital *= (1 + r / 100)
        cum_returns.append(round((capital - initial_capital) / initial_capital * 100, 2))
    result.cumulative_returns = cum_returns
    result.trade_dates = [t.exit_date for t in trades]

    # 最大回撤
    peak = 0
    max_dd = 0
    for cr in cum_returns:
        if cr > peak:
            peak = cr
        dd = peak - cr
        if dd > max_dd:
            max_dd = dd
    result.max_drawdown = round(max_dd, 2)

    # 平均回撤
    dd_list = []
    peak = 0
    for cr in cum_returns:
        if cr > peak:
            peak = cr
        dd_list.append(peak - cr)
    result.avg_drawdown = round(np.mean(dd_list), 2) if dd_list else 0.0

    # Sharpe (简化，假设无风险利率0)
    if np.std(returns) > 0:
        result.sharpe_ratio = round(np.mean(returns) / np.std(returns) * np.sqrt(252 / holding_period), 2)

    # 信号方向准确率（信号方向 vs 实际方向）
    correct_direction = 0
    for t in trades:
        actual_direction = "long" if t.return_pct > 0 else "short"
        if t.direction == actual_direction:
            correct_direction += 1
    result.signal_accuracy = round(correct_direction / len(trades) * 100, 1)

    # 提前止损率
    early_stops = sum(1 for t in trades if t.exit_reason == "stop")
    result.early_exit_rate = round(early_stops / len(trades) * 100, 1)

    # 分年度统计
    annual_returns: dict[str, list] = {}
    for t in trades:
        year = t.exit_date[:4]
        if year not in annual_returns:
            annual_returns[year] = []
        annual_returns[year].append(t.return_pct)

    for year, rets in annual_returns.items():
        wins = sum(1 for r in rets if r > 0)
        result.annual_stats[year] = {
            "交易次数": len(rets),
            "胜率": f"{round(wins / len(rets) * 100, 1)}%",
            "平均收益": f"{round(np.mean(rets), 2)}%",
            "总收益": f"{round(sum(rets), 2)}%",
        }

    logger.info(f"[Backtest] {ticker} {strategy_type} 回测完成: {result.total_trades}笔交易, 胜率{result.win_rate}%")
    return result


def format_backtest_result(result: BacktestResult) -> dict:
    """格式化回测结果为UI展示格式"""
    return {
        "标的": result.ticker,
        "策略": result.strategy_name,
        "回测统计": {
            "总交易次数": result.total_trades,
            "盈利次数": result.winning_trades,
            "亏损次数": result.losing_trades,
            "胜率": f"{result.win_rate}%",
            "平均收益": f"{result.avg_return}%",
            "中位数收益": f"{result.median_return}%",
            "最大单笔盈利": f"{result.max_return}%",
            "最大单笔亏损": f"{result.min_return}%",
        },
        "风险指标": {
            "最大回撤": f"{result.max_drawdown}%",
            "平均回撤": f"{result.avg_drawdown}%",
            "夏普比率": result.sharpe_ratio,
        },
        "信号质量": {
            "方向准确率": f"{result.signal_accuracy}%",
            "提前止损率": f"{result.early_exit_rate}%",
        },
        "分年度表现": result.annual_stats,
        "最近10笔交易收益": result.returns[-10:] if len(result.returns) >= 10 else result.returns,
        "资金曲线总收益": f"{result.cumulative_returns[-1]}%" if result.cumulative_returns else "N/A",
        "免责声明": result.disclaimer,
        "缺少数据": result.missing_data,
    }


def compare_strategies(
    ticker: str,
    fetcher,
    strategies: list[str] = None,
    holding_period: int = 5,
) -> dict:
    """
    对比多个策略的回测表现。
    返回: {strategy_name: BacktestResult}
    """
    if strategies is None:
        strategies = ["combined", "rsi_only", "macd_only", "trend_only"]

    results = {}
    for strat in strategies:
        try:
            bt = backtest_strategy(ticker, fetcher, strategy_type=strat, holding_period=holding_period)
            results[strat] = format_backtest_result(bt)
        except Exception as e:
            logger.warning(f"[Backtest] {ticker} {strat} 回测失败: {e}")
            results[strat] = {"error": str(e)}

    return results
