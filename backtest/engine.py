# ============================================================
# backtest/engine.py — 回测引擎
# 基于 backtrader，支持多策略、绩效分析、信号日志
# ============================================================
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 绩效指标计算
# ============================================================

def compute_metrics(equity_curve: pd.Series, trades: list[dict],
                    benchmark: Optional[pd.Series] = None) -> dict:
    """
    计算完整绩效指标。

    参数:
        equity_curve: 每日净值 Series，index=日期
        trades: 交易记录列表，每条含 {price, quantity, action}
        benchmark: 基准每日价格 Series（同 index）

    返回:
        指标 dict
    """
    if equity_curve.empty:
        return {}

    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1 if len(equity_curve) > 1 else 0

    # 年化收益
    n_days = len(equity_curve)
    years = n_days / 252
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 波动率
    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0

    # 夏普比率（假设无风险利率 4.5%）
    risk_free = 0.045
    sharpe = (annualized - risk_free) / volatility if volatility > 0 else 0

    # 最大回撤
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()

    # 卡玛比率
    calmar = annualized / abs(max_dd) if max_dd != 0 else 0

    # 胜率（基于已平仓交易）
    wins = 0
    total_trades = 0
    total_pnl = 0
    winning_pnl = 0
    losing_pnl = 0
    
    for t in trades:
        if t.get("action") == "SELL" and "realized_pnl" in t:
            pnl = t.get("realized_pnl", 0)
            total_trades += 1
            total_pnl += pnl
            if pnl > 0:
                wins += 1
                winning_pnl += pnl
            else:
                losing_pnl += abs(pnl)
    
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # 盈亏比（平均盈利 / 平均亏损）
    avg_win = winning_pnl / wins if wins > 0 else 0
    avg_loss = losing_pnl / (total_trades - wins) if (total_trades - wins) > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

    # 交易次数
    n_trades = len([t for t in trades if t.get("action") in ("BUY", "SELL")])

    # 基准对比
    excess_return = 0.0
    beta = 0.0
    if benchmark is not None and not benchmark.empty:
        bm_rets = benchmark.pct_change().dropna()
        aligned_rets, aligned_bm = returns.align(bm_rets, join="inner")
        if len(aligned_rets) > 10:
            excess_return = total_return - ((benchmark.iloc[-1] / benchmark.iloc[0]) - 1)
            cov = np.cov(aligned_rets, aligned_bm)[0][1]
            var_bm = np.var(aligned_bm)
            beta = cov / var_bm if var_bm > 0 else 0

    return {
        "total_return": float(total_return) * 100,        # 百分比
        "annualized_return": float(annualized) * 100,
        "volatility": float(volatility) * 100,
        "sharpe_ratio": round(float(sharpe), 3),
        "max_drawdown": float(max_dd) * 100,
        "calmar_ratio": round(float(calmar), 3),
        "win_rate": round(float(win_rate) * 100, 1),
        "n_trades": n_trades,
        "total_trades": total_trades,  # 已平仓交易数
        "profit_factor": round(float(profit_factor), 3),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "excess_return": float(excess_return) * 100,
        "beta": round(float(beta), 3),
        "final_equity": float(equity_curve.iloc[-1]),
        "n_days": n_days,
    }


# ============================================================
# 回测报告生成器
# ============================================================

class BacktestReport:
    """
    生成回测报告，包含指标 + HTML 可视化。
    """

    def __init__(self, output_dir: str = "output/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate(
        self,
        equity_curve: pd.Series,
        trades: list[dict],
        signals: list[dict],
        metrics: dict,
        strategy_name: str = "Strategy",
        ticker: str = "AAPL",
    ) -> str:
        """生成报告并保存"""
        report_path = self.output_dir / f"report_{strategy_name}_{ticker}_{self.timestamp}.html"
        self._save_html(report_path, equity_curve, trades, signals, metrics, strategy_name, ticker)

        # 保存 JSON 摘要
        summary_path = self.output_dir / f"summary_{strategy_name}_{ticker}_{self.timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"[Backtest] 报告已生成: {report_path}")
        return str(report_path)

    def _save_html(
        self,
        path: Path,
        equity_curve: pd.Series,
        trades: list[dict],
        signals: list[dict],
        metrics: dict,
        strategy_name: str,
        ticker: str,
    ):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("净值曲线", "回撤", "交易信号"),
            row_heights=[0.4, 0.25, 0.35],
            vertical_spacing=0.08,
        )

        # 1. 净值曲线
        fig.add_trace(go.Scatter(
            x=equity_curve.index, y=equity_curve.values,
            mode="lines", name="组合净值",
            line=dict(color="#2196F3", width=2),
        ), row=1, col=1)

        # 2. 回撤
        cummax = equity_curve.cummax()
        dd = (equity_curve - cummax) / cummax * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines",
            name="回撤%", line=dict(color="#F44336"),
            fill="tozeroy", fillcolor="rgba(244,67,54,0.1)",
        ), row=2, col=1)

        # 3. 交易信号
        buy_trades = [t for t in trades if t.get("action") == "BUY"]
        sell_trades = [t for t in trades if t.get("action") == "SELL"]
        
        # 价格曲线
        if signals:
            prices = [(s.get("timestamp"), s.get("price")) for s in signals if s.get("price")]
            if prices:
                dates, prices_vals = zip(*prices)
                fig.add_trace(go.Scatter(
                    x=list(dates), y=list(prices_vals),
                    mode="lines", name="价格", line=dict(color="#666", width=1),
                ), row=3, col=1)
        
        if buy_trades:
            fig.add_trace(go.Scatter(
                x=[t["timestamp"] for t in buy_trades],
                y=[t["price"] for t in buy_trades],
                mode="markers", name="买入",
                marker=dict(color="green", size=10, symbol="triangle-up"),
            ), row=3, col=1)
        if sell_trades:
            fig.add_trace(go.Scatter(
                x=[t["timestamp"] for t in sell_trades],
                y=[t["price"] for t in sell_trades],
                mode="markers", name="卖出",
                marker=dict(color="red", size=10, symbol="triangle-down"),
            ), row=3, col=1)

        # 4. 关键指标文本（添加到图表标题中）
        metrics_text = (
            f"总收益率: {metrics.get('total_return', 0):.2f}% | "
            f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f} | "
            f"最大回撤: {metrics.get('max_drawdown', 0):.2f}% | "
            f"胜率: {metrics.get('win_rate', 0):.1f}% | "
            f"盈亏比: {metrics.get('profit_factor', 0):.3f}"
        )

        fig.update_layout(
            title=f"回测报告 — {strategy_name} | {ticker}<br><sup>{metrics_text}</sup>",
            height=900,
            showlegend=True,
            template="plotly_white",
        )
        fig.write_html(str(path))


# ============================================================
# 简单回测运行器（修复版：正确追踪成本和盈亏）
# ============================================================

class SimpleBacktester:
    """
    轻量级回测器：给定信号序列 + OHLCV 数据，计算完整绩效。
    适合快速原型验证。

    修复内容：
    - 正确追踪每笔买入的成本基础
    - 修复 realized_pnl 计算
    - 支持多次买入的平均成本
    """

    def __init__(
        self,
        initial_cash: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0.0
        self.cost_basis = 0.0  # 成本基础（用于计算盈亏）
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.equity_dates: list[str] = []

    def run(self, df: pd.DataFrame, signals: pd.DataFrame, ticker: str = "TICKER") -> dict:
        """
        运行回测。

        参数:
            df: OHLCV DataFrame（含 Close 列）
            signals: 信号 DataFrame（列: signal/confidence/strength/reason）
            ticker: 标的代码

        返回:
            含 equity_curve/trades/metrics 的 dict
        """
        self.reset()
        df = df.copy()
        if "signal" not in signals.columns:
            logger.warning("signals DataFrame 缺少 signal 列")
            return {}

        # 对齐索引
        common_idx = df.index.intersection(signals.index)
        df = df.loc[common_idx]
        signals = signals.loc[common_idx]

        for i, (date, row) in enumerate(df.iterrows()):
            close = row["Close"]
            sig = signals.loc[date, "signal"]
            confidence = signals.loc[date, "confidence"] if "confidence" in signals.columns else 0.5
            strength = signals.loc[date, "strength"] if "strength" in signals.columns else 0.5
            reason = signals.loc[date, "reason"] if "reason" in signals.columns else ""

            # 执行交易
            if sig == "BUY" and self.cash > 0 and confidence >= 0.5:
                # 计算买入数量（使用 95% 现金）
                buy_amount = self.cash * 0.95
                price_with_slip = close * (1 + self.slippage)
                qty = buy_amount / price_with_slip
                cost = qty * price_with_slip
                comm = cost * self.commission
                
                # 更新持仓和成本
                # 新成本 = (旧成本 + 新买入成本) / 新总持仓
                old_value = self.position * self.cost_basis
                new_value = qty * price_with_slip
                new_position = self.position + qty
                
                self.cost_basis = (old_value + new_value) / new_position if new_position > 0 else price_with_slip
                self.position = new_position
                self.cash -= (cost + comm)
                
                self.trades.append({
                    "timestamp": str(date),
                    "ticker": ticker,
                    "action": "BUY",
                    "quantity": qty,
                    "price": price_with_slip,
                    "commission": comm,
                    "slippage": close * self.slippage,
                    "confidence": confidence,
                    "reason": reason,
                    "cost_basis": self.cost_basis,
                    "cash_after": self.cash,
                })

            elif sig == "SELL" and self.position > 0 and confidence >= 0.5:
                # 卖出全部持仓
                price_with_slip = close * (1 - self.slippage)
                sell_qty = self.position
                proceeds = sell_qty * price_with_slip
                comm = proceeds * self.commission
                
                # 计算已实现盈亏
                realized_pnl = proceeds - (sell_qty * self.cost_basis) - comm
                
                self.trades.append({
                    "timestamp": str(date),
                    "ticker": ticker,
                    "action": "SELL",
                    "quantity": sell_qty,
                    "price": price_with_slip,
                    "commission": comm,
                    "slippage": close * self.slippage,
                    "confidence": confidence,
                    "reason": reason,
                    "cost_basis": self.cost_basis,
                    "realized_pnl": realized_pnl,
                    "pnl_pct": (price_with_slip - self.cost_basis) / self.cost_basis * 100,
                    "cash_before": self.cash,
                })
                
                self.cash += (proceeds - comm)
                self.position = 0
                self.cost_basis = 0

            # 记录当日净值
            pos_value = self.position * close
            total_equity = self.cash + pos_value
            self.equity_curve.append(total_equity)
            self.equity_dates.append(date)

        # 最后一个交易日若仍有持仓，按收盘价计算
        if self.position > 0 and len(df) > 0:
            last_close = df["Close"].iloc[-1]
            self.equity_curve[-1] = self.cash + self.position * last_close

        equity_series = pd.Series(self.equity_curve, index=pd.to_datetime(self.equity_dates))
        metrics = compute_metrics(equity_series, self.trades)

        return {
            "equity_curve": equity_series,
            "trades": self.trades,
            "metrics": metrics,
            "initial_cash": self.initial_cash,
            "final_equity": self.equity_curve[-1] if self.equity_curve else self.initial_cash,
        }

    def summary(self) -> str:
        """打印简洁摘要"""
        if not self.equity_curve:
            return "无回测数据"
        final = self.equity_curve[-1]
        ret = (final / self.initial_cash - 1) * 100
        return (f"初始资金: ${self.initial_cash:,.0f} | "
                f"最终净值: ${final:,.0f} | "
                f"收益率: {ret:+.2f}% | "
                f"交易次数: {len(self.trades)}")
