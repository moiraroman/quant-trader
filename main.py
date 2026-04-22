# ============================================================
# main.py — 量化交易系统主入口
# 支持模式：回测 / 模拟交易 / 实盘
# ============================================================
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd

# 项目路径
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from data.fetcher import YFinanceFetcher
from data.storage import ParquetStorage, SQLiteStorage
from strategy.technical import MAStrategy, RSIStrategy, MACDStrategy, BollingerStrategy, EnsembleStrategy
from strategy.ai_model import AIStrategy
from backtest.engine import SimpleBacktester, BacktestReport, compute_metrics
from trading.paper import PaperTrader
from trading.live import MooMooLiveTrader
from risk.manager import RiskManager

# ============================================================
# 日志配置
# ============================================================

def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    fmt = "%(asctime)s [%(levelname)s] %(name)s %(message)s"

    handlers = []
    if log_cfg.get("console", True):
        handlers.append(logging.StreamHandler(sys.stdout))

    log_file = log_cfg.get("file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def load_config() -> dict:
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# ============================================================
# 模式1：回测
# ============================================================

def run_backtest(args, config: dict):
    """运行策略回测"""
    logger = logging.getLogger("backtest")
    logger.info("=" * 50)
    logger.info("启动回测模式")
    logger.info("=" * 50)

    # 数据
    bt_cfg = config.get("backtest", {})
    start = args.start or bt_cfg.get("start_date", "2023-01-01")
    end = args.end or bt_cfg.get("end_date", "2024-12-31")
    ticker = args.ticker or config.get("strategy", {}).get("default", {}).get("symbol", "AAPL")
    benchmark = bt_cfg.get("benchmark", "SPY")

    fetcher = YFinanceFetcher()
    df = fetcher.download_history(ticker, period="5y", interval="1d")
    if df.empty:
        logger.error(f"无法获取 {ticker} 数据，退出")
        return

    df = df[start:end] if start else df
    df = df[:end] if end else df
    logger.info(f"回测区间: {df.index[0].date()} ~ {df.index[-1].date()} 共 {len(df)} 条")

    # 策略
    strategy_cfg = config.get("strategy", {})
    tech_params = strategy_cfg.get("technical", {})

    strategies = {
        "MA": MAStrategy(short_window=tech_params.get("sma_short", 10),
                         long_window=tech_params.get("sma_long", 50)),
        "RSI": RSIStrategy(period=tech_params.get("rsi_period", 14),
                           oversold=tech_params.get("rsi_oversold", 30),
                           overbought=tech_params.get("rsi_overbought", 70)),
        "MACD": MACDStrategy(),
        "BB": BollingerStrategy(period=tech_params.get("bb_period", 20),
                                 std=tech_params.get("bb_std", 2)),
    }

    # 选择策略
    if args.strategy and args.strategy in strategies:
        selected = {args.strategy: strategies[args.strategy]}
    elif args.strategy == "ensemble":
        selected = {"Ensemble": EnsembleStrategy(
            strategies=list(strategies.values()),
            weights=[0.3, 0.3, 0.2, 0.2],
        )}
    else:
        selected = strategies  # 全量运行

    # 执行回测
    initial_cash = strategy_cfg.get("default", {}).get("initial_cash", 100000)
    commission = strategy_cfg.get("default", {}).get("commission", 0.001)
    slippage = strategy_cfg.get("default", {}).get("slippage", 0.0005)

    results = {}
    for name, strat in selected.items():
        logger.info(f"\n>>> 运行策略: {name}")
        bt = SimpleBacktester(initial_cash, commission, slippage)
        sig_df = strat._compute_signals(df)
        sig_df = sig_df.reindex(df.index)  # 对齐
        result = bt.run(df, sig_df, ticker)
        results[name] = result

        m = result["metrics"]
        logger.info(bt.summary())
        logger.info(f"  夏普: {m.get('sharpe_ratio', 'N/A')} | "
                   f"最大回撤: {m.get('max_drawdown', 'N/A')}% | "
                   f"胜率: {m.get('win_rate', 'N/A')}%")

    # 生成报告
    report = BacktestReport(output_dir="output/")
    for name, result in results.items():
        report.generate(
            equity_curve=result["equity_curve"],
            trades=result["trades"],
            signals=[],  # 简化
            metrics=result["metrics"],
            strategy_name=name,
            ticker=ticker,
        )

    logger.info("\n[OK] 回测完成，报告已保存至 output/ 目录")
    return results


# ============================================================
# 模式2：模拟交易
# ============================================================

def run_paper_trading(args, config: dict):
    """运行模拟交易"""
    logger = logging.getLogger("paper")
    logger.info("=" * 50)
    logger.info("启动模拟交易模式")
    logger.info("=" * 50)

    ticker = args.ticker or config.get("strategy", {}).get("default", {}).get("symbol", "AAPL")
    strategy_cfg = config.get("strategy", {})

    # 初始化组件
    storage = SQLiteStorage()
    trader = PaperTrader(
        initial_cash=strategy_cfg.get("default", {}).get("initial_cash", 100000),
        commission=strategy_cfg.get("default", {}).get("commission", 0.001),
        slippage=strategy_cfg.get("default", {}).get("slippage", 0.0005),
        storage=storage,
    )
    risk = RiskManager(
        initial_cash=strategy_cfg.get("default", {}).get("initial_cash", 100000),
        storage=storage,
    )
    fetcher = YFinanceFetcher()

    # 策略
    strategy = RSIStrategy()  # 默认 RSI 策略

    # 获取最新数据
    df = fetcher.download_history(ticker, period="3mo", interval="1d")
    if df.empty:
        logger.error("无法获取数据")
        return

    # 生成信号
    signal = strategy.generate(df, ticker)
    logger.info(f"[Paper] 当前信号: {signal}")

    # 获取当前价格
    quote = fetcher.get_quote(ticker)
    price = quote.get("last_price", df["Close"].iloc[-1])
    logger.info(f"[Paper] 当前价格: ${price}")

    # 执行信号
    if signal.action == "BUY" and signal.is_actionable():
        allowed, reason = risk.check_buy(ticker, price, 1, trader.cash, trader.get_all_positions())
        if allowed:
            qty = risk.max_position_size(ticker, price, "BUY", trader.cash)
            if qty > 0:
                result = trader.buy(ticker, price, qty, signal=str(signal), confidence=signal.confidence)
                logger.info(f"[Paper] 买入结果: {result}")
        else:
            logger.info(f"[Paper] 买入被风控拦截: {reason}")

    elif signal.action == "SELL" and signal.is_actionable():
        pos = trader.get_position(ticker)
        if pos:
            result = trader.sell(ticker, price, qty=None, signal=str(signal), confidence=signal.confidence)
            logger.info(f"[Paper] 卖出结果: {result}")

    # 打印账户状态
    prices = {ticker: price}
    snapshot = trader.get_equity_snapshot(prices)
    logger.info(f"[Paper] 账户快照: {snapshot}")

    logger.info("\n[OK] 模拟交易完成")


# ============================================================
# 模式3：参数优化
# ============================================================

def run_optimize(args, config: dict):
    """参数优化（网格搜索）"""
    logger = logging.getLogger("optimize")
    logger.info("=" * 50)
    logger.info("启动参数优化模式")
    logger.info("=" * 50)

    ticker = args.ticker or "AAPL"
    fetcher = YFinanceFetcher()
    df = fetcher.download_history(ticker, period="5y", interval="1d")
    df = df["2023-01-01":"2024-12-31"]

    if df.empty:
        logger.error("无数据")
        return

    # RSI 参数网格
    results = []
    for period in [7, 14, 21]:
        for oversold in [20, 25, 30]:
            for overbought in [70, 75, 80]:
                strat = RSIStrategy(period=period, oversold=oversold, overbought=overbought)
                bt = SimpleBacktester(100000)
                sig = strat._compute_signals(df)
                sig = sig.reindex(df.index)
                result = bt.run(df, sig, ticker)
                m = result["metrics"]
                results.append({
                    "period": period,
                    "oversold": oversold,
                    "overbought": overbought,
                    "sharpe": m.get("sharpe_ratio", 0),
                    "total_return": m.get("total_return", 0),
                    "max_dd": m.get("max_drawdown", 0),
                    "win_rate": m.get("win_rate", 0),
                })

    df_results = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(df_results.to_string(index=False))

    # 保存结果
    Path("output").mkdir(exist_ok=True)
    df_results.to_csv("output/rsi_optimization.csv", index=False)
    logger.info("参数优化结果已保存: output/rsi_optimization.csv")
    return df_results


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="美股量化交易工具")
    parser.add_argument("mode", choices=["backtest", "paper", "live", "optimize"],
                        help="运行模式")
    parser.add_argument("--ticker", "-t", default="AAPL", help="交易标的")
    parser.add_argument("--strategy", "-s", default=None,
                        choices=["MA", "RSI", "MACD", "BB", "ensemble"],
                        help="策略名称")
    parser.add_argument("--start", default=None, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")

    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    if args.mode == "backtest":
        run_backtest(args, config)
    elif args.mode == "paper":
        run_paper_trading(args, config)
    elif args.mode == "optimize":
        run_optimize(args, config)
    elif args.mode == "live":
        print("[WARN]️ 实盘模式需要先配置 config.yaml 中的 moomoo.api_key 和 moomoo.app_secret")
        print("并确保 MOOMOO OpenAPI 客户端已安装（pip install moomoo-api）")


if __name__ == "__main__":
    main()
