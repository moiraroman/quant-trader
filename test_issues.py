import sys
from pathlib import Path
BASE_DIR = Path('C:/Users/MAOUOTU/.qclaw/workspace-agent-4c3d0311/quant_trader')
sys.path.insert(0, str(BASE_DIR))

log_path = BASE_DIR / 'test_issues.log'
with open(log_path, 'w', encoding='utf-8') as log:
    
    # Issue 1: Check if 'analysis' key exists in report.market_state
    log.write('=== Issue 1: report.market_state analysis key ===\n')
    from ai.orchestrator import AIQuantAnalyst
    import yaml
    config_path = BASE_DIR / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    analyst = AIQuantAnalyst(config=config)
    report = analyst.run_daily_analysis(watchlist=['AAPL'], generate_report=False)
    
    ms = report.market_state
    log.write(f'market_state type: {type(ms)}\n')
    if isinstance(ms, dict):
        has_analysis = 'analysis' in ms
        log.write(f'Has analysis key: {has_analysis}\n')
        analysis_val = ms.get('analysis', 'N/A')
        log.write(f'analysis value: {analysis_val[:50]}...\n')
    else:
        log.write(f'market_state is not dict\n')
    
    # Issue 2: Check report attributes accessed in UI
    log.write('\n=== Issue 2: report attributes ===\n')
    log.write(f'Has top_opportunities: {hasattr(report, "top_opportunities")}\n')
    log.write(f'top_opportunities type: {type(report.top_opportunities)}\n')
    if report.top_opportunities:
        opp = report.top_opportunities[0]
        log.write(f'opp type: {type(opp)}\n')
        if isinstance(opp, dict):
            log.write(f'opp keys: {list(opp.keys())}\n')
    
    log.write(f'Has strategy_recommendations: {hasattr(report, "strategy_recommendations")}\n')
    if report.strategy_recommendations:
        rec = report.strategy_recommendations[0]
        log.write(f'rec type: {type(rec)}\n')
        if isinstance(rec, dict):
            log.write(f'rec keys: {list(rec.keys())}\n')
    
    log.write(f'Has risk_alerts: {hasattr(report, "risk_alerts")}\n')
    log.write(f'Has ai_summary: {hasattr(report, "ai_summary")}\n')
    log.write(f'ai_summary: {report.ai_summary}\n')
    
    # Issue 3: Check backtest tab
    log.write('\n=== Issue 3: Backtest engine ===\n')
    from data.fetcher import YFinanceFetcher
    from backtest.engine import SimpleBacktester
    fetcher = YFinanceFetcher()
    df = fetcher.download_history('AAPL', period='3mo', interval='1d')
    log.write(f'Data shape: {df.shape}\n')
    
    from strategy.technical import MAStrategy
    ma = MAStrategy(short_window=10, long_window=30)
    sig_df = ma._compute_signals(df)
    log.write(f'Signals columns: {list(sig_df.columns)}\n')
    
    engine = SimpleBacktester(initial_cash=100000, commission=0.001)
    result = engine.run(df, sig_df, ticker='AAPL')
    log.write(f'Result keys: {list(result.keys()) if isinstance(result, dict) else "N/A"}\n')
    if isinstance(result, dict):
        has_eq = 'equity_curve' in result
        has_trades = 'trades' in result
        log.write(f'Has equity_curve: {has_eq}\n')
        log.write(f'Has trades: {has_trades}\n')
    
    # Issue 4: Check optimize tab
    log.write('\n=== Issue 4: Optimizer ===\n')
    from optimization.optimizer import StrategyOptimizer
    opt = StrategyOptimizer(data_fetcher=fetcher, backtester=engine, scoring='sharpe')
    result = opt.optimize(strategy_name='MA', ticker='AAPL', method='random', n_trials=5)
    log.write(f'Optimize result type: {type(result)}\n')
    if result:
        log.write(f'Has best_params: {hasattr(result, "best_params")}\n')
        log.write(f'best_score: {result.best_score}\n')
    
    # Issue 5: Check settings tab
    log.write('\n=== Issue 5: Settings config ===\n')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log.write(f'Config keys: {list(config.keys())}\n')
        log.write(f'backtest: {config.get("backtest", {})}\n')
        log.write(f'risk: {config.get("risk", {})}\n')
    else:
        log.write('config.yaml not found!\n')
    
    log.write('\n=== ALL CHECKS COMPLETED ===\n')

print('Test completed')
