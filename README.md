# Quant Trader — AI-Powered US Stock Trading System

> A comprehensive quantitative trading platform for US stocks, featuring backtesting, paper trading, live execution, and AI-driven market analysis.

---

## 🚀 Features

### Core Trading Workflow
- **Backtesting** — Test strategies with full historical data (yfinance), performance metrics, equity curves, and HTML reports
- **Paper Trading** — Live simulation with real market data, zero risk
- **Live Trading** — Real execution via MOOMOO OpenAPI

### Strategies
| Strategy | Code | Description |
|---|---|---|
| Moving Average Crossover | `MAStrategy` | Short/long MA golden/dead cross |
| RSI | `RSIStrategy` | RSI overbought/oversold |
| MACD | `MACDStrategy` | MACD histogram + signal cross |
| Bollinger Bands | `BollingerStrategy` | Price touches upper/lower BB bands |
| **Composite** | `CompositeStrategy` | 5-entry → pick 3, ATR stop/take profit, RSI exit |
| **AI LightGBM** | `AIStrategy` | ML signal prediction (classification) |
| Ensemble | `EnsembleStrategy` | Multi-strategy weighted synthesis |

### AI Analysis Module
- **Macro Scanner** — 7-module weighted regime scoring:
  - Equity (25%), Breadth (20%), VIX (20%), Credit (15%), Safe Haven (10%), DXY (5%), BTC (5%)
  - Sigmoid probability model → Risk-On / Neutral / Risk-Off
  - Conflict detection with confidence scoring

### Dashboard (Streamlit)
- 6 tabs: Backtest / Paper / Live / AI Analysis / Optimization / Settings
- 3 languages: 🇨🇳 中文 / 🇺🇸 English / 🇯🇵 日本語
- Plotly charts, parameter tuning, real-time results

---

## 📁 Project Structure

```
quant_trader/
├── config.yaml              # Global config (parameters, API keys, risk rules)
├── requirements.txt         # Python dependencies
├── main.py                  # CLI entry point
├── dashboard/
│   └── app.py               # Streamlit Web Dashboard
├── strategy/
│   ├── base.py              # BaseStrategy abstract class, Signal data class
│   ├── technical.py        # Technical indicator strategies
│   ├── composite.py        # Composite multi-factor strategy
│   └── ai_model.py         # LightGBM AI signal model
├── backtest/
│   └── engine.py           # Backtesting engine + performance reporting
├── trading/
│   ├── paper.py            # Paper trading executor
│   └── live.py             # Live trading (MOOMOO OpenAPI)
├── data/
│   ├── fetcher.py          # yfinance data fetching
│   └── storage.py          # Parquet + SQLite storage
├── risk/
│   └── manager.py          # Risk management module
├── ai/
│   └── macro_scanner.py    # Macro market regime scanner
├── optimization/
│   └── parameter_search.py # Parameter grid search optimizer
├── notification/
│   └── telegram_bot.py     # Telegram alert bot
├── locales/
│   ├── zh.json             # Chinese translations
│   ├── en.json             # English translations
│   └── ja.json             # Japanese translations
└── docs/
    └── quant_trader_design_doc.md  # Product design document
```

---

## 🛠 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 Windows note: pandas_ta is used instead of TA-Lib (no extra DLLs needed).

### 2. Run Backtest

```bash
# Single strategy
python main.py backtest -t AAPL -s RSI

# MA crossover
python main.py backtest -t AAPL -s MA

# Multi-strategy comparison
python main.py backtest -t AAPL -s ensemble
```

### 3. Web Dashboard

```bash
streamlit run dashboard/app.py
# Opens: http://localhost:8501
```

### 4. Paper / Live Trading

```bash
python main.py paper -t AAPL
# python main.py live -t AAPL  # requires MOOMOO API keys
```

---

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
strategy:
  default:
    symbol: "AAPL"
    initial_cash: 100000   # USD
    commission: 0.001       # 0.1%
    slippage: 0.0005        # 0.05%

risk:
  max_position_pct: 0.2    # Max 20% per position
  max_total_position_pct: 0.8
  daily_loss_stop_pct: 0.10 # Daily loss circuit breaker

data:
  moomoo:
    enable: true
    api_key: "YOUR_API_KEY"
    app_secret: "YOUR_APP_SECRET"
    paper_trade: false     # Set true for paper, false for live
```

---

## 📊 Performance Metrics

After backtest, outputs include:

| Metric | Description |
|---|---|
| Total Return | Overall return over period |
| Annualized Return | Annualized performance |
| Sharpe Ratio | Risk-adjusted return (>1.0 is good) |
| Max Drawdown | Peak-to-trough maximum loss |
| Win Rate | % of profitable trades |
| Profit Factor | avg_win / avg_loss |

Reports auto-generated as HTML files in `output/`.

---

## 🔧 Tech Stack

- **Data**: yfinance, pandas
- **Indicators**: pandas_ta
- **ML**: LightGBM, scikit-learn
- **Backtesting**: Custom SimpleBacktester engine
- **Storage**: Parquet + SQLite
- **Dashboard**: Streamlit + Plotly
- **Brokers**: MOOMOO OpenAPI
- **Notifications**: Telegram Bot API

---

## ⚠️ Disclaimer

This tool is for **research and learning purposes only**. Live trading involves real capital — profit and loss are entirely your own responsibility. Always validate strategies with thorough backtesting before going live.