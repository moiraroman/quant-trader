# AI-Quant — AI-Powered Quantitative Trading System

> A professional-grade quantitative trading platform for US stocks, powered by AI multi-factor analysis, comprehensive backtesting, and real-time market intelligence.

---

## 🇺🇸 English

### Overview

AI-Quant is an advanced quantitative trading system designed for US stock markets. It combines traditional technical analysis with modern AI-driven market intelligence to provide traders with a comprehensive decision-making framework.

### Key Features

#### 1. Multi-Strategy Backtesting Engine
- **Technical Strategies**: MA Crossover, RSI, MACD, Bollinger Bands
- **Composite Strategy**: Multi-factor voting system with ATR-based risk management
- **AI Strategy**: LightGBM machine learning signal prediction
- **Ensemble Strategy**: Weighted synthesis of multiple strategies
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, profit factor
- **Visualization**: Plotly charts with price signals, equity curves, and volume analysis

#### 2. AI-Powered Market Analysis
- **Macro Scanner**: 7-module weighted regime scoring
  - Equity indices (25%): SPY, QQQ, DIA, IWM, SOXX
  - Market breadth (20%): RSP vs SPY, A/D line
  - VIX analysis (20%): Volatility regime, term structure
  - Credit spreads (15%): HYG/LQD ratio, yield curve
  - Safe haven flows (10%): TLT, GLD, UUP, FXY
  - Dollar index (5%): DXY trend analysis
  - Bitcoin (5%): Crypto market sentiment
- **Sector Rotation**: 11 GICS sector ETFs analysis
- **Market Breadth**: Advancing/declining issues, McClellan Oscillator
- **Liquidity Analysis**: Fed rates, yield curve, credit spreads

#### 3. Paper Trading Simulation
- Real-time signal monitoring with configurable intervals
- Virtual portfolio tracking with P&L calculation
- Risk management integration (position limits, daily loss limits)
- Trade history and performance analytics

#### 4. Live Trading (MOOMOO Integration)
- Direct broker API integration via MOOMOO OpenD
- Paper trading mode for strategy validation
- Real order execution with risk controls
- Account balance and position tracking

#### 5. Risk Management System
- ATR-based dynamic stop-loss and take-profit
- Hard stop-loss/take-profit percentages
- Daily loss circuit breaker
- Consecutive loss limits
- Position size limits

#### 6. Strategy Optimization
- Grid Search: Exhaustive parameter exploration
- Random Search: Efficient large-space sampling
- Bayesian Optimization: Intelligent parameter tuning
- Optimization targets: Sharpe ratio, total return, max drawdown

#### 7. Multi-Language Support
- 🇨🇳 Chinese (简体中文)
- 🇺🇸 English
- 🇯🇵 Japanese (日本語)
- Full UI internationalization with locale files

### Architecture

```
quant_trader/
├── config.yaml              # Global configuration
├── requirements.txt         # Python dependencies
├── main.py                  # CLI entry point
├── dashboard/
│   └── app.py               # Streamlit web dashboard
├── strategy/
│   ├── base.py              # Base strategy class
│   ├── technical.py         # Technical indicator strategies
│   ├── composite.py         # Multi-factor composite strategy
│   └── ai_model.py          # LightGBM ML model
├── backtest/
│   └── engine.py            # Backtesting engine
├── trading/
│   ├── paper.py             # Paper trading executor
│   ├── live.py              # Live trading (MOOMOO)
│   └── bot.py               # Automated trading bot
├── data/
│   ├── fetcher.py           # yfinance data fetcher
│   └── storage.py           # Parquet + SQLite storage
├── risk/
│   └── manager.py           # Risk management
├── ai/
│   ├── macro_scanner.py     # Macro regime scanner
│   ├── sector_rotation.py   # Sector rotation analysis
│   ├── market_breadth.py    # Market breadth analyzer
│   ├── liquidity_analyzer.py # Liquidity analysis
│   ├── market_analyzer.py   # Market state analysis
│   ├── stock_screener.py    # 6-factor stock screener
│   └── orchestrator.py      # AI analysis orchestrator
├── optimization/
│   └── optimizer.py         # Parameter optimizer
├── notification/
│   └── notifier.py          # Email/Telegram/WeChat alerts
├── locales/
│   ├── zh.json              # Chinese translations
│   ├── en.json              # English translations
│   └── ja.json              # Japanese translations
└── docs/                    # Documentation
```

### Quick Start

#### Installation

```bash
# Clone repository
git clone https://github.com/moiraroman/quant-trader.git
cd quant-trader

# Install dependencies
pip install -r requirements.txt
```

#### Run Backtest

```bash
# Single strategy
python main.py backtest -t AAPL -s RSI

# Composite strategy
python main.py backtest -t AAPL -s composite

# Ensemble strategy
python main.py backtest -t AAPL -s ensemble
```

#### Launch Dashboard

```bash
streamlit run dashboard/app.py
# Opens: http://localhost:8501
```

#### Paper Trading

```bash
python main.py paper -t AAPL,MSFT,GOOGL
```

### Configuration

Edit `config.yaml`:

```yaml
strategy:
  default:
    symbol: "AAPL"
    initial_cash: 100000
    commission: 0.001
    slippage: 0.0005

risk:
  max_position_pct: 0.2
  max_total_position_pct: 0.8
  daily_loss_stop_pct: 0.10
  atr_stop_mult: 2.0
  atr_take_profit_mult: 8.0

data:
  moomoo:
    enable: true
    api_key: "YOUR_API_KEY"
    app_secret: "YOUR_APP_SECRET"
    paper_trade: true
```

### Tech Stack

- **Data**: yfinance, pandas, numpy
- **Indicators**: pandas_ta, talib
- **ML**: LightGBM, scikit-learn
- **Visualization**: Plotly, Streamlit
- **Storage**: Parquet, SQLite
- **Broker API**: MOOMOO OpenAPI
- **Notifications**: smtplib, python-telegram-bot

---

## 🇯🇵 日本語

### 概要

AI-Quantは、米国株式市場向けの高度な定量的取引システムです。従来のテクニカル分析と現代のAI駆動型市場インテリジェンスを組み合わせ、トレーダーに包括的な意思決定フレームワークを提供します。

### 主な機能

#### 1. マルチストラテジーバックテストエンジン
- **テクニカルストラテジー**: 移動平均クロス、RSI、MACD、ボリンジャーバンド
- **複合ストラテジー**: ATRベースのリスク管理を備えた多因子投票システム
- **AIストラテジー**: LightGBM機械学習シグナル予測
- **アンサンブルストラテジー**: 複数ストラテジーの加重合成
- **パフォーマンス指標**: シャープレシオ、最大ドローダウン、勝率、プロフィットファクター
- **可視化**: 価格シグナル、資産曲線、出来高分析のPlotlyチャート

#### 2. AI搭載市場分析
- **マクロスキャナー**: 7モジュール加重レジームスコアリング
  - 株式指数（25%）: SPY、QQQ、DIA、IWM、SOXX
  - 市場広度（20%）: RSP対SPY、A/Dライン
  - VIX分析（20%）: ボラティリティレジーム、期限構造
  - クレジットスプレッド（15%）: HYG/LQD比率、イールドカーブ
  - 安全資産フロー（10%）: TLT、GLD、UUP、FXY
  - ドル指数（5%）: DXYトレンド分析
  - ビットコイン（5%）: 暗号市場センチメント
- **セクターローテーション**: 11 GICSセクターETF分析
- **市場広度**: 騰落株数、マクレランオシレーター
- **流動性分析**: FRB金利、イールドカーブ、クレジットスプレッド

#### 3. ペーパートレードシミュレーション
- 設定可能な間隔でのリアルタイムシグナル監視
- 損益計算付き仮想ポートフォリオ追跡
- リスク管理統合（ポジション制限、日次損失制限）
- 取引履歴とパフォーマンス分析

#### 4. 実際の取引（MOOMOO統合）
- MOOMOO OpenD経由の直接ブローカーAPI統合
- ストラテジー検証用ペーパートレードモード
- リスク管理付き実注文執行
- 残高とポジション追跡

#### 5. リスク管理システム
- ATRベースの動的損切りと利確
- 固定損切り/利確パーセンテージ
- 日次損失サーキットブレーカー
- 連続損失制限
- ポジションサイズ制限

#### 6. ストラテジー最適化
- グリッドサーチ: 網羅的パラメータ探索
- ランダムサーチ: 効率的な大空間サンプリング
- ベイズ最適化: 知的パラメータ調整
- 最適化目標: シャープレシオ、総収益率、最大ドローダウン

#### 7. 多言語サポート
- 🇨🇳 中国語（简体中文）
- 🇺🇸 英語
- 🇯🇵 日本語
- ロケールファイルによる完全なUI国際化

### クイックスタート

#### インストール

```bash
# リポジトリをクローン
git clone https://github.com/moiraroman/quant-trader.git
cd quant-trader

# 依存関係をインストール
pip install -r requirements.txt
```

#### バックテスト実行

```bash
# 単一ストラテジー
python main.py backtest -t AAPL -s RSI

# 複合ストラテジー
python main.py backtest -t AAPL -s composite

# アンサンブルストラテジー
python main.py backtest -t AAPL -s ensemble
```

#### ダッシュボード起動

```bash
streamlit run dashboard/app.py
# 開く: http://localhost:8501
```

#### ペーパートレード

```bash
python main.py paper -t AAPL,MSFT,GOOGL
```

### 設定

`config.yaml`を編集:

```yaml
strategy:
  default:
    symbol: "AAPL"
    initial_cash: 100000
    commission: 0.001
    slippage: 0.0005

risk:
  max_position_pct: 0.2
  max_total_position_pct: 0.8
  daily_loss_stop_pct: 0.10
  atr_stop_mult: 2.0
  atr_take_profit_mult: 8.0

data:
  moomoo:
    enable: true
    api_key: "YOUR_API_KEY"
    app_secret: "YOUR_APP_SECRET"
    paper_trade: true
```

---

## 🇨🇳 中文

### 概述

AI-Quant 是一个专为美股市场设计的先进量化交易系统。它结合了传统技术分析与现代 AI 驱动的市场智能，为交易者提供全面的决策框架。

### 核心功能

#### 1. 多策略回测引擎
- **技术策略**: 均线交叉、RSI、MACD、布林带
- **复合策略**: 多因子投票系统，ATR 动态止盈止损
- **AI 策略**: LightGBM 机器学习信号预测
- **集成策略**: 多策略加权合成
- **绩效指标**: 夏普比率、最大回撤、胜率、盈亏比
- **可视化**: Plotly 图表，含价格信号、资金曲线、成交量分析

#### 2. AI 智能市场分析
- **宏观扫描器**: 7 模块加权市场环境评分
  - 股票指数（25%）: SPY、QQQ、DIA、IWM、SOXX
  - 市场广度（20%）: RSP vs SPY、涨跌线
  - VIX 分析（20%）: 波动率状态、期限结构
  - 信用利差（15%）: HYG/LQD 比率、收益率曲线
  - 避险资金流向（10%）: TLT、GLD、UUP、FXY
  - 美元指数（5%）: DXY 趋势分析
  - 比特币（5%）: 加密货币市场情绪
- **板块轮动**: 11 大 GICS 板块 ETF 分析
- **市场广度**: 涨跌家数、麦克莱伦振荡器
- **流动性分析**: 美联储利率、收益率曲线、信用利差

#### 3. 模拟交易
- 可配置间隔的实时信号监控
- 虚拟投资组合跟踪与盈亏计算
- 风险管理集成（仓位限制、日亏损限制）
- 交易历史与绩效分析

#### 4. 实盘交易（MOOMOO 集成）
- 通过 MOOMOO OpenD 直连券商 API
- 模拟交易模式用于策略验证
- 带风险控制的实盘订单执行
- 账户余额与持仓跟踪

#### 5. 风险管理系统
- ATR 动态止盈止损
- 固定止盈止损百分比
- 日亏损熔断机制
- 连续亏损限制
- 持仓上限控制

#### 6. 策略优化
- 网格搜索: 穷举参数探索
- 随机搜索: 高效大空间采样
- 贝叶斯优化: 智能参数调优
- 优化目标: 夏普比率、总收益率、最大回撤

#### 7. 多语言支持
- 🇨🇳 中文（简体中文）
- 🇺🇸 英文
- 🇯🇵 日文
- 完整的 UI 国际化，支持 locale 文件

### 项目架构

```
quant_trader/
├── config.yaml              # 全局配置
├── requirements.txt         # Python 依赖
├── main.py                  # CLI 入口
├── dashboard/
│   └── app.py               # Streamlit Web 仪表板
├── strategy/
│   ├── base.py              # 策略基类
│   ├── technical.py         # 技术指标策略
│   ├── composite.py         # 多因子复合策略
│   └── ai_model.py          # LightGBM 机器学习模型
├── backtest/
│   └── engine.py            # 回测引擎
├── trading/
│   ├── paper.py             # 模拟交易执行器
│   ├── live.py              # 实盘交易（MOOMOO）
│   └── bot.py               # 自动交易机器人
├── data/
│   ├── fetcher.py           # yfinance 数据获取
│   └── storage.py           # Parquet + SQLite 存储
├── risk/
│   └── manager.py           # 风险管理
├── ai/
│   ├── macro_scanner.py     # 宏观环境扫描器
│   ├── sector_rotation.py   # 板块轮动分析
│   ├── market_breadth.py    # 市场广度分析器
│   ├── liquidity_analyzer.py # 流动性分析
│   ├── market_analyzer.py   # 市场状态分析
│   ├── stock_screener.py    # 6 因子选股器
│   └── orchestrator.py      # AI 分析编排器
├── optimization/
│   └── optimizer.py         # 参数优化器
├── notification/
│   └── notifier.py          # 邮件/Telegram/企业微信告警
├── locales/
│   ├── zh.json              # 中文翻译
│   ├── en.json              # 英文翻译
│   └── ja.json              # 日文翻译
└── docs/                    # 文档
```

### 快速开始

#### 安装

```bash
# 克隆仓库
git clone https://github.com/moiraroman/quant-trader.git
cd quant-trader

# 安装依赖
pip install -r requirements.txt
```

#### 运行回测

```bash
# 单一策略
python main.py backtest -t AAPL -s RSI

# 复合策略
python main.py backtest -t AAPL -s composite

# 集成策略
python main.py backtest -t AAPL -s ensemble
```

#### 启动仪表板

```bash
streamlit run dashboard/app.py
# 打开: http://localhost:8501
```

#### 模拟交易

```bash
python main.py paper -t AAPL,MSFT,GOOGL
```

### 配置说明

编辑 `config.yaml`:

```yaml
strategy:
  default:
    symbol: "AAPL"
    initial_cash: 100000      # 初始资金（美元）
    commission: 0.001         # 手续费率 0.1%
    slippage: 0.0005          # 滑点 0.05%

risk:
  max_position_pct: 0.2       # 单票最大仓位 20%
  max_total_position_pct: 0.8 # 总仓位上限 80%
  daily_loss_stop_pct: 0.10   # 日亏损熔断 10%
  atr_stop_mult: 2.0          # ATR 止损倍数
  atr_take_profit_mult: 8.0   # ATR 止盈倍数

data:
  moomoo:
    enable: true
    api_key: "YOUR_API_KEY"
    app_secret: "YOUR_APP_SECRET"
    paper_trade: true         # true=模拟盘, false=实盘
```

### 技术栈

- **数据**: yfinance, pandas, numpy
- **指标**: pandas_ta, talib
- **机器学习**: LightGBM, scikit-learn
- **可视化**: Plotly, Streamlit
- **存储**: Parquet, SQLite
- **券商 API**: MOOMOO OpenAPI
- **通知**: smtplib, python-telegram-bot

---

## ⚠️ Disclaimer / 免責事項 / 免责声明

**English**: This tool is for research and educational purposes only. Live trading involves real capital — all profits and losses are entirely your own responsibility. Always validate strategies with thorough backtesting before going live.

**日本語**: このツールは研究・教育目的のみです。実際の取引は実資金を伴います — すべての損益は自己責任です。実取引前に徹底的なバックテストでストラテジーを検証してください。

**中文**: 本工具仅供研究和学习使用。实盘交易涉及真实资金 — 所有盈亏由您自行承担。实盘前请务必通过充分回测验证策略。

---

## 📄 License

MIT License — Open Source
