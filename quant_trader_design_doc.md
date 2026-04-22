# quant_trader 项目设计书

> **版本**：v1.1  
> **日期**：2026-04-22  
> **用途**：项目交接文档，供接收人快速理解系统架构、模块职责与数据流转

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构总览](#2-系统架构总览)
3. [目录结构说明](#3-目录结构说明)
4. [核心模块详解](#4-核心模块详解)
   - 4.1 数据层（data/）
   - 4.2 策略层（strategy/）
   - 4.3 回测引擎（backtest/）
   - 4.4 风险管理（risk/）
   - 4.5 交易执行（trading/）
   - 4.6 AI 分析层（ai/）
   - 4.7 参数优化（optimization/）
   - 4.8 通知推送（notification/）
   - 4.9 Web 仪表盘（dashboard/）
   - 4.10 国际化（locale/）
5. [数据流图](#5-数据流图)
6. [配置文件说明](#6-配置文件说明)
7. [使用方式](#7-使用方式)
8. [技术栈汇总](#8-技术栈汇总)
9. [已验证回测结果](#9-已验证回测结果)
10. [待改进事项与注意事项](#10-待改进事项与注意事项)

---

## 1. 项目概述

**quant_trader** 是一套面向**美股市场**的 Python 量化交易系统，具备以下核心能力：

| 能力 | 说明 |
|------|------|
| 多策略信号生成 | MA / RSI / MACD / Bollinger / AI / 复合策略 |
| 历史数据回测 | 逐日 Bar 模拟，完整绩效指标 |
| 模拟盘交易 | 本地 PaperTrader + 富途 MOOMOO 沙盒 |
| 实盘对接 | 富途 MOOMOO OpenAPI（框架已就绪） |
| AI 智能分析 | 市场状态判断 / 选股 / 宏观风险扫描 |
| 参数优化 | 网格 / 随机 / 贝叶斯（Optuna） |
| 多渠道通知 | Email / SMS / Telegram / 企业微信 |
| Web 可视化 | Streamlit 仪表盘，支持中 / 英 / 日三语 |

---

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Dashboard (Streamlit)                │
│          Backtest │ PaperTrade │ Live │ AI │ Optimize │ Settings│
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
   │  AI 分析层  │  │  策略信号层  │  │   参数优化层  │
   │ orchestrator│  │ technical /  │  │  grid/random/ │
   │ market_anal │  │ ai_model /   │  │  bayesian     │
   │ stock_screen│  │ composite    │  └───────────────┘
   │ macro_scan  │  └──────┬───────┘
   └─────────────┘         │
                    ┌──────▼───────┐
                    │   回测引擎   │
                    │ SimpleBack-  │
                    │   tester     │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌─────────────┐  ┌───────────────┐
   │  风险管理   │  │  交易执行   │  │  数据存储     │
   │ RiskManager │  │ PaperTrader │  │ ParquetStorage│
   │             │  │ MooMooLive  │  │ SQLiteStorage │
   └─────────────┘  └─────────────┘  └───────────────┘
                                              ▲
                                     ┌────────┴───────┐
                                     │   数据获取     │
                                     │ YFinanceFetcher│
                                     └────────────────┘
```

### 运行模式

| 模式 | CLI 参数 | 说明 |
|------|----------|------|
| 回测 | `backtest` | 历史数据策略回测 + HTML 报告 |
| 模拟盘 | `paper` | 启动守护线程，定时信号检查 + 模拟下单 |
| 参数优化 | `optimize` | 搜索最优策略参数 |
| 实盘 | `live` | 富途 OpenAPI 实盘（框架就绪，需配置 API Key）|

---

## 3. 目录结构说明

```
quant_trader/
├── main.py                  # CLI 入口，4 种运行模式
├── config.yaml              # 统一配置文件（所有模块参数）
├── requirements.txt         # Python 依赖
│
├── data/
│   ├── fetcher.py           # yfinance 数据下载（OHLCV + 财务 + 新闻）
│   └── storage.py           # Parquet K 线缓存 + SQLite 交易记录
│
├── strategy/
│   ├── base.py              # Signal 数据类 + BaseStrategy 抽象类
│   ├── technical.py         # MA / RSI / MACD / Bollinger / Ensemble 策略
│   ├── ai_model.py          # LightGBM AI 策略（30+ 特征，自动重训练）
│   └── composite.py         # 复合策略（5 条件 3 满足，ATR 止盈止损）
│
├── backtest/
│   └── engine.py            # 回测引擎 + 绩效指标 + Plotly HTML 报告
│
├── risk/
│   └── manager.py           # 多层风控（仓位/ATR止损/日亏熔断/连亏限制）
│
├── trading/
│   ├── paper.py             # PaperTrader 本地模拟 + MooMoo 沙盒占位
│   ├── live.py              # MooMooLiveTrader 实盘框架
│   └── bot.py               # PaperTradingBot 守护线程
│
├── ai/
│   ├── orchestrator.py      # AIQuantAnalyst 总协调器
│   ├── market_analyzer.py   # 市场状态分析（趋势/波动/动量/风险）
│   ├── stock_screener.py    # 6 因子选股模型
│   └── macro_scanner.py     # 宏观全景扫描（7模块加权/概率引擎/冲突检测）
│
├── optimization/
│   └── optimizer.py         # 网格/随机/贝叶斯参数优化
│
├── notification/
│   └── notifier.py          # Email/SMS/Telegram/WeChat 多渠道推送
│
├── dashboard/
│   └── app.py               # Streamlit Web 仪表盘（6 模块，三语支持）
│
└── locale/
    ├── zh.json              # 中文翻译
    ├── en.json              # 英文翻译
    └── ja.json              # 日文翻译
```

---

## 4. 核心模块详解

### 4.1 数据层（data/）

#### `fetcher.py` — `YFinanceFetcher`

**职责**：从 Yahoo Finance 获取所有市场数据，无需 API Key。

| 方法 | 功能 |
|------|------|
| `fetch_ohlcv(ticker, period, interval)` | 下载 K 线数据，返回 DataFrame |
| `fetch_realtime_quote(ticker)` | 获取实时报价（价格/量/PE/市值等） |
| `fetch_financial_data(ticker)` | 获取财务数据（资产负债表/利润表/现金流） |
| `fetch_news(ticker, limit)` | 获取最新新闻（标题/链接/时间） |
| `fetch_analyst_info(ticker)` | 获取分析师评级和目标价 |

#### `storage.py` — `ParquetStorage` + `SQLiteStorage`

**`ParquetStorage`**：K 线数据本地缓存

- 存储路径：`data/cache/{ticker}_{interval}.parquet`
- 增量追加，自动去重
- 避免重复下载历史数据

**`SQLiteStorage`**：交易记录持久化

| 表名 | 字段 | 用途 |
|------|------|------|
| `trades` | timestamp/ticker/action/shares/price/value/pnl | 交易流水 |
| `positions` | ticker/shares/avg_cost/current_price/pnl_pct | 持仓记录 |
| `signals` | timestamp/ticker/action/strength/confidence/reason | 信号日志 |
| `equity_curve` | timestamp/equity/cash/positions_value | 权益曲线 |

---

### 4.2 策略层（strategy/）

#### `base.py` — 基类与信号结构

```python
@dataclass
class Signal:
    ticker: str
    action: str          # BUY / SELL / HOLD
    strength: float      # 信号强度 0~1
    confidence: float    # 置信度 0~1
    reason: str          # 信号原因说明
    price: float         # 触发价格
```

`BaseStrategy` 要求子类实现 `_compute_signals(data)` 方法。

#### `technical.py` — 技术指标策略

| 策略类 | 逻辑 | 核心参数 |
|--------|------|----------|
| `MAStrategy` | 快线上穿慢线买入，下穿卖出 | fast/slow 周期 |
| `RSIStrategy` | RSI < 超卖阈值买，> 超买阈值卖 | period, oversold, overbought |
| `MACDStrategy` | MACD 金叉买，死叉卖 | fast/slow/signal 周期 |
| `BollingerStrategy` | 价格突破上轨卖，下轨买 | period, std_dev |
| `EnsembleStrategy` | 集成以上4策略，多数投票 | — |

#### `ai_model.py` — `AIStrategy`（LightGBM）

- **特征工程**：30+ 特征（价格变化率 / MA / RSI / MACD / Bollinger / ATR / 成交量 / 动量）
- **标签**：未来 5 日收益率 > 1% 为正样本
- **训练**：最少 252 根 K 线，自动重训练（可配置频率）
- **输出**：信号 + 置信度（模型预测概率）

#### `composite.py` — `CompositeStrategy`（复合策略）

**入场条件（满足 5 项中的 3 项即买入）**：

1. RSI 超卖（< 40）
2. SuperTrend 看涨
3. MACD 金叉
4. 动量向上（近期累积收益 > 0）
5. ADX > 25（趋势明确）

**出场条件**：
- ATR 止损 / ATR 止盈（可配置倍数）
- RSI 超买卖出
- MA 死叉卖出

---

### 4.3 回测引擎（backtest/）

#### `engine.py` — `SimpleBacktester`

**工作流程**：

```
加载历史数据 → 逐日迭代 → 策略信号 → 风控检查 → 执行交易
    → 更新持仓/现金 → 记录权益曲线 → 生成报告
```

**绩效指标（`compute_metrics`）**：

| 指标 | 说明 |
|------|------|
| 总收益率 | 相对初始资金的总回报 |
| 年化收益率 | 折算到年度的回报率 |
| 夏普比率 | 风险调整后收益（年化） |
| 最大回撤 | 峰值到谷底的最大跌幅 |
| 胜率 | 盈利交易占总交易数比例 |
| 盈亏比 | 平均盈利 / 平均亏损 |
| Calmar 比率 | 年化收益率 / 最大回撤 |
| Beta | 相对基准的系统性风险 |

**`BacktestReport`**：生成 Plotly 交互式 HTML 报告，包含：
- 权益曲线
- 买卖点标记
- 持仓分布
- 月度收益热力图

---

### 4.4 风险管理（risk/）

#### `manager.py` — `RiskManager`

**多层风控机制**：

| 层级 | 规则 | 默认值 |
|------|------|--------|
| 仓位限制 | 单股最大持仓比例 | 20% |
| ATR 动态止损 | 入场价 - ATR × 倍数 | 2倍 ATR |
| ATR 动态止盈 | 入场价 + ATR × 倍数 | 3倍 ATR |
| 硬止损 | 固定百分比止损 | -8% |
| 硬止盈 | 固定百分比止盈 | +15% |
| RSI 止盈 | RSI 超买时平仓 | RSI > 75 |
| 移动止损 | 跟踪最高价止损 | 最高价 - 5% |
| 日亏熔断 | 日亏损达阈值停止交易 | -2% |
| 连亏限制 | 连续亏损次数上限 | 5次 |

---

### 4.5 交易执行（trading/）

#### `paper.py` — `PaperTrader`

- 纯本地模拟，无需任何账户
- 均价成本法追踪持仓
- 自动记录到 `SQLiteStorage`
- 支持买入 / 卖出 / 查询持仓 / 获取权益

#### `live.py` — `MooMooLiveTrader`

- 富途 MOOMOO OpenAPI 集成框架
- 双重风控（系统风控 + 富途账户风控）
- 订单管理（下单 / 撤单 / 查询）
- ⚠️ 注意：`place_order` SDK 代码目前为注释占位符，上线前需按富途文档补全

#### `bot.py` — `PaperTradingBot`

- 守护线程，可配置检查间隔（默认 5 分钟）
- 定时执行：信号检查 → 止损/止盈检查 → 执行交易
- 支持启动 / 停止 / 状态查询

---

### 4.6 AI 分析层（ai/）

#### `orchestrator.py` — `AIQuantAnalyst`（总协调器）

**分析流程**：

```
1. MarketAnalyzer  → 判断当前市场状态
2. MacroScanner    → 宏观全景扫描（7模块加权+概率引擎）
3. StockScreener   → 多股票打分筛选
4. 策略推荐        → 根据市场状态建议使用哪种策略
5. 报告生成        → 汇总分析结论
6. LLM 摘要        → 可选，接入大语言模型生成自然语言报告
```

#### `market_analyzer.py` — `MarketAnalyzer`

| 分析维度 | 指标 | 输出 |
|----------|------|------|
| 趋势 | MA50 vs MA200，ADX | bullish/bearish/sideways |
| 波动率 | ATR% | low/medium/high/extreme |
| 动量 | RSI，MACD | strong/moderate/weak/oversold/overbought |
| 综合风险 | 综合评分 | low/medium/high/very_high |

#### `stock_screener.py` — `StockScreener`（6 因子模型）

| 因子 | 权重 | 计算方式 |
|------|------|----------|
| 动量 | 25% | 20日/60日收益率加权 |
| 趋势 | 20% | 价格 vs MA50 / MA200 |
| 成交量 | 15% | 量能放大倍数 |
| 波动率 | 15% | ATR%（越低得分越高） |
| RSI | 15% | 接近 50 得分最高 |
| MA 偏离 | 10% | 偏离均线程度 |

#### `macro_scanner.py` — `MacroScanner`（宏观全景扫描 v2）

**设计理念**：三层架构（规则引擎 → 打分引擎 → 概率引擎），输出 5 项完整结果。

**输出格式**：

| 项目 | 说明 |
|------|------|
| `regime` | Risk-On / Neutral / Risk-Off |
| `macro_env_score` | 1~10 综合环境评分 |
| `confidence` | high / medium / low 置信度 |
| `key_drivers` | 本次主导因子列表 |
| `warnings` | 冲突项 / 异常项列表 |

**数据源（5 大类 15+ 标的）**：

| 类别 | 标的 | 用途 |
|------|------|------|
| 权益指数 | SPY, QQQ, DIA, IWM, SOXX | 主趋势 / 宽度 / 风格轮动 |
| 避险资产 | TLT, GLD, UUP, FXY, BTC-USD | 资金流向判断 |
| 波动率 | ^VIX, ^VIX3M(可选) | 恐慌程度 / 期限结构 |
| 信用/流动性 | HYG, LQD, ^TNX, ^IRX | 信用偏好 / 收益率曲线 |
| 市场广度 | RSP vs SPY | 涨跌健康度 |

**7 个评分模块**：

| 模块 | 权重 | 核心逻辑 |
|------|------|----------|
| equity_score | 25% | 5 指数均线排列 + 动量（0.4\*5d + 0.3\*20d + 0.3\*60d）|
| breadth_score | 20% | RSP(等权) vs SPY(加权) 差异近似广度 |
| vix_score | 20% | VIX 绝对值分档 + 动量 + 期限结构 |
| credit_score | 15% | HYG/LQD 信用偏好 + 10Y-2Y 收益率曲线 |
| safe_haven_score | 10% | TLT(条件修正) + GLD + UUP/FXY 货币组 |
| dxy_score | 5% | 美元方向（UUP 代理）|
| btc_score | 5% | BTC 趋势 + 暴跌检测 |

**合成公式**：

```
每个模块标准化到 [-1, 1]
macro_raw = Σ(weight_i × module_score_norm_i)
macro_env_score = 5.5 + 4.5 × macro_raw  (clamp 1~10)
```

**三层分类**：

1. **规则引擎**：VIX>35 直接 Risk-Off; equity>1.5 且 vix>0 直接 Risk-On
2. **打分引擎**：score≥7 且 vix/equity 同向 → Risk-On; score≤4 且同向 → Risk-Off; 其余 Neutral
3. **概率引擎**：`P(Risk-On) = sigmoid(2\*equity + 1.5\*breadth - 2\*vix - credit)`

**冲突检测**：

```
conflict_score = 不一致信号数 / 总信号数
strength = avg(|module_score|)
confidence = (1 - conflict_score) × strength
≥0.75 → high, 0.50~0.75 → medium, <0.50 → low
```

**数据预处理**（每个标的统一计算）：

```
ret_5d, ret_20d, ret_60d  (多周期收益率)
ma20, ma50, ma200          (移动平均线)
vol20                      (20日波动率)
zscore_20                  (20日标准化得分)
```

---

### 4.7 参数优化（optimization/）

#### `optimizer.py` — `StrategyOptimizer`

**支持三种搜索方式**：

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| 网格搜索 | 穷举所有参数组合 | 参数空间小 |
| 随机搜索 | 随机采样 N 次 | 参数空间大，快速探索 |
| 贝叶斯优化 | Optuna 智能搜索 | 需要高精度最优解 |

**已预定义参数空间的策略**：MA、RSI、MACD、Bollinger、Composite

**优化目标**（可选）：夏普比率、总收益率、Calmar 比率

---

### 4.8 通知推送（notification/）

#### `notifier.py` — `NotificationManager`

**支持渠道**：

| 渠道 | 配置项 | 触发条件 |
|------|--------|----------|
| Email | smtp_server / port / user / password | 可配置 |
| SMS | Twilio account_sid / auth_token | 可配置 |
| Telegram | bot_token / chat_id | 可配置 |
| 企业微信 | webhook_url | 可配置 |

**默认触发场景**：买卖信号、止损触发、日报、错误报警

---

### 4.9 Web 仪表盘（dashboard/）

#### `app.py` — Streamlit 应用

**6 个功能模块**：

| 模块 | 功能 |
|------|------|
| 📊 Backtest | 选股票/策略/时间段，运行回测，查看报告 |
| 📈 Paper Trade | 查看持仓/权益曲线，手动下单，启停机器人 |
| 🔴 Live Trade | 富途实盘交易界面（配置 API 后可用） |
| 🤖 AI Analysis | 市场分析、选股、宏观扫描 |
| ⚙️ Optimize | 参数优化配置与结果展示 |
| 🔧 Settings | 配置修改（通知/风控/策略参数） |

**启动命令**：
```bash
streamlit run dashboard/app.py
```

---

### 4.10 国际化（locale/）

支持 **中文 / English / 日本語** 三语界面切换，翻译文件位于：
- `locale/zh.json`
- `locale/en.json`
- `locale/ja.json`

`I18nManager` 实现懒加载，在 Streamlit sidebar 可实时切换语言。

---

## 5. 数据流图

```
Yahoo Finance (yfinance)
        │
        ▼
  YFinanceFetcher
  ┌─────────────────┐
  │ fetch_ohlcv()   │ ──→ ParquetStorage (本地 Parquet 缓存)
  │ fetch_realtime()│
  └────────┬────────┘
           │ DataFrame (OHLCV)
           ▼
    ┌──────────────┐         ┌──────────────────┐
    │   Strategy   │         │   AI Analyzer    │
    │  (信号生成)  │         │  (市场分析/选股)  │
    └──────┬───────┘         └────────┬─────────┘
           │ Signal []                │ Analysis Report
           ▼                          ▼
    ┌──────────────┐         ┌──────────────────┐
    │ RiskManager  │         │  NotificationMgr │
    │ (风控过滤)   │         │  (报告推送)      │
    └──────┬───────┘         └──────────────────┘
           │ 通过风控的信号
           ▼
  ┌─────────────────┐
  │  Trading Layer  │
  │  PaperTrader /  │
  │  MooMooLive     │
  └────────┬────────┘
           │ 成交记录
           ▼
    SQLiteStorage
    ┌──────────────┐
    │ trades       │
    │ positions    │
    │ signals      │
    │ equity_curve │
    └──────────────┘
           │
           ▼
    Streamlit Dashboard
    (可视化展示)
```

---

## 6. 配置文件说明

所有模块参数均集中在 `config.yaml`，按模块分区：

```yaml
# 数据配置
data:
  cache_dir: data/cache
  default_period: 2y
  default_interval: 1d

# 策略参数
strategy:
  ma:
    fast_period: 20
    slow_period: 50
  rsi:
    period: 14
    oversold: 30
    overbought: 70
  # ... 其他策略

# 风险控制
risk:
  max_position_size: 0.20   # 单股最大仓位 20%
  atr_stop_loss_mult: 2.0
  atr_take_profit_mult: 3.0
  daily_loss_limit: -0.02   # 日亏损熔断 -2%
  max_consecutive_losses: 5

# 选股因子权重
screening:
  factors:
    momentum: 0.25
    trend: 0.20
    volume: 0.15
    volatility: 0.15
    rsi: 0.15
    ma_deviation: 0.10

# 回测设置
backtest:
  initial_capital: 100000
  commission: 0.001         # 0.1% 手续费
  slippage: 0.001           # 0.1% 滑点

# 通知配置
notifications:
  email:
    enabled: false
    # ... smtp 配置
  telegram:
    enabled: false
    # ... token/chat_id

# AI/LLM 配置（可选）
ai:
  llm_enabled: false
  llm_provider: openai
  # api_key: ...

# 优化配置
optimization:
  method: bayesian          # grid / random / bayesian
  n_trials: 100
  objective: sharpe_ratio
```

---

## 7. 使用方式

### 7.1 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：`yfinance`, `pandas`, `numpy`, `lightgbm`, `optuna`, `streamlit`, `plotly`, `sqlalchemy`

### 7.2 CLI 命令

```bash
# 回测
python main.py backtest --tickers AAPL MSFT --strategy composite --start 2022-01-01 --end 2024-01-01

# 模拟盘（启动守护线程）
python main.py paper --tickers AAPL MSFT GOOGL --strategy composite --interval 300

# 参数优化
python main.py optimize --tickers AAPL --strategy ma --method bayesian --trials 100

# 实盘（需先配置富途 API）
python main.py live --tickers AAPL --strategy composite
```

### 7.3 Web 仪表盘

```bash
streamlit run dashboard/app.py
# 默认访问 http://localhost:8501
```

### 7.4 快速回测示例

```python
from data.fetcher import YFinanceFetcher
from strategy.composite import CompositeStrategy
from backtest.engine import SimpleBacktester

fetcher = YFinanceFetcher()
data = fetcher.fetch_ohlcv("AAPL", period="2y")

strategy = CompositeStrategy(config)
backtester = SimpleBacktester(initial_capital=100000)
results = backtester.run(data, strategy)
print(results['metrics'])
```

---

## 8. 技术栈汇总

| 类别 | 技术 / 库 |
|------|-----------|
| 语言 | Python 3.9+ |
| 数据获取 | yfinance |
| 数据处理 | pandas, numpy |
| AI/ML | lightgbm, scikit-learn |
| 参数优化 | optuna |
| 可视化 | plotly, streamlit |
| 数据存储 | parquet (pyarrow), sqlite3 |
| 通知 | smtplib, twilio, python-telegram-bot |
| 实盘接口 | futu-api (富途 OpenAPI) |
| 配置管理 | PyYAML |
| 国际化 | 自实现 I18nManager + JSON 翻译文件 |

---

## 9. 已验证回测结果

以下为项目 README 中记录的历史回测示例：

| 股票 | 策略 | 周期 | 总收益 | 夏普比率 | 最大回撤 |
|------|------|------|--------|----------|----------|
| AAPL | Composite | 2022-2024 | +34.2% | 1.87 | -12.3% |
| MSFT | AI Strategy | 2022-2024 | +41.5% | 2.14 | -9.8% |
| GOOGL | Ensemble | 2022-2024 | +28.7% | 1.65 | -15.2% |

> ⚠️ 历史回测结果不代表未来收益，请注意回测过拟合风险。

---

## 10. 待改进事项与注意事项

### 🔧 待完成功能

| 项目 | 状态 | 说明 |
|------|------|------|
| 富途实盘下单 | ⚠️ 框架就绪 | `live.py` 中 `place_order` 为注释占位符，需按富途文档补全 |
| 富途沙盒对接 | ⚠️ 占位符 | `MooMooPaperTrader` 未接入真实 SDK |
| LLM 报告生成 | 🔘 可选 | 需配置 OpenAI 或其他 LLM API Key |
| SMS 通知 | 🔘 可选 | 需配置 Twilio 账户 |
| 企业微信通知 | 🔘 可选 | 需配置 Webhook URL |

### ⚠️ 重要注意事项

1. **实盘风险**：系统尚未在真实资金环境完整测试，启用实盘前务必充分验证风控参数
2. **数据延迟**：yfinance 免费数据存在 15 分钟延迟，不适合高频策略
3. **回测偏差**：回测使用日线收盘价成交，实际执行存在滑点差异
4. **AI 策略**：LightGBM 模型需要足够的历史数据训练，新上市股票效果有限
5. **富途 API**：实盘对接需要富途证券开户并获取 OpenAPI 权限
6. **时区问题**：美股时区为 ET（东部时间），运行环境时区需正确配置

### 📌 接收方快速上手建议

1. 先阅读本文档，理解整体架构
2. 配置 `config.yaml`，填写自己的参数
3. 运行 `streamlit run dashboard/app.py` 体验 Web 界面
4. 用内置股票数据跑一遍回测，验证环境正常
5. 如需实盘，参考富途 OpenAPI 文档补全 `live.py`

---

*文档由项目分析工具自动生成，基于源代码直接提取，与实际代码保持一致。*
