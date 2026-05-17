# Quant Trader AI全面分析 — 产品功能文档 v2.0

## 一、提示词缺漏分析

### 1.1 提示词优点
- 分步骤结构清晰（5步：价格→技术面→宏观→策略→总结）
- 权重分配明确（技术70% / 宏观30%）
- 要求量化数据、标注来源
- 强调多重深度搜索+交叉验证

### 1.2 提示词缺漏与不足

| # | 缺漏项 | 说明 | 严重程度 |
|---|--------|------|----------|
| 1 | **数据获取可行性未评估** | SpotGEX/UnusualWhales/OptionCharts 均需付费API，提示词未提供替代方案或降级策略 | 🔴高 |
| 2 | **风险管理框架缺失** | 有止盈止损但无仓位管理、最大回撤控制、相关性对冲 | 🔴高 |
| 3 | **历史回测验证缺失** | 策略完全前瞻，无"类似历史节点回测胜率"的量化验证机制 | 🔴高 |
| 4 | **SPY与GLD相关性分析缺失** | 二者通常负相关（Risk-On vs Risk-Off），但提示词将它们独立分析 | 🟡中 |
| 5 | **非交易时段处理未定义** | 盘前/盘后/周末/节假日的数据更新策略不明确 | 🟡中 |
| 6 | **置信度/胜率计算方法缺失** | 要求"胜率？%"但未定义计算方法（历史回测？蒙特卡洛？主观评估？） | 🟡中 |
| 7 | **权重70/30无动态调整机制** | 在极端行情（如VIX>40）下技术面权重应降低，宏观权重应升高 | 🟡中 |
| 8 | **无情景分析（Scenario Analysis）** | 缺少"如果X发生则Y"的条件分支分析 | 🟡中 |
| 9 | **4小时K线数据获取未覆盖** | yfinance免费版仅支持最近60天的intraday数据 | 🟡中 |
| 10 | **艾略特波浪主观性过强** | 波浪计数高度主观，不同分析师结论差异大，缺乏量化约束 | 🟢低 |

### 1.3 改进建议

1. **数据降级策略**：付费数据 → 免费近似替代  → 标注"缺少"
2. **动态权重**：根据VIX/波动率自动调整技术/宏观权重比
3. **历史模式匹配**：用DTW/相似度搜索找到历史类似节点，量化回测胜率
4. **相关性矩阵**：SPY/GLD/VIX/DXY联动分析
5. **情景树**：基准/乐观/悲观三情景概率分配

---

## 二、功能架构设计

### 2.1 模块映射

| 提示词步骤 | 当前模块 | 缺失模块 | 新建/增强 |
|------------|----------|----------|-----------|
| Step1: 实时价格 | fetcher.py (get_quote) | 时区判断、盘前/后价格 | 增强 fetcher |
| Step2.1: 宏观趋势 | market_analyzer.py | 多时间框架(月/周/日) | 新建 multi_timeframe.py |
| Step2.2: K线+指标 | market_analyzer.py | K线形态识别、背离检测 | 新建 pattern_recognition.py |
| Step2.3: 成交量分析 | ❌ | VPVR/VWAP/LVN | 新建 volume_profile.py |
| Step2.4: 衍生品 | macro_scanner.py (VIX) | GEX/PC Ratio/Max Pain | 新建 derivatives.py |
| Step2.5: 经典理论 | ❌ | 道氏/波浪 | 新建 classical_theory.py |
| Step2.6: 支撑阻力 | ❌ | 系统性识别 | 新建 support_resistance.py |
| Step2.7: 盘中异动 | ❌ | 大单检测 | 新建 institutional_detector.py |
| Step3.1: 宏观政策 | macro_scanner.py | 新闻/政策/节日 | 新建 macro_policy.py |
| Step3.2: 市场广度 | macro_scanner.py (RSP/SPY) | NH/NL, A/D | 增强 macro_scanner |
| Step3.3: 信贷市场 | macro_scanner.py (HYG/LQD) | JNK/深度信贷 | 已有基础 |
| Step3.4: 恐惧贪婪 | ❌ | CNN FG/AAII | 新建 sentiment.py |
| Step3.5: 机构资金 | ❌ | ETF flows/Smart Money | 新建 institutional_flows.py |
| Step4: 交易策略 | ❌ | 完整策略输出 | 新建 strategy_advisor.py |
| Step5: 总结 | ❌ | 表格归纳 | 集成到 orchestrator |

### 2.2 新增文件清单

```
ai/
├── multi_timeframe.py      # 多时间框架分析
├── pattern_recognition.py   # K线形态识别 + 背离检测
├── volume_profile.py        # 成交量分布分析
├── derivatives.py           # 衍生品数据分析
├── classical_theory.py      # 道氏/波浪理论
├── support_resistance.py    # 支撑/阻力位识别
├── institutional_detector.py # 机构异动检测
├── sentiment.py             # 市场情绪分析
├── institutional_flows.py   # 机构资金流向
├── strategy_advisor.py      # 交易策略生成器
└── full_analyzer.py         # 全量分析编排器（新入口）
```

### 2.3 数据流架构

```
用户请求(SPY+GLD)
    │
    ▼
full_analyzer.py (FullAnalyzer)
    │
    ├── Step1: 实时价格 (fetcher.get_quote + timezone判断)
    │
    ├── Step2: 技术面分析 [权重70%]
    │   ├── multi_timeframe.py → 月/周/日/4h 分析
    │   ├── pattern_recognition.py → K线形态 + 背离
    │   ├── volume_profile.py → VPVR/VWAP/LVN
    │   ├── derivatives.py → VIX结构/GEX/PC Ratio
    │   ├── classical_theory.py → 道氏/波浪
    │   ├── support_resistance.py → 关键价位
    │   └── institutional_detector.py → 尾盘异动
    │
    ├── Step3: 宏观+情绪 [权重30%]
    │   ├── macro_scanner.py → 环境评分
    │   ├── sector_rotation.py → 板块轮动
    │   ├── sentiment.py → 恐惧贪婪/AAII
    │   └── institutional_flows.py → ETF资金流
    │
    ├── Step4: 策略生成
    │   └── strategy_advisor.py → 5天/1月策略
    │
    └── Step5: 总结输出
```

---

## 三、实现优先级

### P0 — 核心缺失（无替代） ✅ 已完成
1. **strategy_advisor.py** — 最终交付物，5天/1月策略 ✅
2. **multi_timeframe.py** — 多时间框架是技术分析核心 ✅
3. **pattern_recognition.py** — K线形态+背离检测 ✅
4. **support_resistance.py** — 支撑阻力位 ✅
5. **full_analyzer.py** — 编排器，串联所有模块 ✅

### P1 — 重要增强 ✅ 已完成
6. **volume_profile.py** — 成交量分布（VPVR/VWAP/POC/VAH/VAL/LVN）✅
7. **derivatives.py** — 衍生品数据（VIX期限结构/GEX估算/PCR）✅
8. **sentiment.py** — 恐惧贪婪指数（CNN/AAII/CTA/市场广度）✅
9. **classical_theory.py** — 道氏理论+艾略特波浪估算 ✅

### P2 — 锦上添花 ✅ 已完成
10. **institutional_detector.py** — 机构异动检测（尾盘/开盘/成交量/跳空）✅
11. **institutional_flows.py** — ETF资金流近似（价格+成交量推断）✅
12. **macro_policy.py** — 宏观政策分析（FOMC/经济日历/政策风险）✅

### P3 — 高级功能 ✅ 已完成
13. **pattern_matcher.py** — 历史模式匹配（DTW+相关系数，回测胜率统计）✅
14. **dynamic_weights.py** — 动态权重调整（VIX/趋势/波动率三因子）✅
15. **scenario_analysis.py** — 三情景分析（基准/乐观/悲观+概率+目标价）✅
16. **correlation_matrix.py** — 相关性矩阵（SPY/GLD/VIX/DXY/TLT/QQQ/IWM）✅

### P4 — 未来扩展 ✅ 已完成
17. **backtest_engine.py** — 策略回测引擎（RSI+MACD+均线综合信号，胜率/收益/回撤/Sharpe）✅
18. **monte_carlo.py** — 蒙特卡洛模拟（GBM几何布朗运动，1000路径，VaR/CVaR，目标价/止损触及概率）✅
19. **ml_predictor.py** — 机器学习预测（RandomForest分类/回归，18维特征集，sklearn降级统计方案）✅
20. **alert_system.py** — 实时预警系统（价格/RSI/成交量/VIX多条件，冷却机制，严重度分级）✅

### P5 — 未来方向（待规划）
21. **组合优化** — 多标的仓位分配（马科维茨/风险平价）
22. **事件驱动分析** — 财报/政策/宏观事件前后价格影响
23. **期权链分析** — 完整Greeks/IV Skew/期限结构（需付费数据）
24. **情绪NLP** — 新闻/社交媒体情绪提取（需NLP模型）

---

## 四、策略输出格式

### 4.1 趋势概率判断

| 标的 | 时间维度 | 看多概率 | 中性概率 | 看空概率 | 核心依据 |
|------|----------|----------|----------|----------|----------|
| SPY | 5天 | XX% | XX% | XX% | ... |
| SPY | 1个月 | XX% | XX% | XX% | ... |
| GLD | 5天 | XX% | XX% | XX% | ... |
| GLD | 1个月 | XX% | XX% | XX% | ... |

### 4.2 交易策略格式

**5天交易策略：**
- 实时方案：现价X，做[多/空/等待]，止盈Z，止损Y，R:R≈N:1，胜率XX%，依据XXX
- 优先方案（分批）：[多/空]，X价投X%，X价投X%，R:R≈，胜率XX%
- 次选方案（分批）：同上

**1个月交易策略：** 同上格式

### 4.3 总结表格

| 维度 | SPY结论 | GLD结论 | 风险提示 | 机会提示 |
|------|---------|---------|----------|----------|
| 技术面 | ... | ... | ... | ... |
| 宏观面 | ... | ... | ... | ... |
| 策略 | ... | ... | ... | ... |
