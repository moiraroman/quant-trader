# Task Artifact: quant_trader AI全量分析模块实现
Time: 2026-04-23 20:46 GMT+9

## 任务目标
分析用户提供的SPY+GLD全量分析提示词，识别缺漏，规划实现，逐步实现P0优先级模块。

## 提示词缺漏分析（产品文档已写入 docs/PRODUCT_AI_FULL_ANALYSIS.md）
- 付费数据(SpotGamma/GEX等)无替代方案 → 需降级策略
- 风险管理框架缺失（无仓位/回撤控制）
- 历史回测验证缺失
- SPY/GLD相关性分析缺失
- 置信度/胜率计算方法缺失
- 权重70/30无动态调整
- 关键：禁止估算价格，缺失数据标注"缺少"

## 实现完成（P0全部5个模块）
所有文件位于 quant_trader/ai/，语法全部验证通过：

### 1. multi_timeframe.py
- 分析月/周/日/4H四个时间框架
- 数据：yfinance（月线2y、周线2y、日线1y、4H 60d）
- 功能：趋势判断(HH/HL/MA排列)、均线/RSI/MACD/ADX/ATR指标计算、K线形态识别、RSI/MACD背离检测
- 数据限制：yfinance 4H最多60天，历史数据可缓存

### 2. support_resistance.py
- Pivot Point（标准+Camarilla+Fibonacci）
- 摆动高低点识别（60日窗口）
- 均线隐性支撑/阻力（MA20/50/100/200）
- 成交量轮廓（VP：POC/VAH/VAL/LVN）
- 历史高低点（52周、YTD）
- 心理关口识别
- 附近价位合并去重
- 输出最近支撑/阻力/R:R机会

### 3. strategy_advisor.py
- 胜率估算：基于历史趋势跟踪策略统计规律（ADX>25胜率55-65%，横盘40-50%）
- 背离/超买超卖有明确调整规则
- 生成5天/1个月交易策略（入场/止损/止盈/R:R/胜率）
- 分批入场方案（1个月3批次，5天2批次）
- 宏观评分动态调整置信度
- 趋势概率判断（看多/中性/看空百分比）

### 4. full_analyzer.py
- 串联所有模块的总入口
- Step1：实时价格（yfinance，~15min延迟）
- Step2：技术面70%（调用multi_timeframe + support_resistance）
- Step3：宏观/情绪30%（调用macro_scanner + 搜索恐惧贪婪指数）
- Step4：策略生成（调用strategy_advisor）
- Step5：总结归纳
- 数据降级：缺失数据标注"缺少[数据源]"，绝不估算

### 5. pattern_recognition.py
- 单根形态：锤头、射击之星、十字星、纺锤、大阴线/大阳线
- 双根形态：吞没、孕线、贯穿、乌云盖顶
- 三根形态：三乌鸦、三兵前进、黄昏星、晨星、收敛整理
- 背离检测：RSI顶背离/底背离、MACD顶背离/底背离
- 多时间框架复用（日/周/月/4H）

## 数据原则
- 所有价格/指标来自yfinance真实历史数据
- 缺失数据标注"缺少[数据源]"，绝不估算/假设
- 历史数据可缓存（历史数据不会变化）
- 付费数据（SpotGamma GEX、Barchart Max Pain等）：搜索或标注缺失

## 下一步（P1优先级）
- sentiment.py：恐惧贪婪指数获取（网络搜索）
- derivatives.py：VIX结构（已有）+ PC Ratio + GEX估算
- institutional_flows.py：ETF资金流向
- classical_theory.py：道氏理论+艾略特波浪
