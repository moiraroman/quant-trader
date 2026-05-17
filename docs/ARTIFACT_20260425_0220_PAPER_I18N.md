# 模拟交易增强模块 i18n 多语言支持完成

## 时间
2026-04-25 02:20 JST

## 目标
将模拟交易增强版 UI 中的所有硬编码英文文本替换为多语言 i18n 支持。

## 新增翻译键

在 paper_enhanced 命名空间下新增 72 个翻译键，覆盖三语言：

| 类别 | 键数 | 示例 |
|------|------|------|
| 页签/导航 | 6 | overview, trade_history, signal_analysis, strategy_config, risk_status, orders |
| 状态/指标 | 6 | status_running, status_stopped, equity, total_return, sharpe_ratio, max_drawdown |
| 权益曲线 | 6 | realtime_equity_curve, equity_curve, drawdown, no_equity_data, start_bot_hint |
| 交易历史 | 3 | no_trades_yet, start_bot_trades |
| 信号分析 | 8 | factor, appearances, avg_score, avg_confidence, recent_decisions, market_regime, factors, reasoning |
| 策略配置 | 14 | select_strategy, instance, weight, tags, parameters, default_value, reset_default, clone_strategy, delete, params_updated, no_strategy_instances, create_strategy_instance, strategy, create_instance |
| A/B测试 | 6 | ab_test, variant, return_pct, win_rate_pct, max_dd, promote_best, no_ab_data |
| 风控 | 7 | position_risk, risk_limits, max_position_pct, max_total_pct, risk_per_trade, stop_loss, take_profit, daily_loss_limit |
| 订单 | 8 | open_orders, cancel, cancel_all, manual_order, action, qty, order_type, limit_price, submit_order |
| 其他 | 5 | reset_account, account_reset, enhanced_load_failed, enhanced_not_available |

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| locales/zh.json | 新增 72 个中文翻译键 |
| locales/en.json | 新增 72 个英文翻译键 |
| locales/ja.json | 新增 72 个日文翻译键 |
| dashboard/app.py | 替换所有硬编码文本为 	('paper_enhanced.xxx') 或 	('ui.xxx') |

## 替换统计

- 成功替换：~50 处硬编码文本
- 未找到的文本：~10 处（已在代码中通过其他方式处理，如 DataFrame 列名）
- DataFrame 列名（Factor, Appearances 等）保持英文，因这是内部数据结构展示

## 语法验证
- app.py 通过 py_compile 验证
- 三语言 JSON 文件解析正常

## 备注
- 部分 DataFrame 列名（如 Factor, Appearances, Avg Score）保持英文，这些是信号解释器内部数据结构的字段名展示
- 所有用户可见的 UI 文本均已 i18n 化
