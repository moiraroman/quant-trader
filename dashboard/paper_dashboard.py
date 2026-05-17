"""
dashboard/paper_dashboard.py — 模拟交易 Dashboard 增强模块

功能：
    - 实时净值曲线（Plotly）
    - 交易历史明细表
    - 盈亏分析图表
    - 信号可视化
    - 风控状态面板
    - 策略参数调整界面
    - 绩效归因展示
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ============================================================
# 净值曲线组件
# ============================================================

def render_equity_curve(equity_tracker, days: int = 30):
    """渲染净值曲线"""
    df = equity_tracker.get_equity_curve(days)
    
    if df.empty:
        st.info("暂无净值数据，开始交易后将显示曲线")
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=('权益曲线', '回撤'),
    )
    
    # 权益曲线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['total_equity'],
            mode='lines',
            name='总资产',
            line=dict(color='#00C851', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 81, 0.1)',
        ),
        row=1, col=1,
    )
    
    # 现金
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['cash'],
            mode='lines',
            name='现金',
            line=dict(color='#FF8800', width=1),
        ),
        row=1, col=1,
    )
    
    # 持仓市值
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['position_value'],
            mode='lines',
            name='持仓市值',
            line=dict(color='#33B5E5', width=1),
        ),
        row=1, col=1,
    )
    
    # 回撤
    cummax = df['total_equity'].cummax()
    drawdown = (df['total_equity'] - cummax) / cummax * 100
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=drawdown,
            mode='lines',
            name='回撤%',
            line=dict(color='#FF4444', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.1)',
        ),
        row=2, col=1,
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_yaxes(title_text="金额 ($)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 绩效指标卡片
# ============================================================

def render_performance_cards(metrics):
    """渲染绩效指标卡片"""
    if not metrics:
        st.info("暂无绩效数据")
        return
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            label="总收益率",
            value=f"{metrics.total_return_pct:+.2f}%",
        )
    
    with cols[1]:
        st.metric(
            label="夏普比率",
            value=f"{metrics.sharpe_ratio:.2f}",
        )
    
    with cols[2]:
        st.metric(
            label="最大回撤",
            value=f"{metrics.max_drawdown_pct:.2f}%",
        )
    
    with cols[3]:
        st.metric(
            label="胜率",
            value=f"{metrics.win_rate_pct:.1f}%",
        )
    
    # 第二行
    cols2 = st.columns(4)
    
    with cols2[0]:
        st.metric(
            label="年化收益",
            value=f"{metrics.annualized_return_pct:+.2f}%",
        )
    
    with cols2[1]:
        st.metric(
            label="Sortino比率",
            value=f"{metrics.sortino_ratio:.2f}",
        )
    
    with cols2[2]:
        st.metric(
            label="Calmar比率",
            value=f"{metrics.calmar_ratio:.2f}",
        )
    
    with cols2[3]:
        st.metric(
            label="当前回撤",
            value=f"{metrics.current_drawdown_pct:.2f}%",
        )

# ============================================================
# 交易历史表格
# ============================================================

def render_trade_history(equity_tracker, days: int = 30):
    """渲染交易历史"""
    df = equity_tracker.get_trade_history(days)
    
    if df.empty:
        st.info("暂无交易记录")
        return
    
    # 格式化
    display_df = df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
    display_df['realized_pnl'] = display_df['realized_pnl'].apply(
        lambda x: f"${x:+.2f}" if pd.notna(x) else "-"
    )
    
    # 颜色标记
    def highlight_pnl(val):
        if isinstance(val, str) and val.startswith('$+'):
            return 'background-color: rgba(0, 200, 81, 0.2)'
        elif isinstance(val, str) and val.startswith('$-'):
            return 'background-color: rgba(255, 68, 68, 0.2)'
        return ''
    
    styled = display_df.style.applymap(highlight_pnl, subset=['realized_pnl'])
    
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            'trade_id': st.column_config.TextColumn("交易ID", width="small"),
            'timestamp': st.column_config.TextColumn("时间", width="medium"),
            'ticker': st.column_config.TextColumn("标的", width="small"),
            'action': st.column_config.TextColumn("动作", width="small"),
            'qty': st.column_config.NumberColumn("数量", width="small"),
            'price': st.column_config.TextColumn("价格", width="small"),
            'signal_source': st.column_config.TextColumn("信号源", width="medium"),
            'signal_confidence': st.column_config.ProgressColumn(
                "置信度",
                min_value=0,
                max_value=1,
                format="%.0%%",
            ),
            'realized_pnl': st.column_config.TextColumn("实现盈亏", width="small"),
        },
    )

# ============================================================
# 信号可视化
# ============================================================

def render_signal_visualization(trade_history_df, price_data=None):
    """在价格图上标记交易信号"""
    if trade_history_df.empty:
        st.info("暂无信号数据")
        return
    
    fig = go.Figure()
    
    # 如果有价格数据，绘制K线
    if price_data is not None:
        fig.add_trace(go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name='价格',
        ))
    
    # 标记买入信号
    buy_signals = trade_history_df[trade_history_df['action'] == 'BUY']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['timestamp'],
            y=buy_signals['price'],
            mode='markers',
            name='买入',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='#00C851',
                line=dict(width=2, color='white'),
            ),
            text=[f"{row['ticker']}<br>置信度: {row.get('signal_confidence', 0):.0%}"
                  for _, row in buy_signals.iterrows()],
            hovertemplate='%{text}<br>价格: $%{y:.2f}<extra></extra>',
        ))
    
    # 标记卖出信号
    sell_signals = trade_history_df[trade_history_df['action'] == 'SELL']
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['timestamp'],
            y=sell_signals['price'],
            mode='markers',
            name='卖出',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='#FF4444',
                line=dict(width=2, color='white'),
            ),
            text=[f"{row['ticker']}<br>置信度: {row.get('signal_confidence', 0):.0%}"
                  for _, row in sell_signals.iterrows()],
            hovertemplate='%{text}<br>价格: $%{y:.2f}<extra></extra>',
        ))
    
    fig.update_layout(
        title="交易信号可视化",
        height=500,
        template='plotly_dark',
        xaxis_title="时间",
        yaxis_title="价格 ($)",
        hovermode='x unified',
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 风控状态面板
# ============================================================

def render_risk_status(risk_manager, portfolio_value: float):
    """渲染风控状态面板"""
    st.subheader("风控状态")
    
    # 获取风控状态
    status = risk_manager.get_status() if hasattr(risk_manager, 'get_status') else {}
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric(
            label="当日回撤",
            value=f"{status.get('daily_drawdown', 0):.2f}%",
            delta=f"限额: {status.get('daily_drawdown_limit', 2)}%",
        )
    
    with cols[1]:
        st.metric(
            label="连续亏损",
            value=f"{status.get('consecutive_losses', 0)} 次",
            delta=f"限额: {status.get('max_consecutive_losses', 3)} 次",
        )
    
    with cols[2]:
        st.metric(
            label="冷静期",
            value="活跃" if status.get('cooling_down', False) else "正常",
            delta=f"剩余: {status.get('cooldown_remaining', 0)} 分钟",
        )
    
    # 风控规则状态
    st.write("---")
    st.write("**风控规则状态**")
    
    rules = [
        ("单日亏损熔断", status.get('daily_loss_triggered', False), "当日亏损超过限额"),
        ("连续亏损暂停", status.get('consecutive_loss_triggered', False), "连续亏损次数超限"),
        ("冷静期", status.get('cooling_down', False), "触发后暂停交易"),
        ("移动止损", status.get('trailing_stop_active', False), "跟踪最高盈利回撤"),
    ]
    
    for rule_name, is_triggered, description in rules:
        col1, col2, col3 = st.columns([2, 1, 3])
        with col1:
            st.write(f"• {rule_name}")
        with col2:
            st.badge(
                "已触发" if is_triggered else "正常",
                color="red" if is_triggered else "green",
            )
        with col3:
            st.caption(description)

# ============================================================
# 策略参数调整界面
# ============================================================

def render_strategy_config(strategy_config_manager, instance_id: str):
    """渲染策略参数调整界面"""
    instance = strategy_config_manager.get_instance(instance_id)
    
    if not instance:
        st.error(f"策略实例 {instance_id} 不存在")
        return
    
    st.subheader(f"策略配置: {instance.strategy_name} ({instance.ticker})")
    
    # 参数调整
    st.write("**参数调整**")
    
    updated = False
    for param_name, param in instance.parameters.items():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if param.param_type == "int":
                new_value = st.number_input(
                    f"{param.name} ({param.description})",
                    min_value=int(param.min_value) if param.min_value else None,
                    max_value=int(param.max_value) if param.max_value else None,
                    value=int(param.value),
                    step=int(param.step) if param.step else 1,
                    key=f"param_{instance_id}_{param_name}",
                )
            elif param.param_type == "float":
                new_value = st.number_input(
                    f"{param.name} ({param.description})",
                    min_value=param.min_value,
                    max_value=param.max_value,
                    value=float(param.value),
                    step=param.step or 0.01,
                    key=f"param_{instance_id}_{param_name}",
                )
            elif param.param_type == "bool":
                new_value = st.checkbox(
                    f"{param.name} ({param.description})",
                    value=bool(param.value),
                    key=f"param_{instance_id}_{param_name}",
                )
            elif param.param_type == "choice":
                new_value = st.selectbox(
                    f"{param.name} ({param.description})",
                    options=param.choices,
                    index=param.choices.index(param.value) if param.value in param.choices else 0,
                    key=f"param_{instance_id}_{param_name}",
                )
            else:
                new_value = param.value
            
            if new_value != param.value:
                strategy_config_manager.update_parameter(instance_id, param_name, new_value)
                updated = True
        
        with col2:
            st.caption(f"默认: {param.default_value}")
    
    if updated:
        st.success("参数已更新")
    
    # 操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("重置为默认", key=f"reset_{instance_id}"):
            strategy_config_manager.reset_parameters(instance_id)
            st.success("已重置为默认参数")
            st.rerun()
    
    with col2:
        if st.button("克隆策略", key=f"clone_{instance_id}"):
            new_inst = strategy_config_manager.clone_instance(instance_id)
            if new_inst:
                st.success(f"已克隆为 {new_inst.instance_id}")
    
    with col3:
        if st.button("删除实例", key=f"delete_{instance_id}"):
            strategy_config_manager.delete_instance(instance_id)
            st.success("已删除")
            st.rerun()

# ============================================================
# 绩效归因展示
# ============================================================

def render_attribution(equity_tracker, days: int = 30):
    """渲染绩效归因"""
    attribution = equity_tracker.analyze_attribution(days)
    
    if not attribution:
        st.info("暂无归因数据")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**按信号源归因**")
        source_data = attribution.get('by_signal_source', {})
        if source_data:
            df = pd.DataFrame([
                {'信号源': k, '盈亏': v}
                for k, v in source_data.items()
            ])
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df['信号源'],
                    y=df['盈亏'],
                    marker_color=['#00C851' if x > 0 else '#FF4444' for x in df['盈亏']],
                )
            ])
            fig.update_layout(
                height=300,
                template='plotly_dark',
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**按标的归因**")
        ticker_data = attribution.get('by_ticker', {})
        if ticker_data:
            df = pd.DataFrame([
                {'标的': k, '盈亏': v}
                for k, v in ticker_data.items()
            ])
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df['标的'],
                    y=df['盈亏'],
                    marker_color=['#00C851' if x > 0 else '#FF4444' for x in df['盈亏']],
                )
            ])
            fig.update_layout(
                height=300,
                template='plotly_dark',
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 滑点分析
    st.write("---")
    st.write("**执行质量分析**")
    
    slippage = attribution.get('slippage_analysis', {})
    exec_quality = attribution.get('execution_quality', {})
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("平均滑点", f"${slippage.get('avg_slippage_per_trade', 0):.4f}")
    with cols[1]:
        st.metric("总滑点成本", f"${slippage.get('total_slippage_cost', 0):.2f}")
    with cols[2]:
        st.metric("滑点/PnL", f"{slippage.get('slippage_as_pct_of_pnl', 0):.2f}%")
    with cols[3]:
        st.metric("平均延迟", f"{exec_quality.get('avg_execution_delay_ms', 0):.0f}ms")

# ============================================================
# 主渲染函数
# ============================================================

def render_paper_trading_dashboard(
    paper_trader,
    equity_tracker,
    order_manager,
    signal_explainer,
    strategy_config_manager,
    risk_manager,
):
    """
    渲染完整的模拟交易 Dashboard。
    
    页签结构：
    1. 概览 - 净值曲线 + 绩效卡片
    2. 交易历史 - 详细交易记录
    3. 信号分析 - 信号可视化 + 归因
    4. 策略配置 - 参数调整 + A/B测试
    5. 风控状态 - 风控规则 + 持仓风险
    6. 订单管理 - 当前订单 + 下单
    """
    
    st.title("模拟交易")
    
    tabs = st.tabs([
        "概览",
        "交易历史",
        "信号分析",
        "策略配置",
        "风控状态",
        "订单管理",
    ])
    
    # === 概览 ===
    with tabs[0]:
        st.subheader("实时净值")
        render_equity_curve(equity_tracker, days=30)
        
        st.subheader("绩效指标")
        metrics = equity_tracker.calculate_metrics(days=30)
        render_performance_cards(metrics)
        
        # 实时状态
        st.subheader("当前状态")
        status = equity_tracker.get_realtime_status()
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("当前权益", f"${status.get('current_equity', 0):,.2f}")
        with cols[1]:
            st.metric("现金", f"${status.get('cash', 0):,.2f}")
        with cols[2]:
            st.metric("持仓市值", f"${status.get('position_value', 0):,.2f}")
        with cols[3]:
            st.metric("未实现盈亏", f"${status.get('unrealized_pnl', 0):+.2f}")
    
    # === 交易历史 ===
    with tabs[1]:
        st.subheader("交易记录")
        render_trade_history(equity_tracker, days=30)
        
        st.subheader("信号可视化")
        trades_df = equity_tracker.get_trade_history(days=30)
        render_signal_visualization(trades_df)
    
    # === 信号分析 ===
    with tabs[2]:
        st.subheader("绩效归因")
        render_attribution(equity_tracker, days=30)
        
        st.subheader("因子统计")
        factor_stats = signal_explainer.get_factor_statistics(days=30)
        if factor_stats:
            df = pd.DataFrame([
                {
                    '因子': k,
                    '出现次数': v['appearances'],
                    '平均得分': f"{v['avg_score']:+.3f}",
                    '平均置信度': f"{v['avg_confidence']:.1%}",
                }
                for k, v in factor_stats.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # === 策略配置 ===
    with tabs[3]:
        st.subheader("策略实例")
        
        instances = strategy_config_manager.get_active_instances()
        if instances:
            instance_options = {f"{i.strategy_name} ({i.ticker})": i.instance_id for i in instances}
            selected = st.selectbox("选择策略", options=list(instance_options.keys()))
            
            if selected:
                render_strategy_config(strategy_config_manager, instance_options[selected])
        else:
            st.info("暂无策略实例，请在设置中创建")
        
        st.write("---")
        st.subheader("A/B 测试")
        
        ab_results = strategy_config_manager.get_ab_test_results()
        if ab_results:
            df = pd.DataFrame([
                {
                    '变体': k,
                    '总收益': f"{v['total_return']:+.2f}%",
                    '夏普': f"{v['sharpe']:.2f}",
                    '胜率': f"{v['win_rate']:.1f}%",
                    '最大回撤': f"{v['max_dd']:.2f}%",
                }
                for k, v in ab_results.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            if st.button("提升最佳变体"):
                promoted = strategy_config_manager.promote_best_variant()
                if promoted:
                    st.success(f"已提升 {promoted}")
        else:
            st.info("暂无 A/B 测试数据")
    
    # === 风控状态 ===
    with tabs[4]:
        render_risk_status(risk_manager, paper_trader.get_total_equity() if hasattr(paper_trader, 'get_total_equity') else 0)
        
        st.write("---")
        st.subheader("持仓风险")
        
        positions = paper_trader.get_positions() if hasattr(paper_trader, 'get_positions') else {}
        if positions:
            df = pd.DataFrame([
                {
                    '标的': ticker,
                    '数量': pos['qty'],
                    '成本': f"${pos['avg_cost']:.2f}",
                    '当前': f"${pos.get('current_price', 0):.2f}",
                    '盈亏': f"${pos.get('unrealized_pnl', 0):+.2f}",
                    '权重': f"{pos.get('weight_pct', 0):.1f}%",
                }
                for ticker, pos in positions.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # === 订单管理 ===
    with tabs[5]:
        st.subheader("当前订单")
        
        open_orders = order_manager.get_open_orders()
        if open_orders:
            df = pd.DataFrame([o.to_dict() for o in open_orders])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            if st.button("撤销全部订单"):
                order_manager.cancel_all_orders()
                st.success("已撤销")
                st.rerun()
        else:
            st.info("暂无未完成订单")
        
        st.write("---")
        st.subheader("手动下单")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            order_ticker = st.text_input("标的", value="AAPL")
        with col2:
            order_action = st.selectbox("动作", ["BUY", "SELL"])
        with col3:
            order_qty = st.number_input("数量", min_value=1, value=100)
        
        order_type = st.selectbox("订单类型", ["MARKET", "LIMIT", "STOP"])
        
        price = None
        stop_price = None
        
        if order_type == "LIMIT":
            price = st.number_input("限价", min_value=0.01, value=150.0)
        elif order_type == "STOP":
            stop_price = st.number_input("止损价", min_value=0.01, value=140.0)
        
        if st.button("提交订单"):
            order = order_manager.submit_order(
                ticker=order_ticker,
                action=order_action,
                qty=order_qty,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
            )
            st.success(f"订单已提交: {order.order_id}")
