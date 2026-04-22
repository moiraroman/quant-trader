# ============================================================
# dashboard/app.py - Streamlit Web Dashboard (v5)
# - 页面顶部 tabs，移除 sidebar
# - 每个 tab 内容独立，内部包含所有必要参数
# - 语言选择移至 Settings tab
# ============================================================
import os
import sys
import importlib
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Force reload core modules
_RELOAD_MODULES = [
    "strategy.technical", "strategy.composite", "strategy.base",
    "backtest.engine", "data.fetcher", "data.storage",
    "risk.manager", "trading.paper", "trading.bot",
    "ai.market_analyzer", "ai.stock_screener", "ai.orchestrator",
    "ai.macro_scanner",
]
for _mod_name in _RELOAD_MODULES:
    if _mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[_mod_name])
        except Exception:
            pass


# ============================================================
# I18n Manager
# ============================================================
class I18nManager:
    SUPPORTED_LANGUAGES = {"en": "English", "zh": "中文", "ja": "日本語"}

    def __init__(self, locales_dir=None, default_lang="zh"):
        if locales_dir is None:
            locales_dir = BASE_DIR / "locales"
        self.locales_dir = Path(locales_dir)
        self.default_lang = default_lang
        self._translations = {}
        self._load_all()

    def _load_all(self):
        import json
        for lang in self.SUPPORTED_LANGUAGES:
            path = self.locales_dir / f"{lang}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._translations[lang] = json.load(f)

    def get(self, key, lang=None, _fallback=False):
        if lang is None:
            lang = st.session_state.get("language", self.default_lang)
        keys = key.split(".")
        value = self._translations.get(lang, {})
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, "")
            else:
                return ""
        if not value and not _fallback:
            if lang != "en":
                return self.get(key, "en", _fallback=True)
            elif lang != self.default_lang:
                return self.get(key, self.default_lang, _fallback=True)
        return value if value else key


_i18n = None

def t(key):
    global _i18n
    if _i18n is None:
        _i18n = I18nManager()
    return _i18n.get(key)


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title=t("app.title"),
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Zero top padding; title flush with viewport top
st.markdown("<style>:root { --top-padding: 0rem; }</style>", unsafe_allow_html=True)
st.markdown(
    "<style>" "div[data-testid='stApp'] { padding-top: 0 !important; }" "</style>",
    unsafe_allow_html=True,
)

# ============================================================
# Header: Title (left) + Tabs (right) via CSS flex
# ============================================================
TAB_LABELS = [
    f"📊 {t('app.mode_backtest')}",
    f"📈 {t('app.mode_paper')}",
    f"🔴 {t('app.mode_live')}",
    f"🤖 {t('app.mode_ai')}",
    f"⚙️ {t('app.mode_optimize')}",
    f"🔧 {t('app.mode_settings')}",
]

# CSS: title-block left, tabs-container right, vertically centered, no gap
st.markdown(f"""
<style>
/* ---- Hide Streamlit built-in header (deploy button etc.) ---- */
header[data-testid="stHeader"] {{
    display: none !important;
}}
.stDeployButton {{
    display: none !important;
}}

/* Flex container: title | tabs */
div.header-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.0rem 1.5rem 0.6rem 1.5rem;
    border-bottom: 1px solid #e0e0e0;
    background: #fff;
    position: sticky;
    top: 0;
    z-index: 999;
    gap: 0;
}}
/* Title area */
div.header-title {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin: 0;
    padding: 0;
}}
div.header-title h1 {{
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    padding: 0 !important;
    color: #1a1a2e !important;
    line-height: 1.2 !important;
}}
div.header-title span {{
    font-size: 0.72rem !important;
    color: #888 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}}
/* Move tabs-container up into header row */
div[data-testid='stTabs'] {{
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
    width: auto !important;
}}
/* Override block-container top padding */
div.block-container {{
    padding-top: 0.5rem !important;
}}
</style>
<div class="header-row">
    <div class="header-title">
        <h1>📈 {t('app.title')}</h1>
        <span>{t('app.version')}: 0.1 &nbsp;|&nbsp; {t('app.contact_author')}: noizu19@gmail.com</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs immediately after the header HTML — CSS will pull them into the header row
tab_backtest, tab_paper, tab_live, tab_ai, tab_optimize, tab_settings = st.tabs(TAB_LABELS)

# Thin separator below tabs
st.markdown("<style>div[data-testid='stTabs'] {{ border-bottom: 2px solid #e0e0e0; }}</style>", unsafe_allow_html=True)


# ============================================================
# TAB 0: Backtest Mode
# ============================================================
with tab_backtest:
    st.markdown(f"### 📋 {t('app.mode_backtest')}")

    # Parameters for this tab
    col_ticker, col_period = st.columns([1, 1])
    with col_ticker:
        ticker = st.text_input(t("backtest.ticker_label"), value="AAPL", key="bt_ticker").upper()
    with col_period:
        period = st.selectbox(
            t("backtest.period_label"),
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2, key="bt_period",
        )

    col_cash, col_comm = st.columns([1, 1])
    with col_cash:
        initial_cash = st.number_input(
            t("backtest.initial_cash"),
            min_value=10000, max_value=10000000, value=100000, step=10000, key="bt_cash",
        )
    with col_comm:
        commission = st.number_input(
            t("backtest.commission"),
            min_value=0.0, max_value=0.01, value=0.001, step=0.0001, format="%.4f", key="bt_comm",
        )

    # Strategy selection
    st.markdown(f"#### 🎯 {t('strategy.title')}")
    available_strategies = {
        "MA": t("strategy.strategy_ma"),
        "RSI": t("strategy.strategy_rsi"),
        "MACD": t("strategy.strategy_macd"),
        "Bollinger": t("strategy.strategy_bollinger"),
        "Composite": t("strategy.strategy_composite"),
    }
    selected_strategies = st.multiselect(
        t("strategy.select_strategies"),
        options=list(available_strategies.keys()),
        default=["MA", "RSI"],
        format_func=lambda x: available_strategies[x], key="bt_strats",
    )
    min_vote = st.slider(
        t("strategy.min_votes"), 1,
        max(len(selected_strategies), 2) if selected_strategies else 2,
        max(1, len(selected_strategies) // 2 + 1) if selected_strategies else 1,
        key="bt_minvote",
    )

    # Strategy params
    strategy_params = {}
    for strat in selected_strategies:
        with st.expander(f"⚙️ {available_strategies[strat]}", expanded=False):
            enabled = st.checkbox(t("strategy.enabled"), value=True, key=f"bt_{strat}_en")
            weight = st.slider(t("strategy.weight"), 0.0, 3.0, 1.0, 0.1, key=f"bt_{strat}_wt")
            params = {}
            if strat == "MA":
                params["short_window"] = st.number_input(t("strategy.ma_short"), 5, 50, 10, key=f"bt_{strat}_sw")
                params["long_window"] = st.number_input(t("strategy.ma_long"), 20, 200, 30, key=f"bt_{strat}_lw")
            elif strat == "RSI":
                params["period"] = st.number_input(t("strategy.rsi_period"), 5, 30, 14, key=f"bt_{strat}_rp")
                params["oversold"] = st.number_input(t("strategy.rsi_oversold"), 20, 40, 30, key=f"bt_{strat}_ros")
                params["overbought"] = st.number_input(t("strategy.rsi_overbought"), 60, 80, 70, key=f"bt_{strat}_rob")
            elif strat == "MACD":
                params["fast"] = st.number_input(t("strategy.macd_fast"), 5, 20, 12, key=f"bt_{strat}_f")
                params["slow"] = st.number_input(t("strategy.macd_slow"), 20, 40, 26, key=f"bt_{strat}_sl")
                params["signal"] = st.number_input(t("strategy.macd_signal"), 5, 15, 9, key=f"bt_{strat}_sg")
            elif strat == "Bollinger":
                params["period"] = st.number_input(t("strategy.boll_period"), 10, 30, 20, key=f"bt_{strat}_bp")
                params["std_dev"] = st.number_input(t("strategy.boll_std"), 1.0, 3.0, 2.0, 0.1, key=f"bt_{strat}_bsd")
            elif strat == "Composite":
                params["min_conditions"] = st.number_input(t("strategy.composite_min_conditions"), 1, 5, 3, key=f"bt_{strat}_mc")
                params["risk_per_trade"] = st.slider(t("strategy.composite_risk_per_trade"), 1, 10, 6, key=f"bt_{strat}_rpt") / 100
                params["stop_loss_atr"] = st.slider(t("strategy.composite_stop_loss_atr"), 1.0, 5.0, 2.0, 0.5, key=f"bt_{strat}_sla")
                params["take_profit_atr"] = st.slider(t("strategy.composite_take_profit_atr"), 2.0, 15.0, 8.0, 0.5, key=f"bt_{strat}_tpa")
                params["rsi_oversold"] = st.number_input(t("strategy.composite_rsi_oversold"), 20, 50, 40, key=f"bt_{strat}_ros")
                params["rsi_overbought"] = st.number_input(t("strategy.composite_rsi_overbought"), 60, 90, 70, key=f"bt_{strat}_rob")
                params["adx_range_max"] = st.number_input(t("strategy.composite_adx_range_max"), 15, 35, 30, key=f"bt_{strat}_arm")
                params["adx_trend_min"] = st.number_input(t("strategy.composite_adx_trend_min"), 30, 60, 40, key=f"bt_{strat}_atm")
            strategy_params[strat] = {"enabled": enabled, "weight": weight, "params": params}

    # Risk params
    st.markdown(f"#### 🛡️ {t('risk.title')}")
    col_r1, col_r2, col_r3, col_r4 = st.columns([1, 1, 1, 1])
    with col_r1:
        risk_pct = st.slider(t("risk.per_trade"), 0.5, 5.0, 2.0, 0.5, key="bt_risk_pct") / 100
    with col_r2:
        stop_atr = st.slider(t("risk.stop_loss_atr"), 1.0, 5.0, 2.0, 0.5, key="bt_stop_atr")
    with col_r3:
        tp_atr = st.slider(t("risk.take_profit_atr"), 1.0, 10.0, 3.0, 0.5, key="bt_tp_atr")
    with col_r4:
        hard_stop_pct = st.slider(t("risk.hard_stop_loss"), 3.0, 20.0, 8.0, 1.0, key="bt_hard_stop") / 100
    risk_reward = tp_atr / stop_atr if stop_atr > 0 else 1.5

    st.markdown("---")

    # Run button
    run_btn = st.button(f"🚀 {t('backtest.run_backtest')}", key="bt_run_btn", use_container_width=True)

    st.markdown(f"**{t('ui.target')}**: {ticker} | **{t('ui.backtest_period')}**: {period} | **{t('ui.risk_reward_ratio')}**: {risk_reward:.1f}:1")

    if selected_strategies:
        st.markdown(f"**{t('ui.selected_strategies')}**: " + " | ".join(available_strategies[s] for s in selected_strategies))
    else:
        st.warning(t("ui.no_strategy_selected"))

    st.markdown("---")

    if run_btn or "bt_result" not in st.session_state:
        with st.spinner(t("ui.fetching_data")):
            try:
                from data.fetcher import YFinanceFetcher
                from strategy.technical import MAStrategy, RSIStrategy, MACDStrategy, BollingerStrategy
                from strategy.composite import CompositeStrategy
                from backtest.engine import SimpleBacktester as BacktestEngine

                fetcher = YFinanceFetcher()
                df = fetcher.download_history(ticker, period=period, interval="1d")

                if df.empty:
                    st.error(t("ui.no_data"))
                else:
                    all_signals = []
                    strategy_map = {
                        "MA": (MAStrategy, {"short_window": 10, "long_window": 30}),
                        "RSI": (RSIStrategy, {"period": 14, "oversold": 30, "overbought": 70}),
                        "MACD": (MACDStrategy, {"fast": 12, "slow": 26, "signal": 9}),
                        "Bollinger": (BollingerStrategy, {"period": 20, "std_dev": 2.0}),
                    }

                    for strat_name in selected_strategies:
                        cfg = strategy_params.get(strat_name, {"enabled": True, "weight": 1.0, "params": {}})
                        if not cfg.get("enabled", True):
                            continue
                        weight = cfg.get("weight", 1.0)
                        user_params = cfg.get("params", {})

                        if strat_name == "Composite":
                            comp_cfg = strategy_params.get("Composite", {"enabled": True, "weight": 1.0, "params": {}})
                            comp_params = comp_cfg.get("params", {})
                            comp = CompositeStrategy(**comp_params)
                            sig_df = comp._compute_signals(df)
                            if not sig_df.empty:
                                sig_df["weight"] = weight
                                all_signals.append(sig_df)
                        elif strat_name in strategy_map:
                            cls, default_p = strategy_map[strat_name]
                            merged = {**default_p, **user_params}
                            inst = cls(**merged)
                            sig_df = inst._compute_signals(df)
                            if not sig_df.empty:
                                sig_df["weight"] = weight
                                all_signals.append(sig_df)

                    if not all_signals:
                        st.warning(t("ui.no_strategy_selected"))
                    else:
                        combined = all_signals[0][["signal"]].copy()
                        combined["buy_votes"] = 0.0
                        combined["sell_votes"] = 0.0
                        combined["confidence"] = 0.0
                        combined["strength"] = 0.0
                        combined["reason"] = ""

                        for sig_df in all_signals:
                            w = sig_df["weight"].iloc[0] if "weight" in sig_df.columns else 1.0
                            common_idx = combined.index.intersection(sig_df.index)
                            for idx in common_idx:
                                row = sig_df.loc[idx]
                                sig = row.get("signal", "HOLD")
                                conf = row.get("confidence", 0.5)
                                stren = row.get("strength", 0.5)
                                reason = str(row.get("reason", ""))
                                if sig == "BUY":
                                    combined.at[idx, "buy_votes"] += 1
                                elif sig == "SELL":
                                    combined.at[idx, "sell_votes"] += 1
                                combined.at[idx, "confidence"] = max(combined.at[idx, "confidence"], conf * w)
                                combined.at[idx, "strength"] = max(combined.at[idx, "strength"], stren * w)
                                if reason:
                                    existing = combined.at[idx, "reason"]
                                    combined.at[idx, "reason"] = f"{existing}; {reason}" if existing else reason

                        for idx in combined.index:
                            bv = combined.at[idx, "buy_votes"]
                            sv = combined.at[idx, "sell_votes"]
                            if bv >= min_vote and bv > sv:
                                combined.at[idx, "signal"] = "BUY"
                            elif sv >= min_vote and sv > bv:
                                combined.at[idx, "signal"] = "SELL"
                            else:
                                combined.at[idx, "signal"] = "HOLD"

                        engine = BacktestEngine(initial_cash=initial_cash, commission=commission)
                        result = engine.run(df, combined, ticker=ticker)
                        st.session_state["bt_result"] = result

            except Exception as e:
                import traceback
                st.error(f"{t('common.error')}: {e}")
                st.code(traceback.format_exc())

    if "bt_result" in st.session_state:
        result = st.session_state["bt_result"]
        if not result:
            st.warning(t("ui.no_data"))
        else:
            metrics = result.get("metrics", {})
            equity_curve = result.get("equity_curve")
            trades = result.get("trades", [])

            st.markdown(f"### 📊 {t('backtest.results_title')}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(t("backtest.metrics_total_return"), f"{metrics.get('total_return', 0):.2f}%")
            k2.metric(t("backtest.metrics_sharpe"), f"{metrics.get('sharpe_ratio', 0):.2f}")
            k3.metric(t("backtest.metrics_max_drawdown"), f"{metrics.get('max_drawdown', 0):.2f}%")
            k4.metric(t("backtest.metrics_win_rate"), f"{metrics.get('win_rate', 0):.1f}%")

            st.markdown("---")
            st.markdown(f"**📈 {t('ui.price_signal_chart')}**")

            try:
                fetcher = YFinanceFetcher()
                df = fetcher.download_history(ticker, period=period, interval="1d")
            except Exception:
                df = pd.DataFrame()

            if not df.empty and equity_curve is not None:
                price_max = df["High"].max()
                price_range = df["High"].max() - df["Low"].min()
                top_annotation_y = price_max + price_range * 0.12

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.05,
                )

                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"],
                    name=t("ui.close_price"),
                ), row=1, col=1)

                buy_trades = [tr for tr in trades if tr.get("action") == "BUY"]
                sell_trades = [tr for tr in trades if tr.get("action") == "SELL"]

                annotations = []
                for tr in buy_trades:
                    ts = tr["timestamp"]
                    price = tr["price"]
                    fig.add_vline(x=ts, line=dict(color="green", width=2, dash="dot"), row=1, col=1)
                    annotations.append(dict(
                        x=ts, y=top_annotation_y,
                        text=f"BUY @ ${price:.2f}",
                        showarrow=True, arrowhead=2, arrowcolor="green",
                        arrowsize=1, arrowwidth=2, ax=0, ay=-40,
                        font=dict(color="green", size=10),
                        bgcolor="rgba(0,255,0,0.1)",
                    ))

                for tr in sell_trades:
                    ts = tr["timestamp"]
                    price = tr["price"]
                    fig.add_vline(x=ts, line=dict(color="red", width=2, dash="dot"), row=1, col=1)
                    annotations.append(dict(
                        x=ts, y=top_annotation_y,
                        text=f"SELL @ ${price:.2f}",
                        showarrow=True, arrowhead=2, arrowcolor="red",
                        arrowsize=1, arrowwidth=2, ax=0, ay=-40,
                        font=dict(color="red", size=10),
                        bgcolor="rgba(255,0,0,0.1)",
                    ))

                fig.add_trace(go.Scatter(
                    x=equity_curve.index, y=equity_curve.values,
                    mode="lines", name=t("ui.equity"),
                    line=dict(color="blue"),
                ), row=2, col=1)

                if "Volume" in df.columns:
                    fig.add_trace(go.Bar(
                        x=df.index, y=df["Volume"],
                        name=t("ui.volume"), marker_color="lightblue",
                    ), row=3, col=1)

                fig.update_layout(
                    annotations=annotations,
                    height=600, showlegend=True, template="plotly_white"
                )
                fig.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**📋 {t('ui.trade_records')}**")
            if trades:
                st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
            else:
                st.info(t("ui.no_trades"))


# ============================================================
# TAB 1: Paper Trading Mode
# ============================================================
with tab_paper:
    st.markdown(f"### 💹 {t('paper_trading.title')}")

    if "paper_bot_status" not in st.session_state:
        st.session_state["paper_bot_status"] = {"is_running": False, "total_checks": 0, "total_trades": 0}
    if "paper_positions" not in st.session_state:
        st.session_state["paper_positions"] = {}
    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []
    if "paper_cash" not in st.session_state:
        st.session_state["paper_cash"] = 100000.0

    paper_tab1, paper_tab2, paper_tab3, paper_tab4 = st.tabs([
        f"🤖 {t('paper_trading.bot_control')}",
        t("paper_trading.signal_monitor"),
        t("risk.status_title"),
        t("paper_trading.positions"),
    ])

    with paper_tab1:
        st.markdown(f"**{t('paper_trading.bot_control')}**")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            paper_tickers_input = st.text_input(
                t("paper_trading.monitor_tickers"), value="AAPL,MSFT,GOOGL", key="pt_tickers",
            )
            paper_tickers = [tk.strip().upper() for tk in paper_tickers_input.split(",") if tk.strip()]
            paper_interval = st.select_slider(
                t("paper_trading.monitor_interval"),
                options=[1, 5, 15, 30, 60], value=15, key="pt_interval",
                format_func=lambda x: f"{x} {t('ui.minutes')}",
            )

        with col_right:
            status = st.session_state["paper_bot_status"]
            is_running = status.get("is_running", False)
            st.markdown(f"**{t('ui.current_status')}:** {'🟢 ' + t('ui.running') if is_running else '🔴 ' + t('ui.stopped')}")
            st.metric(t("paper_trading.check_count"), status.get("total_checks", 0))
            st.metric(t("paper_trading.trade_count"), status.get("total_trades", 0))

        pt_btn1, pt_btn2 = st.columns(2)
        with pt_btn1:
            if st.button(f"▶️ {t('ui.start_monitor')}", disabled=is_running, key="pt_start", use_container_width=True):
                try:
                    from data.fetcher import YFinanceFetcher
                    from trading.paper import PaperTrader
                    from strategy.composite import CompositeStrategy
                    from risk.manager import RiskManager
                    from trading.bot import create_bot

                    fetcher = YFinanceFetcher()
                    trader = PaperTrader(initial_cash=st.session_state.get("paper_cash", 100000.0))
                    strategy = CompositeStrategy()
                    risk_mgr = RiskManager()
                    bot = create_bot(
                        trader=trader,
                        fetcher=fetcher,
                        strategy=strategy,
                        tickers=paper_tickers,
                        interval_minutes=paper_interval,
                        risk_manager=risk_mgr,
                    )
                    bot.start()
                    st.session_state["paper_bot_status"]["is_running"] = True
                    st.success(t("ui.bot_started"))
                except Exception as e:
                    st.error(f"{t('common.error')}: {e}")

        with pt_btn2:
            if st.button(f"⏹️ {t('ui.stop_monitor')}", disabled=not is_running, key="pt_stop", use_container_width=True):
                try:
                    from trading.bot import get_bot
                    bot = get_bot()
                    if bot:
                        bot.stop()
                    st.session_state["paper_bot_status"]["is_running"] = False
                    st.success(t("ui.bot_stopped"))
                except Exception as e:
                    st.error(f"{t('common.error')}: {e}")
                st.rerun()

    with paper_tab2:
        st.markdown(f"**{t('paper_trading.signal_monitor')}**")
        st.info(t("paper_trading.manual_check_hint"))

    with paper_tab3:
        st.markdown(f"**{t('risk.status_title')}**")
        st.metric(t("risk.per_trade"), f"{risk_pct:.1%}" if 'risk_pct' in dir() else "2.0%")
        st.metric(t("risk.stop_loss_atr"), "2.0x ATR")
        st.metric(t("risk.take_profit_atr"), "3.0x ATR")

    with paper_tab4:
        st.markdown(f"**{t('paper_trading.positions')}**")
        positions = st.session_state.get("paper_positions", {})
        cash = st.session_state.get("paper_cash", 100000.0)
        st.metric(t("paper_trading.available_cash"), f"${cash:,.2f}")

        if positions:
            pos_df = pd.DataFrame([
                {t("ui.ticker"): tk, t("ui.quantity"): p.get("qty", 0), t("ui.avg_cost"): f"${p.get('avg_cost', 0):.2f}"}
                for tk, p in positions.items()
            ])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info(t("paper_trading.no_positions"))


# ============================================================
# TAB 2: Live Trading Mode
# ============================================================
with tab_live:
    st.markdown(f"### 🔴 {t('app.mode_live')}")
    st.warning(t("ui.live_warning"))

    st.markdown(f"**{t('ui.live_checklist')}**")
    st.markdown(f"1. ✅ {t('ui.check_item_1')}")
    st.markdown(f"2. ✅ {t('ui.check_item_2')}")
    st.markdown(f"3. ✅ {t('ui.check_item_3')}")
    st.markdown(f"4. ✅ {t('ui.check_item_4')}")
    st.markdown(f"5. ✅ {t('ui.check_item_5')}")

    if st.button(t("ui.connect_live"), key="live_connect"):
        st.error(t("ui.please_config_moomoo"))


# ============================================================
# TAB 3: AI Analysis Mode
# ============================================================
with tab_ai:
    st.markdown(f"### 🤖 {t('ai_analysis.title')}")

    if "macro_scan_done" not in st.session_state:
        st.session_state["macro_scan_done"] = False
    if "macro_scan_result" not in st.session_state:
        st.session_state["macro_scan_result"] = None

    def _render_env_banner(macro_result):
        if macro_result is None:
            return
        score = macro_result.environment.environment_score
        risk = macro_result.environment.risk_appetite
        if risk == "Risk-Off" or score <= 3:
            st.error(f"⚠️ **{t('ui.environment_fail')}** — {t('ui.macro_high_risk')}")
        elif score <= 5 or risk == "Neutral":
            st.warning(f"⚠️ **{t('ui.environment_warn')}** — {t('ui.macro_caution')}")

    if st.session_state["macro_scan_result"] is not None:
        _render_env_banner(st.session_state["macro_scan_result"])

    ai_tab0, ai_tab1, ai_tab2, ai_tab3 = st.tabs([
        f"🌐 {t('ui.tab_macro')}",
        f"📊 {t('ui.tab_market')}",
        f"🔍 {t('ui.tab_screener')}",
        f"📋 {t('ui.tab_report')}",
    ])

    with ai_tab0:
        st.markdown(f"#### {t('ui.macro_scan')}")
        st.caption(t("ui.macro_scan_desc"))

        if st.button(f"🔍 {t('ui.scan_now')}", key="macro_scan_btn"):
            with st.spinner(t("ui.scanning")):
                try:
                    from ai.macro_scanner import MacroScanner
                    scanner = MacroScanner()
                    scan_result = scanner.scan()
                    st.session_state["macro_scan_result"] = scan_result
                    st.session_state["macro_scan_done"] = True

                    # 核心指标
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric(t("ui.overall_score"), f"{scan_result.environment.macro_env_score}/10")
                    sc2.metric(t("ui.macro_regime"), scan_result.environment.regime)
                    sc3.metric(t("ui.macro_confidence"), t(f"confidence_{scan_result.environment.confidence}"))
                    sc4.metric("VIX", f"{scan_result.vix:.2f} ({scan_result.environment.vix_signal})")

                    # 概率分布
                    st.markdown(f"#### {t('ui.macro_probability')}")
                    prob_col1, prob_col2, prob_col3 = st.columns(3)
                    prob_col1.metric(f"{t('ui.macro_p_risk_on')}", f"{scan_result.environment.P_risk_on*100:.1f}%")
                    prob_col2.metric(f"{t('ui.macro_p_neutral')}", f"{scan_result.environment.P_neutral*100:.1f}%")
                    prob_col3.metric(f"{t('ui.macro_p_risk_off')}", f"{scan_result.environment.P_risk_off*100:.1f}%")

                    # 主导因子
                    if scan_result.environment.key_drivers:
                        st.markdown(f"#### 🔑 {t('ui.macro_key_drivers')}")
                        st.markdown(", ".join(scan_result.environment.key_drivers))

                    # 风险预警
                    if scan_result.environment.warnings:
                        st.markdown(f"#### ⚠️ {t('ui.macro_warnings')}")
                        for w in scan_result.environment.warnings:
                            st.warning(w)
                    else:
                        st.info(f"✅ {t('ui.no_warnings')}")

                    # 模块分数
                    with st.expander(f"📊 {t('ui.macro_module_scores') if t('ui.macro_module_scores') != 'ui.macro_module_scores' else '模块评分详情'}", expanded=False):
                        mod_col1, mod_col2 = st.columns(2)
                        with mod_col1:
                            for mod, score in list(scan_result.environment.module_scores.items())[:4]:
                                st.metric(f"{t(f'ui.macro_module_{mod}') if t(f'ui.macro_module_{mod}') != f'ui.macro_module_{mod}' else mod}", f"{score:.2f}")
                        with mod_col2:
                            for mod, score in list(scan_result.environment.module_scores.items())[4:]:
                                st.metric(f"{t(f'ui.macro_module_{mod}') if t(f'ui.macro_module_{mod}') != f'ui.macro_module_{mod}' else mod}", f"{score:.2f}")

                    st.markdown(f"### 📝 {t('ui.macro_summary')}")
                    st.markdown(scan_result.environment.summary)

                    if scan_result.index_results:
                        st.markdown("#### 📊 指数趋势")
                        idx_data = []
                        for r in scan_result.index_results:
                            idx_data.append({
                                "指数": f"{r.ticker} ({r.name})",
                                "趋势": r.trend,
                                "周涨幅": f"{r.ret_5d:+.2f}%",
                                "月涨幅": f"{r.ret_20d:+.2f}%",
                                "量比": f"{r.volume_ratio:.2f}",
                            })
                        st.dataframe(pd.DataFrame(idx_data), use_container_width=True, hide_index=True)

                    if scan_result.haven_results:
                        st.markdown("#### 🛡️ 避险资产")
                        haven_data = []
                        for r in scan_result.haven_results:
                            haven_data.append({
                                "资产": f"{r.ticker} ({r.name})",
                                "趋势": r.trend,
                                "周涨幅": f"{r.ret_5d:+.2f}%",
                                "月涨幅": f"{r.ret_20d:+.2f}%",
                                "异动": "⚠️ 是" if r.is_abnormal else "否",
                            })
                        st.dataframe(pd.DataFrame(haven_data), use_container_width=True, hide_index=True)

                    # 信用/流动性分析
                    if scan_result.credit_result:
                        st.markdown(f"#### 💳 {t('ui.macro_credit_analysis')}")
                        credit = scan_result.credit_result
                        credit_data = [{
                            t('ui.macro_hyg_lqd'): f"{credit.hyg_lqd_ratio:.3f}",
                            t('ui.macro_yield_curve'): f"{credit.spread_10y_2y:.2f}%",
                            t('ui.macro_curve_status'): credit.curve_status,
                            "评分": f"{credit.credit_score:.2f}",
                        }]
                        st.dataframe(pd.DataFrame(credit_data), use_container_width=True, hide_index=True)

                    # 市场广度分析
                    if scan_result.breadth_result:
                        st.markdown(f"#### 📈 {t('ui.macro_breadth_analysis')}")
                        breadth = scan_result.breadth_result
                        breadth_data = [{
                            t('ui.macro_rsp_vs_spy'): f"{breadth.rsp_vs_spy_diff:+.2f}%",
                            "RSP 20d": f"{breadth.rsp_ret_20d:+.2f}%",
                            "SPY 20d": f"{breadth.spy_ret_20d:+.2f}%",
                            t('ui.macro_breadth_signal'): breadth.breadth_signal,
                        }]
                        st.dataframe(pd.DataFrame(breadth_data), use_container_width=True, hide_index=True)

                except Exception as e:
                    import traceback
                    st.error(f"{t('ui.macro_no_data')}: {e}")
                    st.code(traceback.format_exc())

    with ai_tab1:
        st.markdown(f"#### {t('ai_analysis.market_analyze')}")

        if st.button(f"🔄 {t('ui.analyze_current_market')}", key="analyze_market"):
            with st.spinner(t("ui.analyzing_market")):
                try:
                    from ai.market_analyzer import MarketAnalyzer
                    analyzer = MarketAnalyzer()
                    state = analyzer.analyze()

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(t("ai_analysis.trend"), state.trend)
                    m2.metric(t("ai_analysis.volatility"), state.volatility)
                    m3.metric(t("ai_analysis.momentum"), state.momentum)
                    m4.metric(t("ai_analysis.risk_level"), f"{state.risk_level}/10")

                    rec_str = ', '.join(state.recommended_strategies) if state.recommended_strategies else ''
                    st.markdown(f"**{t('ui.recommended_strategies')}:** {rec_str}")
                    st.info(state.analysis_text)

                    if state.analysis_steps:
                        with st.expander(f"📝 {t('ui.analysis_steps')}", expanded=False):
                            for step in state.analysis_steps:
                                st.markdown(f"**{step.step_name}**")
                                st.caption(f"输入: {step.input_data}")
                                st.markdown(f"计算: `{step.calculation}`")
                                st.markdown(f"结果: {step.result}")
                                st.markdown(f"推理: {step.reasoning}")
                                st.markdown("---")

                    if state.raw_data:
                        with st.expander(f"📊 {t('ui.raw_data')}", expanded=False):
                            st.json(state.raw_data)

                except Exception as e:
                    import traceback
                    st.error(f"{t('ui.macro_no_data')}: {e}")
                    st.code(traceback.format_exc())

    with ai_tab2:
        st.markdown(f"#### {t('ai_analysis.screener_pool')}")

        macro_result = st.session_state.get("macro_scan_result")
        if macro_result is not None and macro_result.environment.environment_score < 4:
            st.warning(f"⚠️ {t('ui.environment_warn')} — {t('ui.macro_caution')}")

        col1, col2 = st.columns([2, 1])
        with col1:
            pool = st.selectbox(
                t("ui.stock_pool"), ["popular", "tech", "etf"],
                format_func=lambda x: {"popular": t("ui.popular_stocks"), "tech": t("ui.tech_stocks"), "etf": "ETF"}[x],
                key="ai_pool",
            )
        with col2:
            top_n = st.number_input(t("ai_analysis.screener_top_n"), min_value=5, max_value=50, value=10, key="ai_topn")

        if st.button(f"🔍 {t('ai_analysis.screener_run')}", key="screen_stocks"):
            with st.spinner(t("ui.screening")):
                try:
                    from ai.stock_screener import StockScreener
                    screener = StockScreener()
                    candidates = screener.screen(pool_name=pool, top_n=top_n)

                    if candidates:
                        results = []
                        for c in candidates:
                            results.append({
                                t("ui.ticker"): c.ticker,
                                t("ui.score"): f"{c.score:.0f}",
                                t("ui.signal"): c.signal,
                                t("ui.price"): f"${c.price:.2f}",
                                t("ui.reason"): c.reason,
                            })
                        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                    else:
                        st.warning(t("ui.no_condition"))
                except Exception as e:
                    import traceback
                    st.error(f"{t('ui.macro_no_data')}: {e}")
                    st.code(traceback.format_exc())

    with ai_tab3:
        st.markdown(f"#### {t('ui.daily_report')}")
        watchlist = st.text_area(
            t("ai_analysis.report_watchlist"),
            value="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA", height=68, key="ai_watchlist",
        )

        if st.button(f"📋 {t('ai_analysis.report_generate')}", key="generate_report"):
            with st.spinner(t("ui.analyzing_market")):
                try:
                    from ai.orchestrator import AIQuantAnalyst
                    import yaml

                    config_path = BASE_DIR / "config.yaml"
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)

                    analyst = AIQuantAnalyst(config=config)
                    tickers_list = [tk.strip().upper() for tk in watchlist.split(",") if tk.strip()]
                    report = analyst.run_daily_analysis(watchlist=tickers_list, generate_report=False)

                    st.markdown(f"**{t('ui.date')}: {report.date}**")
                    ms = report.market_state
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric(t("ai_analysis.trend"), ms.get("trend", "unknown"))
                    r2.metric(t("ai_analysis.volatility"), ms.get("volatility", "unknown"))
                    r3.metric(t("ai_analysis.momentum"), ms.get("momentum", "unknown"))
                    r4.metric(t("ai_analysis.risk_level"), f"{ms.get('risk_level', 5)}/10")

                    recommended = ms.get("recommended_strategies", [])
                    if recommended:
                        rec_str = ', '.join(recommended)
                        st.markdown(f"**{t('ui.recommended_strategies')}:** {rec_str}")

                    analysis = ms.get("analysis", "")
                    if analysis:
                        st.info(analysis)

                    if report.top_opportunities:
                        st.markdown(f"### 🔥 {t('ui.top_opportunities')}")
                        opp_data = []
                        for opp in report.top_opportunities:
                            opp_data.append({
                                "标的": opp.get("ticker", ""),
                                "得分": f"{opp.get('score', 0):.0f}",
                                "信号": opp.get("signal", ""),
                                "价格": f"${opp.get('price', 0):.2f}",
                                "原因": opp.get("reason", ""),
                            })
                        st.dataframe(pd.DataFrame(opp_data), use_container_width=True, hide_index=True)

                    if report.strategy_recommendations:
                        st.markdown(f"### 📊 {t('ui.strategy_recommendations')}")
                        rec_data = []
                        for rec in report.strategy_recommendations:
                            rec_data.append({
                                "标的": rec.get("ticker", ""),
                                "推荐策略": ", ".join(rec.get("strategies", [])),
                                "原因": rec.get("reason", ""),
                            })
                        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

                    if report.risk_alerts:
                        st.markdown(f"### ⚠️ {t('ui.risk_alerts')}")
                        for alert in report.risk_alerts:
                            st.warning(alert)

                    if report.ai_summary:
                        st.markdown(f"### 🤖 {t('ui.summary')}")
                        st.markdown(report.ai_summary)

                except Exception as e:
                    import traceback
                    st.error(f"{t('ui.macro_no_data')}: {e}")
                    st.code(traceback.format_exc())


# ============================================================
# TAB 4: Optimize Mode
# ============================================================
with tab_optimize:
    st.markdown(f"### ⚙️ {t('optimization.title')}")

    o1, o2, o3 = st.columns(3)
    with o1:
        opt_strategy = st.selectbox(
            t("ui.optimize_strategy"),
            ["MA", "RSI", "MACD", "Bollinger", "Composite"], key="opt_strat",
        )
    with o2:
        opt_ticker = st.text_input(t("ui.target"), value="AAPL", key="opt_ticker").upper()
    with o3:
        opt_method = st.selectbox(
            t("ui.optimize_method"),
            ["grid", "random", "bayesian"], key="opt_method",
            format_func=lambda x: {"grid": t("ui.grid_search"), "random": t("ui.random_search"), "bayesian": t("ui.bayesian")}[x],
        )

    opt_trials = st.number_input(t("optimization.n_trials"), min_value=10, max_value=500, value=50, key="opt_trials")
    opt_cash = st.number_input(t("backtest.initial_cash"), min_value=10000, max_value=10000000, value=100000, step=10000, key="opt_cash")

    if st.button(f"🚀 {t('optimization.run_optimization')}", key="btn_run_optimize", use_container_width=True):
        with st.spinner(t("ui.optimizing")):
            try:
                from data.fetcher import YFinanceFetcher
                from backtest.engine import SimpleBacktester
                from optimization.optimizer import StrategyOptimizer

                fetcher = YFinanceFetcher()
                df = fetcher.download_history(opt_ticker, period="2y", interval="1d")
                if df.empty:
                    st.error(t("ui.no_data"))
                    st.stop()

                backtester = SimpleBacktester(initial_cash=opt_cash)
                optimizer = StrategyOptimizer(data_fetcher=fetcher, backtester=backtester, scoring="sharpe")

                result = optimizer.optimize(
                    strategy_name=opt_strategy,
                    ticker=opt_ticker,
                    method=opt_method,
                    n_trials=opt_trials,
                )

                if result:
                    st.success(f"✅ {t('optimization.complete')}")
                    st.json(result.best_params)

                    m1, m2, m3 = st.columns(3)
                    m1.metric(t("backtest.metrics_sharpe"), f"{result.best_score:.2f}")
                    m2.metric(t("optimization.method"), opt_method)
                    m3.metric(t("optimization.n_trials"), result.n_trials)

                    if result.all_results:
                        st.markdown(f"### 📋 {t('optimization.results_table')}")
                        st.dataframe(pd.DataFrame(result.all_results), use_container_width=True)
                else:
                    st.error(t("ui.optimize_failed"))

            except Exception as e:
                import traceback
                st.error(f"{t('common.error')}: {e}")
                st.code(traceback.format_exc())


# ============================================================
# TAB 5: Settings Mode (with language select)
# ============================================================
with tab_settings:
    st.markdown(f"### 🔧 {t('app.mode_settings')}")

    # ---- Deploy to Cloud ----
    st.markdown(f"#### ☁️ {t('settings.deploy_title')}")
    st.caption(t("settings.deploy_desc"))
    st.link_button(f"🚀 {t('settings.deploy_button')}", "https://streamlit.io/cloud")
    st.markdown("---")

    # Language selector IN the settings tab
    st.markdown(f"#### 🌍 {t('app.language_select')}")
    lang_options = I18nManager.SUPPORTED_LANGUAGES
    lang_keys = list(lang_options.keys())
    lang_default = st.session_state.get("language", "zh")
    lang_index = lang_keys.index(lang_default) if lang_default in lang_keys else 1
    sel_lang = st.selectbox(
        t("app.language_select"),
        options=lang_keys,
        format_func=lambda x: lang_options[x],
        index=lang_index, key="lang_sel",
    )
    if st.session_state.get("language") != sel_lang:
        st.session_state["language"] = sel_lang
        st.rerun()

    st.markdown("---")

    # Config settings
    import yaml
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    set_tab1, set_tab2, set_tab3 = st.tabs([
        t("settings.tab_general"),
        t("settings.tab_risk"),
        t("settings.tab_notification"),
    ])

    with set_tab1:
        st.markdown(f"#### 📊 {t('settings.general_conf')}")

        new_initial_cash = st.number_input(
            t("backtest.initial_cash"),
            min_value=10000, max_value=10000000,
            value=config.get("backtest", {}).get("initial_capital", 100000), step=10000, key="set_cash",
        )
        new_commission = st.number_input(
            t("backtest.commission"),
            min_value=0.0, max_value=0.01,
            value=config.get("backtest", {}).get("commission", 0.001), step=0.0001, format="%.4f", key="set_comm",
        )
        new_default_strategy = st.selectbox(
            t("settings.default_strategy"),
            ["MA", "RSI", "MACD", "Bollinger", "Composite"], index=4, key="set_def_strat",
        )

    with set_tab2:
        st.markdown(f"#### 🛡️ {t('risk.title')}")

        new_max_position = st.slider(
            t("settings.max_position"),
            0.05, 0.5,
            config.get("risk", {}).get("max_position_size", 0.20), key="set_max_pos",
        )
        new_stop_atr = st.number_input(
            t("risk.stop_loss_atr"),
            1.0, 5.0,
            config.get("risk", {}).get("atr_stop_loss_mult", 2.0), key="set_stop_atr",
        )
        new_tp_atr = st.number_input(
            t("risk.take_profit_atr"),
            1.0, 10.0,
            config.get("risk", {}).get("atr_take_profit_mult", 3.0), key="set_tp_atr",
        )
        new_daily_loss = st.number_input(
            t("settings.daily_loss_limit"),
            0.01, 0.10,
            config.get("risk", {}).get("daily_loss_limit", 0.02), format="%.2f", key="set_daily_loss",
        )

    with set_tab3:
        st.markdown(f"#### 📧 {t('notification.title')}")

        email_enabled = st.checkbox(
            t("notification.email_enabled"),
            value=config.get("notifications", {}).get("email", {}).get("enabled", False), key="set_email_en",
        )
        tg_enabled = st.checkbox(
            t("notification.telegram_enabled"),
            value=config.get("notifications", {}).get("telegram", {}).get("enabled", False), key="set_tg_en",
        )

    st.markdown("---")
    if st.button(f"💾 {t('settings.save_conf')}", key="set_save_btn", use_container_width=True):
        if "backtest" not in config:
            config["backtest"] = {}
        if "risk" not in config:
            config["risk"] = {}
        if "notifications" not in config:
            config["notifications"] = {}

        config["backtest"]["initial_capital"] = new_initial_cash
        config["backtest"]["commission"] = new_commission
        config["risk"]["max_position_size"] = new_max_position
        config["risk"]["atr_stop_loss_mult"] = new_stop_atr
        config["risk"]["atr_take_profit_mult"] = new_tp_atr
        config["risk"]["daily_loss_limit"] = new_daily_loss

        if "email" not in config["notifications"]:
            config["notifications"]["email"] = {}
        config["notifications"]["email"]["enabled"] = email_enabled

        if "telegram" not in config["notifications"]:
            config["notifications"]["telegram"] = {}
        config["notifications"]["telegram"]["enabled"] = tg_enabled

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)

        st.success(t("settings.saved"))

    with st.expander(t("settings.view_conf")):
        st.json(config)
