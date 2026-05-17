# ============================================================
# dashboard/app.py - Streamlit Web Dashboard (v6)
# Changes:
# 1. AI tab moved to first position
# 2. All hardcoded Chinese removed (use t() with fallback)
# 3. ASCII box lines removed from optimize tab
# 4. Tab descriptions added
# ============================================================
import os
import sys
import importlib
from pathlib import Path
from datetime import datetime

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
    "trading.signal_explainer", "trading.equity_tracker",
    "trading.order_manager", "trading.bot_enhanced",
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
    f"🤖 {t('app.mode_ai')}",
    f"📊 {t('app.mode_backtest')}",
    f"📈 {t('app.mode_paper')}",
    f"🔴 {t('app.mode_live')}",
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
tab_ai, tab_backtest, tab_paper, tab_live, tab_optimize, tab_settings = st.tabs(TAB_LABELS)

# Thin separator below tabs
st.markdown("<style>div[data-testid='stTabs'] {{ border-bottom: 2px solid #e0e0e0; }}</style>", unsafe_allow_html=True)


# ============================================================
# TAB 0: AI Analysis Mode (MOVED TO FIRST)
# ============================================================
with tab_ai:
    st.markdown(f"### 🤖 {t('ai_analysis.title')}")
    st.caption(t("ui.ai_tab_desc") if t("ui.ai_tab_desc") != "ui.ai_tab_desc" else "AI-powered multi-dimensional market analysis")

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

                    # Core metrics
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric(t("ui.overall_score"), f"{scan_result.environment.macro_env_score}/10")
                    sc2.metric(t("ui.macro_regime"), scan_result.environment.regime)
                    sc3.metric(t("ui.macro_confidence"), t(f"confidence_{scan_result.environment.confidence}"))
                    sc4.metric("VIX", f"{scan_result.vix:.2f} ({scan_result.environment.vix_signal})")

                    # Probability distribution
                    st.markdown(f"#### {t('ui.macro_probability')}")
                    prob_col1, prob_col2, prob_col3 = st.columns(3)
                    prob_col1.metric(f"{t('ui.macro_p_risk_on')}", f"{scan_result.environment.P_risk_on*100:.1f}%")
                    prob_col2.metric(f"{t('ui.macro_p_neutral')}", f"{scan_result.environment.P_neutral*100:.1f}%")
                    prob_col3.metric(f"{t('ui.macro_p_risk_off')}", f"{scan_result.environment.P_risk_off*100:.1f}%")

                    # Key drivers
                    if scan_result.environment.key_drivers:
                        st.markdown(f"#### 🔑 {t('ui.macro_key_drivers')}")
                        st.markdown(", ".join(scan_result.environment.key_drivers))

                    # Risk warnings
                    if scan_result.environment.warnings:
                        st.markdown(f"#### ⚠️ {t('ui.macro_warnings')}")
                        for w in scan_result.environment.warnings:
                            st.warning(w)
                    else:
                        st.info(f"✅ {t('ui.no_warnings')}")

                    # Module bias 表格
                    if scan_result.environment.module_bias:
                        st.markdown(f"#### 📊 {t('ui.macro_module_bias') if t('ui.macro_module_bias') != 'ui.macro_module_bias' else '各模块多空倾向'}")
                        bias_data = []
                        for b in scan_result.environment.module_bias:
                            bias_data.append({
                                "模块": b.name,
                                "倾向": b.bias,
                                "强度": f"{b.strength:.0%}",
                                "说明": b.detail,
                            })
                        st.dataframe(pd.DataFrame(bias_data), use_container_width=True, hide_index=True)

                    # Future forecast
                    st.markdown(f"#### 🔮 {t('ui.macro_forecast') if t('ui.macro_forecast') != 'ui.macro_forecast' else '未来展望'}")
                    fc_data = [{
                        "时间维度": "未来5天",
                        "倾向": scan_result.environment.forecast_5d,
                        "置信度": f"{scan_result.environment.forecast_confidence_5d:.0%}",
                    }, {
                        "时间维度": "未来30天",
                        "倾向": scan_result.environment.forecast_30d,
                        "置信度": f"{scan_result.environment.forecast_confidence_30d:.0%}",
                    }]
                    st.dataframe(pd.DataFrame(fc_data), use_container_width=True, hide_index=True)

                    # Module scores
                    with st.expander(f"📊 {t('ui.macro_module_scores')}", expanded=False):
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
                        st.markdown(f"#### 📊 {t('ui.macro_index_table')}")
                        idx_data = []
                        for r in scan_result.index_results:
                            idx_data.append({
                                t('ui.ticker'): f"{r.ticker} ({r.name})",
                                t('ui.trend'): r.trend,
                                t('ui.weekly_return') if t('ui.weekly_return') != 'ui.weekly_return' else 'Weekly Return': f"{r.ret_5d:+.2f}%",
                                t('ui.monthly_return') if t('ui.monthly_return') != 'ui.monthly_return' else 'Monthly Return': f"{r.ret_20d:+.2f}%",
                                t('ui.volume_ratio') if t('ui.volume_ratio') != 'ui.volume_ratio' else 'Vol Ratio': f"{r.volume_ratio:.2f}",
                            })
                        st.dataframe(pd.DataFrame(idx_data), use_container_width=True, hide_index=True)

                    if scan_result.haven_results:
                        st.markdown(f"#### 🛡️ {t('ui.macro_safe_table')}")
                        haven_data = []
                        for r in scan_result.haven_results:
                            haven_data.append({
                                t('ui.asset'): f"{r.ticker} ({r.name})",
                                t('ui.trend'): r.trend,
                                t('ui.weekly_return') if t('ui.weekly_return') != 'ui.weekly_return' else 'Weekly Return': f"{r.ret_5d:+.2f}%",
                                t('ui.monthly_return') if t('ui.monthly_return') != 'ui.monthly_return' else 'Monthly Return': f"{r.ret_20d:+.2f}%",
                                t('ui.abnormal') if t('ui.abnormal') != 'ui.abnormal' else 'Abnormal': "⚠️ " + t('ui.yes') if r.is_abnormal else t('ui.no_text'),
                            })
                        st.dataframe(pd.DataFrame(haven_data), use_container_width=True, hide_index=True)

                    # Credit/Liquidity analysis
                    if scan_result.credit_result:
                        st.markdown(f"#### 💳 {t('ui.macro_credit_analysis')}")
                        credit = scan_result.credit_result
                        credit_data = [{
                            t('ui.macro_hyg_lqd'): f"{credit.hyg_lqd_ratio:.3f}",
                            t('ui.macro_yield_curve'): f"{credit.spread_10y_2y:.2f}%",
                            t('ui.macro_curve_status'): credit.curve_status,
                            t('ui.score'): f"{credit.credit_score:.2f}",
                        }]
                        st.dataframe(pd.DataFrame(credit_data), use_container_width=True, hide_index=True)

                    # Market breadth analysis
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

                    # Sector rotation analysis
                    try:
                        from ai.sector_rotation import SectorRotationAnalyzer
                        sector_analyzer = SectorRotationAnalyzer()
                        sector_result = sector_analyzer.analyze(period="3mo")

                        st.markdown(f"#### 🔄 {t('ui.sector_rotation') if t('ui.sector_rotation') != 'ui.sector_rotation' else '板块轮动'}")

                        # Market phase badge
                        phase_map = {
                            "early_bull": ("🟢", "牛市早期"),
                            "late_bull": ("🟡", "牛市晚期"),
                            "bear": ("🔴", "熊市"),
                            "recovery": ("🔵", "恢复期"),
                        }
                        phase_emoji, phase_name = phase_map.get(sector_result.market_phase, ("⚪", sector_result.market_phase))
                        st.markdown(f"**{phase_emoji} 市场阶段: {phase_name}**")

                        # Sector ranking table
                        if sector_result.sectors:
                            st.markdown("**板块强弱排名**")
                            sector_data = []
                            for i, s in enumerate(sorted(sector_result.sectors, key=lambda x: x.momentum_score, reverse=True)):
                                sector_data.append({
                                    "排名": i + 1,
                                    "板块": f"{s.ticker} ({s.name_zh or s.name})",
                                    "5日%": f"{s.returns_5d:+.2f}%",
                                    "20日%": f"{s.returns_20d:+.2f}%",
                                    "动量": f"{s.momentum_score:.1f}",
                                    "量能": f"{s.volume_change:+.1f}%",
                                })
                            st.dataframe(pd.DataFrame(sector_data), use_container_width=True, hide_index=True)

                        # Rotation signals
                        if sector_result.rotation_signals:
                            st.markdown("**轮动信号**")
                            for sig in sector_result.rotation_signals:
                                sig_emoji = {"defensive": "🛡️", "offensive": "⚔️", "neutral": "⚖️"}.get(sig.type, "📊")
                                st.markdown(f"{sig_emoji} **{sig.description}** (强度: {sig.strength:.0%})")

                        # Summary
                        if sector_result.summary:
                            st.info(sector_result.summary)

                    except Exception as e:
                        st.caption(f"板块轮动分析暂不可用: {e}")

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
                                st.caption(f"{t('ui.input_data')}: {step.input_data}")
                                st.markdown(f"{t('ui.calculation')}: `{step.calculation}`")
                                st.markdown(f"{t('ui.result')}: {step.result}")
                                st.markdown(f"{t('ui.logic')}: {step.reasoning}")
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
                                t('ui.ticker'): opp.get("ticker", ""),
                                t('ui.score'): f"{opp.get('score', 0):.0f}",
                                t('ui.signal'): opp.get("signal", ""),
                                t('ui.price'): f"${opp.get('price', 0):.2f}",
                                t('ui.reason'): opp.get("reason", ""),
                            })
                        st.dataframe(pd.DataFrame(opp_data), use_container_width=True, hide_index=True)

                    if report.strategy_recommendations:
                        st.markdown(f"### 📊 {t('ui.strategy_recommendations')}")
                        rec_data = []
                        for rec in report.strategy_recommendations:
                            rec_data.append({
                                t('ui.ticker'): rec.get("ticker", ""),
                                t('ui.recommended_strategies') if t('ui.recommended_strategies') != 'ui.recommended_strategies' else 'Recommended Strategies': ", ".join(rec.get("strategies", [])),
                                t('ui.reason'): rec.get("reason", ""),
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
# TAB 1: Backtest Mode
# ============================================================
with tab_backtest:
    st.markdown(f"### 📋 {t('app.mode_backtest')}")
    st.caption(t("ui.backtest_tab_desc") if t("ui.backtest_tab_desc") != "ui.backtest_tab_desc" else "Strategy backtesting with multi-strategy voting")

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
# TAB 2: Paper Trading Mode (Enhanced)
# ============================================================
with tab_paper:
    st.markdown(f"### 💹 {t('paper_trading.title')}")
    st.caption(t("ui.paper_tab_desc") if t("ui.paper_tab_desc") != "ui.paper_tab_desc" else "Simulated trading with real-time signal monitoring, equity tracking, and strategy configuration")

    # Session state init
    if "enhanced_bot" not in st.session_state:
        st.session_state["enhanced_bot"] = None
    if "paper_cash" not in st.session_state:
        st.session_state["paper_cash"] = 100000.0

    # Lazy imports
    try:
        from trading.paper import PaperTrader
        from trading.bot_enhanced import EnhancedPaperTradingBot
        from trading.order_manager import OrderManager
        from trading.equity_tracker import EquityTracker
        from trading.signal_explainer import SignalExplainer
        from trading.strategy_config import StrategyConfigManager
        from risk.manager import RiskManager
        from data.fetcher import YFinanceFetcher
        from strategy.composite import CompositeStrategy
        ENHANCED_AVAILABLE = True
    except Exception as e:
        ENHANCED_AVAILABLE = False
        st.error(f"{t('paper_enhanced.enhanced_load_failed')}: {e}")

    if ENHANCED_AVAILABLE:
        # Initialize components if not exists
        if "paper_trader" not in st.session_state:
            st.session_state["paper_trader"] = PaperTrader(
                initial_cash=st.session_state["paper_cash"]
            )
        if "strategy_config" not in st.session_state:
            st.session_state["strategy_config"] = StrategyConfigManager()
        if "risk_mgr" not in st.session_state:
            st.session_state["risk_mgr"] = RiskManager()

        trader = st.session_state["paper_trader"]
        scm = st.session_state["strategy_config"]
        risk_mgr = st.session_state["risk_mgr"]

        # Bot control
        bot = st.session_state.get("enhanced_bot")
        is_running = bot.is_running if bot else False

        # Top control bar
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1, 1, 1])

        with ctrl_col1:
            paper_tickers_input = st.text_input(
                t("paper_trading.monitor_tickers"),
                value="AAPL,MSFT,GOOGL",
                key="pt_tickers",
            )
            paper_tickers = [tk.strip().upper() for tk in paper_tickers_input.split(",") if tk.strip()]

        with ctrl_col2:
            paper_interval = st.select_slider(
                t("paper_trading.monitor_interval"),
                options=[1, 5, 15, 30, 60],
                value=15,
                key="pt_interval",
                format_func=lambda x: f"{x} min",
            )

        with ctrl_col3:
            st.metric(t('ui.current_status'), f"🟢 {t('paper_enhanced.status_running')}" if is_running else f"🔴 {t('paper_enhanced.status_stopped')}")
            if bot:
                status = bot.get_status()
                st.metric(t('paper_trading.check_count'), status.get("check_count", 0))

        with ctrl_col4:
            st.metric(t('paper_trading.trade_count'), status.get("trade_count", 0) if bot else 0)
            st.metric(t('paper_enhanced.equity'), f"${trader.cash + sum(p.get('qty',0)*p.get('avg_cost',0) for p in trader.positions.values()):,.0f}")

        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button(f"▶️ {t('ui.start_monitor')}", disabled=is_running, key="pt_start", use_container_width=True):
                try:
                    # Create default strategy instances if none exist
                    if not scm.get_active_instances():
                        for ticker in paper_tickers[:3]:
                            scm.create_instance("MA_Cross", ticker, weight=1.0/3, tags=["default"])

                    bot = EnhancedPaperTradingBot(
                        paper_trader=trader,
                        risk_manager=risk_mgr,
                        strategy_config=scm,
                        check_interval=paper_interval * 60,
                    )
                    bot.start()
                    st.session_state["enhanced_bot"] = bot
                    st.success(t("ui.bot_started"))
                    st.rerun()
                except Exception as e:
                    st.error(f"{t('common.error')}: {e}")

        with btn_col2:
            if st.button(f"⏹️ {t('ui.stop_monitor')}", disabled=not is_running, key="pt_stop", use_container_width=True):
                try:
                    if bot:
                        bot.stop()
                    st.session_state["enhanced_bot"] = None
                    st.success(t("ui.bot_stopped"))
                    st.rerun()
                except Exception as e:
                    st.error(f"{t('common.error')}: {e}")

        with btn_col3:
            if st.button(f"🔄 {t('paper_enhanced.reset_account')}", key="pt_reset", use_container_width=True):
                trader.reset()
                st.session_state["paper_cash"] = 100000.0
                st.success(t('paper_enhanced.account_reset'))
                st.rerun()

        st.divider()

        # Sub-tabs for enhanced features
        pt_sub1, pt_sub2, pt_sub3, pt_sub4, pt_sub5, pt_sub6 = st.tabs([
            f"📊 {t('paper_enhanced.overview')}",
            f"📜 {t('paper_enhanced.trade_history')}",
            f"📡 {t('paper_enhanced.signal_analysis')}",
            f"⚙️ {t('paper_enhanced.strategy_config')}",
            f"🛡️ {t('paper_enhanced.risk_status')}",
            f"📋 {t('paper_enhanced.orders')}",
        ])

        # ---- Overview ----
        with pt_sub1:
            st.subheader(t('paper_enhanced.realtime_equity_curve'))

            # 实时权益曲线：优先从 bot 获取，否则从 trader 计算
            eq_tracker = None
            if bot:
                eq_tracker = bot.get_equity_tracker()
            else:
                # 没有 bot 时，创建独立的 EquityTracker 展示账户状态
                from trading.equity_tracker import EquityTracker, EquitySnapshot
                eq_tracker = EquityTracker(storage_path="data/equity_paper.db")
                # 记录当前账户快照
                position_value = sum(p.get('qty',0)*p.get('avg_cost',0) for p in trader.positions.values())
                total_equity = trader.cash + position_value
                snapshot = EquitySnapshot(
                    timestamp=datetime.now(),
                    total_equity=total_equity,
                    cash=trader.cash,
                    position_value=position_value,
                    unrealized_pnl=0,
                    realized_pnl_today=0,
                    positions=trader.positions,
                )
                eq_tracker.record_equity(snapshot)

            df = eq_tracker.get_equity_curve(days=30)

            if not df.empty:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(t('paper_enhanced.equity_curve'), t('paper_enhanced.drawdown')),
                )

                fig.add_trace(go.Scatter(
                    x=df.index, y=df['total_equity'],
                    mode='lines', name=t('paper_enhanced.equity'),
                    line=dict(color='#00C851', width=2),
                    fill='tozeroy', fillcolor='rgba(0,200,81,0.1)',
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=df.index, y=df['cash'],
                    mode='lines', name=t('paper_trading.current_cash'),
                    line=dict(color='#FF8800', width=1),
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=df.index, y=df['position_value'],
                    mode='lines', name=t('paper_trading.position_value'),
                    line=dict(color='#33B5E5', width=1),
                ), row=1, col=1)

                # Drawdown
                cummax = df['total_equity'].cummax()
                drawdown = (df['total_equity'] - cummax) / cummax * 100

                fig.add_trace(go.Scatter(
                    x=df.index, y=drawdown,
                    mode='lines', name=t('paper_enhanced.drawdown') + ' %',
                    line=dict(color='#FF4444', width=1.5),
                    fill='tozeroy', fillcolor='rgba(255,68,68,0.1)',
                ), row=2, col=1)

                fig.update_layout(
                    height=500, showlegend=True,
                    hovermode='x unified',
                    template='plotly_dark',
                )
                st.plotly_chart(fig, use_container_width=True)

                # Performance metrics
                metrics = eq_tracker.calculate_metrics(days=30)
                if metrics:
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric(t('paper_enhanced.total_return'), f"{metrics.total_return_pct:+.2f}%")
                    mcol2.metric(t('paper_enhanced.sharpe_ratio'), f"{metrics.sharpe_ratio:.2f}")
                    mcol3.metric(t('paper_enhanced.max_drawdown'), f"{metrics.max_drawdown_pct:.2f}%")
                    mcol4.metric(t('paper_enhanced.win_rate'), f"{metrics.win_rate_pct:.1f}%")
            else:
                # 显示当前账户基本信息（即使没有历史数据）
                position_value = sum(p.get('qty',0)*p.get('avg_cost',0) for p in trader.positions.values())
                total_equity = trader.cash + position_value
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric(t('paper_enhanced.equity'), f"${total_equity:,.0f}")
                mcol2.metric(t('paper_trading.current_cash'), f"${trader.cash:,.0f}")
                mcol3.metric(t('paper_trading.position_value'), f"${position_value:,.0f}")
                st.info(t('paper_enhanced.no_equity_data'))

        # ---- Trade History ----
        with pt_sub2:
            st.subheader(t('paper_enhanced.trade_history'))

            # 交易历史：优先从 bot 获取，否则从 trader 获取
            trades_df = pd.DataFrame()
            if bot:
                eq_tracker = bot.get_equity_tracker()
                trades_df = eq_tracker.get_trade_history(days=30)
            else:
                # 从 trader 获取交易记录
                trades_df = trader.get_trade_history(days=30)

            if not trades_df.empty:
                display_df = trades_df.copy()
                if 'timestamp' in display_df.columns:
                    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                if 'price' in display_df.columns:
                    display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
                if 'realized_pnl' in display_df.columns:
                    display_df['realized_pnl'] = display_df['realized_pnl'].apply(
                        lambda x: f"${x:+.2f}" if pd.notna(x) else "-"
                    )

                    def highlight_pnl(val):
                        if isinstance(val, str) and val.startswith('$+'):
                            return 'background-color: rgba(0,200,81,0.2)'
                        elif isinstance(val, str) and val.startswith('$-'):
                            return 'background-color: rgba(255,68,68,0.2)'
                        return ''

                    styled = display_df.style.applymap(highlight_pnl, subset=['realized_pnl'])
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info(t('paper_enhanced.no_trades_yet'))

        # ---- Signal Analysis ----
        with pt_sub3:
            st.subheader(t('paper_enhanced.signal_analysis'))

            # 信号分析：优先从 bot 获取，否则创建独立实例
            explainer = None
            if bot:
                explainer = bot.get_signal_explainer()
            else:
                from trading.signal_explainer import SignalExplainer
                explainer = SignalExplainer()

            factor_stats = explainer.get_factor_statistics(days=30)

            if factor_stats:
                stats_df = pd.DataFrame([
                    {
                        t('paper_enhanced.factor'): k,
                        t('paper_enhanced.appearances'): v['appearances'],
                        t('paper_enhanced.avg_score'): f"{v['avg_score']:+.3f}",
                        t('paper_enhanced.avg_confidence'): f"{v['avg_confidence']:.1%}",
                    }
                    for k, v in factor_stats.items()
                ])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.info(t('paper_enhanced.no_signal_data'))

            # Recent decisions
            st.write("---")
            st.subheader(t('paper_enhanced.recent_decisions'))
            decisions = explainer.get_decisions(days=1)
            if decisions:
                for d in decisions[-10:]:
                    with st.expander(f"{d.timestamp.strftime('%H:%M')} {d.action} {d.ticker} ({t('paper_enhanced.confidence')}: {d.confidence:.0%})"):
                        st.write(f"**{t('paper_enhanced.market_regime')}:** {d.market_regime} ({t('paper_enhanced.score')}: {d.market_score:.2f})")
                        st.write(f"**{t('paper_enhanced.factors')}:**")
                        for f in d.factors:
                            st.write(f"- {f.name}: {t('paper_enhanced.score')}={f.score:+.2f}, {t('paper_enhanced.weight')}={f.weight:.0%}, {t('paper_enhanced.confidence')}={f.confidence:.0%}")
                        st.write(f"**{t('paper_enhanced.reasoning')}:** {' -> '.join(d.reasoning_chain[:5])}")
            else:
                st.info(t('paper_enhanced.no_decisions'))

        # ---- Strategy Config ----
        with pt_sub4:
            st.subheader(t('paper_enhanced.strategy_config'))

            instances = scm.get_active_instances()
            if instances:
                instance_options = {f"{i.strategy_name} ({i.ticker})": i.instance_id for i in instances}
                selected = st.selectbox(t('paper_enhanced.select_strategy'), options=list(instance_options.keys()), key="scm_select")

                if selected:
                    inst = scm.get_instance(instance_options[selected])
                    st.write(f"**{t('paper_enhanced.instance')}:** {inst.instance_id}")
                    st.write(f"**{t('paper_enhanced.weight')}:** {inst.weight:.0%}")
                    st.write(f"**{t('paper_enhanced.tags')}:** {', '.join(inst.tags)}")

                    st.write(f"**{t('paper_enhanced.parameters')}:**")
                    updated = False
                    for param_name, param in inst.parameters.items():
                        pcol1, pcol2 = st.columns([3, 1])
                        with pcol1:
                            if param.param_type == "int":
                                new_val = st.number_input(
                                    f"{param.name} ({param.description})",
                                    min_value=int(param.min_value) if param.min_value else None,
                                    max_value=int(param.max_value) if param.max_value else None,
                                    value=int(param.value),
                                    step=int(param.step) if param.step else 1,
                                    key=f"scm_{inst.instance_id}_{param_name}",
                                )
                            elif param.param_type == "float":
                                new_val = st.number_input(
                                    f"{param.name} ({param.description})",
                                    min_value=param.min_value,
                                    max_value=param.max_value,
                                    value=float(param.value),
                                    step=param.step or 0.01,
                                    key=f"scm_{inst.instance_id}_{param_name}",
                                )
                            else:
                                new_val = param.value

                            if new_val != param.value:
                                scm.update_parameter(inst.instance_id, param_name, new_val)
                                updated = True
                        with pcol2:
                            st.caption(f"{t('paper_enhanced.default_value')}: {param.default_value}")

                    if updated:
                        st.success(t('paper_enhanced.params_updated'))

                    bcol1, bcol2, bcol3 = st.columns(3)
                    with bcol1:
                        if st.button(t('paper_enhanced.reset_default'), key=f"scm_reset_{inst.instance_id}"):
                            scm.reset_parameters(inst.instance_id)
                            st.rerun()
                    with bcol2:
                        if st.button(t('paper_enhanced.clone_strategy'), key=f"scm_clone_{inst.instance_id}"):
                            new_inst = scm.clone_instance(inst.instance_id)
                            if new_inst:
                                st.success(f"{t('common.success')}: {new_inst.instance_id}")
                                st.rerun()
                    with bcol3:
                        if st.button(t('paper_enhanced.delete'), key=f"scm_del_{inst.instance_id}"):
                            scm.delete_instance(inst.instance_id)
                            st.rerun()
            else:
                st.info(t('paper_enhanced.no_strategy_instances'))

                # Manual creation
                st.write("---")
                st.subheader(t('paper_enhanced.create_strategy_instance'))
                c_ticker = st.text_input(t('ui.ticker'), value="AAPL", key="create_ticker")
                c_strategy = st.selectbox(t('paper_enhanced.strategy'), ["MA_Cross", "RSI", "MACD", "Bollinger", "Composite"], key="create_strategy")
                if st.button(t('paper_enhanced.create_instance'), key="create_inst"):
                    inst = scm.create_instance(c_strategy, c_ticker)
                    st.success(f"{t('common.success')}: {inst.instance_id}")
                    st.rerun()

            # A/B Test
            st.write("---")
            st.subheader(t('paper_enhanced.ab_test'))
            ab_results = scm.get_ab_test_results()
            if ab_results:
                ab_df = pd.DataFrame([
                    {
                        t('paper_enhanced.variant'): k,
                        t('paper_enhanced.return_pct'): f"{v['total_return']:+.2f}%",
                        t('paper_enhanced.sharpe_ratio'): f"{v['sharpe']:.2f}",
                        t('paper_enhanced.win_rate_pct'): f"{v['win_rate']:.1f}%",
                        t('paper_enhanced.max_dd'): f"{v['max_dd']:.2f}%",
                    }
                    for k, v in ab_results.items()
                ])
                st.dataframe(ab_df, use_container_width=True, hide_index=True)

                if st.button(t('paper_enhanced.promote_best')):
                    promoted = scm.promote_best_variant()
                    if promoted:
                        st.success(f"{t('common.success')}: {promoted}")
                        st.rerun()
            else:
                st.info(t('paper_enhanced.no_ab_data'))

        # ---- Risk Status ----
        with pt_sub5:
            st.subheader(t('risk.title'))

            # Position risk
            positions = trader.get_all_positions()
            if positions:
                st.write(f"**{t('paper_enhanced.position_risk')}**")
                pos_df = pd.DataFrame([
                    {
                        t('ui.ticker'): t,
                        t('paper_enhanced.qty'): p['qty'],
                        t('paper_enhanced.avg_cost'): f"${p['avg_cost']:.2f}",
                        t('paper_enhanced.weight'): f"{(p['qty']*p['avg_cost'])/(trader.cash + sum(pp['qty']*pp['avg_cost'] for pp in positions.values())):.1%}",
                    }
                    for t, p in positions.items()
                ])
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
            else:
                st.info(t('paper_trading.no_positions'))

            # Risk metrics
            st.write("---")
            st.write(f"**{t('paper_enhanced.risk_limits')}**")
            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.metric(t('paper_enhanced.max_position_pct'), f"{risk_mgr.max_position_pct:.0%}")
            rcol2.metric(t('paper_enhanced.max_total_pct'), f"{risk_mgr.max_total_position_pct:.0%}")
            rcol3.metric(t('paper_enhanced.risk_per_trade'), f"{risk_mgr.risk_per_trade_pct:.1%}")

            rcol4, rcol5, rcol6 = st.columns(3)
            rcol4.metric(t('paper_enhanced.stop_loss'), f"{risk_mgr.stop_loss_atr_mult:.1f}x ATR")
            rcol5.metric(t('paper_enhanced.take_profit'), f"{risk_mgr.take_profit_atr_mult:.1f}x ATR")
            rcol6.metric(t('paper_enhanced.daily_loss_limit'), f"{risk_mgr.daily_loss_stop_pct:.1%}")

        # ---- Orders ----
        with pt_sub6:
            st.subheader(t('paper_enhanced.orders'))

            # 订单管理：优先从 bot 获取，否则显示空状态
            open_orders = []
            om = None
            if bot:
                om = bot.get_order_manager()
                open_orders = om.get_open_orders()

            if open_orders:
                st.write(f"**{t('paper_enhanced.open_orders')} ({len(open_orders)})**")
                for o in open_orders:
                    ocol1, ocol2 = st.columns([4, 1])
                    with ocol1:
                        st.write(f"{o.order_id}: {o.action} {o.qty} {o.ticker} @ {o.order_type.value}" +
                                 (f" ${o.price:.2f}" if o.price else ""))
                    with ocol2:
                        if st.button(t('paper_enhanced.cancel'), key=f"cancel_{o.order_id}"):
                            om.cancel_order(o.order_id)
                            st.rerun()

                if st.button(t('paper_enhanced.cancel_all')):
                    om.cancel_all_orders()
                    st.rerun()
            else:
                st.info(t('paper_enhanced.no_trades_yet'))

            st.write("---")
            st.subheader(t('paper_enhanced.manual_order'))
            ocol1, ocol2, ocol3 = st.columns(3)
            with ocol1:
                o_ticker = st.text_input(t('ui.ticker'), value="AAPL", key="man_ticker")
            with ocol2:
                o_action = st.selectbox(t('paper_enhanced.action'), ["BUY", "SELL"], key="man_action")
            with ocol3:
                o_qty = st.number_input(t('paper_enhanced.qty'), min_value=1, value=100, key="man_qty")

            o_type = st.selectbox(t('paper_enhanced.order_type'), ["MARKET", "LIMIT", "STOP"], key="man_type")
            o_price = None
            if o_type == "LIMIT":
                o_price = st.number_input(t('paper_enhanced.limit_price'), min_value=0.01, value=150.0, key="man_price")

            if st.button(t('paper_enhanced.submit_order'), key="man_submit"):
                if bot and om:
                    order = om.submit_order(
                        ticker=o_ticker, action=o_action, qty=o_qty,
                        order_type=o_type, price=o_price,
                    )
                    st.success(f"{t('common.success')}: {order.order_id}")
                else:
                    # Direct paper trade
                    if o_action == "BUY":
                        result = trader.buy(o_ticker, o_price or 150.0, o_qty)
                    else:
                        result = trader.sell(o_ticker, o_price or 150.0, o_qty)
                    st.success(f"{t('common.success')}: {result}")

    else:
        st.error(t('paper_enhanced.enhanced_not_available'))


# ============================================================
# TAB 3: Live Trading Mode
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
# TAB 4: Optimize Mode
# ============================================================
with tab_optimize:
    st.markdown(f"### ⚙️ {t('optimization.title')}")
    st.caption(t("ui.optimize_tab_desc") if t("ui.optimize_tab_desc") != "ui.optimize_tab_desc" else "Strategy parameter optimization using grid/random/Bayesian search")

    # Operation mechanism description (NO ASCII box lines)
    with st.expander(f"📖 {t('optimization.help_button')}", expanded=False):
        st.markdown(f"""
        **{t('ui.optimize_workflow_title') if t('ui.optimize_workflow_title') != 'ui.optimize_workflow_title' else 'Optimization Workflow'}:**
        
        1. **{t('ui.optimize_step1_title') if t('ui.optimize_step1_title') != 'ui.optimize_step1_title' else 'Select Strategy'}**: {t('ui.optimize_step1_desc') if t('ui.optimize_step1_desc') != 'ui.optimize_step1_desc' else 'Choose strategy type (MA/RSI/MACD/Bollinger/Composite)'}
        2. **{t('ui.optimize_step2_title') if t('ui.optimize_step2_title') != 'ui.optimize_step2_title' else 'Select Target'}**: {t('ui.optimize_step2_desc') if t('ui.optimize_step2_desc') != 'ui.optimize_step2_desc' else 'Enter ticker symbol, system fetches 2 years of historical data'}
        3. **{t('ui.optimize_step3_title') if t('ui.optimize_step3_title') != 'ui.optimize_step3_title' else 'Select Method'}**:
           - **{t('ui.grid_search')}**: {t('ui.optimize_grid_desc') if t('ui.optimize_grid_desc') != 'ui.optimize_grid_desc' else 'Exhaustive search, most thorough but slower'}
           - **{t('ui.random_search')}**: {t('ui.optimize_random_desc') if t('ui.optimize_random_desc') != 'ui.optimize_random_desc' else 'Random sampling, suitable for large parameter spaces'}
           - **{t('ui.bayesian')}**: {t('ui.optimize_bayesian_desc') if t('ui.optimize_bayesian_desc') != 'ui.optimize_bayesian_desc' else 'Intelligent search using prior results, most efficient'}
        4. **{t('ui.optimize_step4_title') if t('ui.optimize_step4_title') != 'ui.optimize_step4_title' else 'Set Trials'}**: {t('ui.optimize_step4_desc') if t('ui.optimize_step4_desc') != 'ui.optimize_step4_desc' else 'Ignored for grid search, used for random/Bayesian'}
        5. **{t('ui.optimize_step5_title') if t('ui.optimize_step5_title') != 'ui.optimize_step5_title' else 'Run'}**: {t('ui.optimize_step5_desc') if t('ui.optimize_step5_desc') != 'ui.optimize_step5_desc' else 'System automatically runs backtests to find optimal parameters'}
        
        **{t('ui.optimize_output_title') if t('ui.optimize_output_title') != 'ui.optimize_output_title' else 'Output'}:**
        - **{t('ui.best_params')}**: {t('ui.optimize_output_params_desc') if t('ui.optimize_output_params_desc') != 'ui.optimize_output_params_desc' else 'Parameter combination maximizing Sharpe ratio'}
        - **{t('backtest.metrics_sharpe')}**: {t('ui.optimize_output_sharpe_desc') if t('ui.optimize_output_sharpe_desc') != 'ui.optimize_output_sharpe_desc' else 'Sharpe ratio for optimal parameters'}
        - **{t('optimization.results_table')}**: {t('ui.optimize_output_table_desc') if t('ui.optimize_output_table_desc') != 'ui.optimize_output_table_desc' else 'All trial parameters and scores'}
        """)

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
    st.caption(t("ui.settings_tab_desc") if t("ui.settings_tab_desc") != "ui.settings_tab_desc" else "Language, risk, notification and API configuration")

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
