# ============================================================
# dashboard/settings.py — Settings Page (i18n)
# ============================================================
import os
import sys
import logging
from pathlib import Path
from typing import Dict

import streamlit as st
import yaml

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)
CONFIG_FILE = BASE_DIR / "config.yaml"


def _t(key):
    """Shortcut to get i18n text from app's I18nManager."""
    from dashboard.app import t
    return t(key)


def load_config() -> Dict:
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config: Dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def render_settings_page():
    st.markdown(f"## ⚙️ {_t('settings.title')}")

    config = load_config()

    tab1, tab2, tab3, tab4 = st.tabs([
        f"📧 {_t('settings.section_notification')}",
        f"🤖 {_t('settings.section_ai')}",
        f"📈 {_t('settings.section_moomoo')}",
        f"📊 {_t('settings.section_other')}",
    ])

    with tab1:
        _render_notification_settings(config)
    with tab2:
        _render_ai_settings(config)
    with tab3:
        _render_moomoo_settings(config)
    with tab4:
        _render_other_settings(config)


def _render_notification_settings(config: Dict):
    st.markdown(f"### 📧 {_t('settings.notif_title')}")

    notification_cfg = config.get("notification", {})

    enable_notification = st.toggle(
        _t("settings.notif_enable"),
        value=notification_cfg.get("enable", False),
        key="enable_notification",
    )

    if enable_notification:
        st.markdown("---")

        # Email
        with st.expander(f"📧 {_t('settings.email_notif')}", expanded=False):
            email_cfg = notification_cfg.get("email", {})
            enable_email = st.checkbox(_t("settings.email_enable"), value=email_cfg.get("enable", False))

            if enable_email:
                col1, col2 = st.columns(2)
                with col1:
                    smtp_host = st.text_input(_t("settings.smtp_host"), value=email_cfg.get("smtp_host", ""), placeholder="smtp.qq.com")
                    smtp_port = st.number_input(_t("settings.smtp_port"), value=email_cfg.get("smtp_port", 587), min_value=1, max_value=65535)
                with col2:
                    smtp_user = st.text_input(_t("settings.smtp_user"), value=email_cfg.get("smtp_user", ""), placeholder="your@email.com")
                    smtp_password = st.text_input(_t("settings.smtp_password"), value=email_cfg.get("smtp_password", ""), type="password")

                from_addr = st.text_input(_t("settings.from_addr"), value=email_cfg.get("from_addr", ""), placeholder="your@email.com")
                to_addrs = st.text_area(_t("settings.to_addrs"), value="\n".join(email_cfg.get("to_addrs", [])), height=100)

                if st.button(f"📨 {_t('settings.test_email')}"):
                    if smtp_host and smtp_user and smtp_password:
                        try:
                            from notification.notifier import EmailNotifier, NotificationMessage
                            notifier = EmailNotifier(
                                smtp_host=smtp_host, smtp_port=smtp_port,
                                smtp_user=smtp_user, smtp_password=smtp_password,
                                from_addr=from_addr or smtp_user,
                                to_addrs=[a.strip() for a in to_addrs.split("\n") if a.strip()],
                            )
                            msg = NotificationMessage(
                                title=_t("settings.test_email_title"),
                                content=_t("settings.test_email_content"),
                                level="info",
                            )
                            if notifier.send(msg):
                                st.success(_t("settings.test_success"))
                            else:
                                st.error(_t("settings.test_fail"))
                        except Exception as e:
                            st.error(f"{_t('settings.test_fail')}: {e}")
                    else:
                        st.warning(_t("settings.smtp_required"))

        # Telegram
        with st.expander(f"💬 {_t('settings.tg_notif')}", expanded=False):
            tg_cfg = notification_cfg.get("telegram", {})
            enable_telegram = st.checkbox(_t("settings.tg_enable"), value=tg_cfg.get("enable", False))

            if enable_telegram:
                bot_token = st.text_input("Bot Token", value=tg_cfg.get("bot_token", ""), placeholder="123456:ABC-DEF...")
                chat_id = st.text_input("Chat ID", value=tg_cfg.get("chat_id", ""), placeholder="123456789")

                st.markdown(_t("settings.tg_help"))

                if st.button(f"💬 {_t('settings.test_tg')}"):
                    if bot_token and chat_id:
                        try:
                            from notification.notifier import TelegramNotifier, NotificationMessage
                            notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
                            msg = NotificationMessage(
                                title=_t("settings.test_msg_title"),
                                content=_t("settings.test_msg_content"),
                                level="info",
                            )
                            if notifier.send(msg):
                                st.success(_t("settings.test_success"))
                            else:
                                st.error(_t("settings.test_fail"))
                        except Exception as e:
                            st.error(f"{_t('settings.test_fail')}: {e}")
                    else:
                        st.warning(_t("settings.tg_required"))

        # WeChat
        with st.expander(f"🏢 {_t('settings.wechat_notif')}", expanded=False):
            wechat_cfg = notification_cfg.get("wechat", {})
            enable_wechat = st.checkbox(_t("settings.wechat_enable"), value=wechat_cfg.get("enable", False))

            if enable_wechat:
                webhook_url = st.text_input("Webhook URL", value=wechat_cfg.get("webhook_url", ""), placeholder="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx")

                if st.button(f"🏢 {_t('settings.test_wechat')}"):
                    if webhook_url:
                        try:
                            from notification.notifier import WeChatNotifier, NotificationMessage
                            notifier = WeChatNotifier(webhook_url=webhook_url)
                            msg = NotificationMessage(
                                title=_t("settings.test_msg_title"),
                                content=_t("settings.test_wechat_content"),
                                level="info",
                            )
                            if notifier.send(msg):
                                st.success(_t("settings.test_success"))
                            else:
                                st.error(_t("settings.test_fail"))
                        except Exception as e:
                            st.error(f"{_t('settings.test_fail')}: {e}")
                    else:
                        st.warning(_t("settings.wechat_required"))

        # Triggers
        st.markdown("---")
        st.markdown(f"**{_t('settings.notif_triggers')}**")
        triggers_cfg = notification_cfg.get("triggers", {})

        t1, t2, t3 = st.columns(3)
        with t1:
            on_trade = st.checkbox(_t("settings.trigger_trade"), value=triggers_cfg.get("on_trade", True))
            on_signal = st.checkbox(_t("settings.trigger_signal"), value=triggers_cfg.get("on_signal", True))
        with t2:
            on_error = st.checkbox(_t("settings.trigger_error"), value=triggers_cfg.get("on_error", True))
            on_daily = st.checkbox(_t("settings.trigger_daily"), value=triggers_cfg.get("on_daily_report", True))
        with t3:
            on_risk = st.checkbox(_t("settings.trigger_risk"), value=triggers_cfg.get("on_risk_alert", True))

    if st.button(f"💾 {_t('settings.save_notif')}", type="primary"):
        config["notification"] = {
            "enable": enable_notification,
            "email": {
                "enable": enable_email if enable_notification else False,
                "smtp_host": smtp_host if enable_notification and enable_email else "",
                "smtp_port": smtp_port if enable_notification and enable_email else 587,
                "smtp_user": smtp_user if enable_notification and enable_email else "",
                "smtp_password": smtp_password if enable_notification and enable_email else "",
                "from_addr": from_addr if enable_notification and enable_email else "",
                "to_addrs": [a.strip() for a in to_addrs.split("\n") if a.strip()] if enable_notification and enable_email else [],
            },
            "telegram": {
                "enable": enable_telegram if enable_notification else False,
                "bot_token": bot_token if enable_notification and enable_telegram else "",
                "chat_id": chat_id if enable_notification and enable_telegram else "",
            },
            "wechat": {
                "enable": enable_wechat if enable_notification else False,
                "webhook_url": webhook_url if enable_notification and enable_wechat else "",
            },
            "triggers": {
                "on_trade": on_trade, "on_signal": on_signal,
                "on_error": on_error, "on_daily_report": on_daily,
                "on_risk_alert": on_risk,
            },
        }
        save_config(config)
        st.success(_t("settings.save_success"))


def _render_ai_settings(config: Dict):
    st.markdown(f"### 🤖 {_t('settings.ai_title')}")

    ai_cfg = config.get("ai", {})
    llm_cfg = ai_cfg.get("llm", {})

    enable_llm = st.toggle(
        _t("settings.ai_enable"),
        value=llm_cfg.get("enable", False),
        key="enable_llm",
    )

    if enable_llm:
        st.markdown("---")

        provider = st.selectbox(
            _t("settings.ai_provider"),
            options=["openai", "anthropic", "deepseek", "local"],
            index=["openai", "anthropic", "deepseek", "local"].index(llm_cfg.get("provider", "openai")),
        )

        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input(_t("settings.ai_api_key"), value=llm_cfg.get("api_key", ""), type="password")
            api_base = st.text_input(
                _t("settings.ai_api_base"),
                value=llm_cfg.get("api_base", ""),
                placeholder="https://api.openai.com/v1",
            )
        with col2:
            model = st.text_input(_t("settings.ai_model"), value=llm_cfg.get("model", "gpt-4o-mini"))
            temperature = st.slider(
                _t("settings.ai_temperature"),
                min_value=0.0, max_value=1.0,
                value=llm_cfg.get("temperature", 0.7), step=0.1,
            )

        max_tokens = st.number_input(
            _t("settings.ai_max_tokens"),
            min_value=100, max_value=8000,
            value=llm_cfg.get("max_tokens", 2000), step=100,
        )

        if st.button(f"🧪 {_t('settings.ai_test')}"):
            if api_key and model:
                try:
                    import openai
                    client = openai.OpenAI(api_key=api_key, base_url=api_base if api_base else None)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Hello, this is a test."}],
                        max_tokens=50, temperature=temperature,
                    )
                    st.success(f"{_t('settings.ai_test_ok')} {response.choices[0].message.content}")
                except Exception as e:
                    st.error(f"{_t('settings.ai_test_fail')}: {e}")
            else:
                st.warning(_t("settings.ai_key_required"))

    if st.button(f"💾 {_t('settings.save_ai')}", type="primary"):
        config["ai"] = {
            "llm": {
                "enable": enable_llm,
                "provider": provider if enable_llm else "openai",
                "api_key": api_key if enable_llm else "",
                "api_base": api_base if enable_llm else "",
                "model": model if enable_llm else "gpt-4o-mini",
                "temperature": temperature if enable_llm else 0.7,
                "max_tokens": max_tokens if enable_llm else 2000,
            },
        }
        save_config(config)
        st.success(_t("settings.save_success"))


def _render_moomoo_settings(config: Dict):
    st.markdown(f"### 📈 {_t('settings.moomoo_title')}")

    st.markdown(_t("settings.moomoo_help"))

    moomoo_cfg = config.get("data", {}).get("moomoo", {})

    enable_moomoo = st.toggle(
        _t("settings.moomoo_enable"),
        value=moomoo_cfg.get("enable", False),
        key="enable_moomoo",
    )

    if enable_moomoo:
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input(_t("settings.moomoo_host"), value=moomoo_cfg.get("host", "127.0.0.1"))
            port = st.number_input(_t("settings.moomoo_port"), value=moomoo_cfg.get("port", 11111), min_value=1, max_value=65535)
        with col2:
            paper_trade = st.checkbox(
                _t("settings.moomoo_paper"),
                value=moomoo_cfg.get("paper_trade", True),
            )
            trade_pwd = st.text_input(
                _t("settings.moomoo_pwd"),
                value=moomoo_cfg.get("trade_pwd", ""),
                type="password",
            ) if not paper_trade else ""

        timeout = st.number_input(
            _t("settings.moomoo_timeout"),
            value=moomoo_cfg.get("timeout_seconds", 30),
            min_value=5, max_value=300,
        )

        if st.button(f"🔌 {_t('settings.moomoo_test')}"):
            try:
                from moomoo import OpenQuoteContext
                ctx = OpenQuoteContext(host=host, port=port)
                ret, data = ctx.get_global_state()
                ctx.close()
                if ret == 0:
                    st.success(f"{_t('settings.moomoo_test_ok')} {data}")
                else:
                    st.error(f"{_t('settings.moomoo_test_fail')} {data}")
            except ImportError:
                st.error(_t("settings.moomoo_not_installed"))
            except Exception as e:
                st.error(f"{_t('settings.moomoo_test_fail')} {e}")

        if not paper_trade:
            st.warning(_t("settings.moomoo_live_warning"))

    if st.button(f"💾 {_t('settings.save_moomoo')}", type="primary"):
        if "data" not in config:
            config["data"] = {}
        config["data"]["moomoo"] = {
            "enable": enable_moomoo,
            "host": host if enable_moomoo else "127.0.0.1",
            "port": port if enable_moomoo else 11111,
            "paper_trade": paper_trade if enable_moomoo else True,
            "trade_pwd": trade_pwd if enable_moomoo and not paper_trade else "",
            "timeout_seconds": timeout if enable_moomoo else 30,
        }
        save_config(config)
        st.success(_t("settings.save_success"))


def _render_other_settings(config: Dict):
    st.markdown(f"### 📊 {_t('settings.other_title')}")

    data_cfg = config.get("data", {}).get("yf", {})

    st.markdown(f"#### {_t('settings.data_update')}")
    col1, col2 = st.columns(2)
    with col1:
        auto_update = st.checkbox(_t("settings.auto_update"), value=data_cfg.get("auto_update", False))
    with col2:
        update_interval = st.number_input(
            _t("settings.update_interval"),
            value=data_cfg.get("update_interval_minutes", 30), min_value=1, max_value=1440,
        )

    st.markdown(f"#### {_t('settings.strategy_opt')}")
    opt_cfg = config.get("optimization", {})
    col1, col2 = st.columns(2)
    with col1:
        enable_optimization = st.checkbox(_t("settings.enable_opt"), value=opt_cfg.get("enable", False))
        method = st.selectbox(
            _t("settings.opt_method"), ["grid", "random", "bayesian"],
            index=["grid", "random", "bayesian"].index(opt_cfg.get("method", "grid")),
        )
    with col2:
        n_trials = st.number_input(_t("settings.opt_trials"), value=opt_cfg.get("n_trials", 100), min_value=10, max_value=1000)
        scoring = st.selectbox(
            _t("settings.opt_target"), ["sharpe", "return", "max_drawdown"],
            index=["sharpe", "return", "max_drawdown"].index(opt_cfg.get("scoring", "sharpe")),
        )

    st.markdown(f"#### {_t('settings.scheduler')}")
    scheduler_cfg = config.get("scheduler", {})
    enable_scheduler = st.checkbox(_t("settings.enable_scheduler"), value=scheduler_cfg.get("enable", False))

    if st.button(f"💾 {_t('settings.save_other')}", type="primary"):
        if "data" not in config:
            config["data"] = {}
        if "yf" not in config["data"]:
            config["data"]["yf"] = {}
        config["data"]["yf"]["auto_update"] = auto_update
        config["data"]["yf"]["update_interval_minutes"] = update_interval

        config["optimization"] = {
            "enable": enable_optimization, "method": method,
            "n_trials": n_trials, "scoring": scoring, "output_dir": "optimization/",
        }

        config["scheduler"] = {
            "enable": enable_scheduler,
            "data_update_cron": "0 */30 9-16 * * 1-5",
            "daily_analysis_cron": "0 18 * * 1-5",
            "weekly_report_cron": "0 18 * * 5",
        }

        save_config(config)
        st.success(_t("settings.save_success"))


if __name__ == "__main__":
    st.set_page_config(page_title="Quant Trader - Settings", page_icon="⚙️", layout="wide")
    render_settings_page()
