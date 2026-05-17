# ============================================================
# ai/alert_system.py — 实时预警系统
# 基于价格/指标阈值触发预警，支持多种条件组合
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlertCondition:
    """预警条件定义"""
    condition_id: str
    ticker: str
    condition_type: str  # "price_above", "price_below", "rsi_above", "rsi_below",
                         # "macd_cross_up", "macd_cross_down", "volume_spike",
                         # "atr_spike", "gap_up", "gap_down", "support_break",
                         # "resistance_break", "divergence", "vix_spike"
    threshold: float
    params: dict = field(default_factory=dict)  # 额外参数
    enabled: bool = True
    cooldown_minutes: int = 60  # 冷却时间（避免重复触发）
    last_triggered: Optional[str] = None
    trigger_count: int = 0


@dataclass
class AlertEvent:
    """预警事件"""
    alert_id: str
    ticker: str
    condition_type: str
    triggered_at: str
    severity: str  # "info" / "warning" / "critical"
    message: str
    current_value: float
    threshold: float
    context: dict = field(default_factory=dict)  # 触发时的市场上下文


@dataclass
class AlertSystemState:
    """预警系统状态"""
    ticker: str
    current_price: float = 0.0
    current_rsi: Optional[float] = None
    current_macd_hist: Optional[float] = None
    current_volume_ratio: Optional[float] = None
    current_atr_pct: Optional[float] = None
    current_vix: Optional[float] = None
    # 预警历史
    alert_history: list = field(default_factory=list)
    active_alerts: list = field(default_factory=list)
    # 统计
    total_triggers: int = 0
    triggers_today: int = 0
    last_check: str = ""


class AlertSystem:
    """实时预警系统"""

    def __init__(self):
        self.conditions: dict[str, AlertCondition] = {}
        self.state: dict[str, AlertSystemState] = {}
        self.callbacks: list[Callable] = []

    def add_condition(self, condition: AlertCondition) -> None:
        """添加预警条件"""
        self.conditions[condition.condition_id] = condition
        if condition.ticker not in self.state:
            self.state[condition.ticker] = AlertSystemState(ticker=condition.ticker)
        logger.info(f"[Alert] 添加预警条件: {condition.condition_id} ({condition.condition_type})")

    def remove_condition(self, condition_id: str) -> None:
        """移除预警条件"""
        if condition_id in self.conditions:
            del self.conditions[condition_id]
            logger.info(f"[Alert] 移除预警条件: {condition_id}")

    def register_callback(self, callback: Callable) -> None:
        """注册预警回调函数"""
        self.callbacks.append(callback)

    def _check_price_condition(
        self,
        state: AlertSystemState,
        cond: AlertCondition,
    ) -> Optional[AlertEvent]:
        """检查价格条件"""
        if state.current_price == 0:
            return None

        if cond.condition_type == "price_above" and state.current_price > cond.threshold:
            return AlertEvent(
                alert_id=cond.condition_id,
                ticker=cond.ticker,
                condition_type=cond.condition_type,
                triggered_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                severity="warning",
                message=f"{cond.ticker} 价格突破 {cond.threshold}: 当前 {state.current_price:.2f}",
                current_value=state.current_price,
                threshold=cond.threshold,
            )
        elif cond.condition_type == "price_below" and state.current_price < cond.threshold:
            return AlertEvent(
                alert_id=cond.condition_id,
                ticker=cond.ticker,
                condition_type=cond.condition_type,
                triggered_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                severity="warning",
                message=f"{cond.ticker} 价格跌破 {cond.threshold}: 当前 {state.current_price:.2f}",
                current_value=state.current_price,
                threshold=cond.threshold,
            )
        return None

    def _check_rsi_condition(
        self,
        state: AlertSystemState,
        cond: AlertCondition,
    ) -> Optional[AlertEvent]:
        """检查RSI条件"""
        if state.current_rsi is None:
            return None

        if cond.condition_type == "rsi_above" and state.current_rsi > cond.threshold:
            return AlertEvent(
                alert_id=cond.condition_id,
                ticker=cond.ticker,
                condition_type=cond.condition_type,
                triggered_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                severity="info" if state.current_rsi < 75 else "warning",
                message=f"{cond.ticker} RSI超买: {state.current_rsi:.1f} (阈值{cond.threshold})",
                current_value=state.current_rsi,
                threshold=cond.threshold,
            )
        elif cond.condition_type == "rsi_below" and state.current_rsi < cond.threshold:
            return AlertEvent(
                alert_id=cond.condition_id,
                ticker=cond.ticker,
                condition_type=cond.condition_type,
                triggered_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                severity="info" if state.current_rsi > 25 else "warning",
                message=f"{cond.ticker} RSI超卖: {state.current_rsi:.1f} (阈值{cond.threshold})",
                current_value=state.current_rsi,
                threshold=cond.threshold,
            )
        return None

    def _check_volume_condition(
        self,
        state: AlertSystemState,
        cond: AlertCondition,
    ) -> Optional[AlertEvent]:
        """检查成交量条件"""
        if state.current_volume_ratio is None:
            return None

        if cond.condition_type == "volume_spike" and state.current_volume_ratio > cond.threshold:
            return AlertEvent(
                alert_id=cond.condition_id,
                ticker=cond.ticker,
                condition_type=cond.condition_type,
                triggered_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                severity="info",
                message=f"{cond.ticker} 成交量突增: {state.current_volume_ratio:.1f}x均值",
                current_value=state.current_volume_ratio,
                threshold=cond.threshold,
            )
        return None

    def _check_vix_condition(
        self,
        state: AlertSystemState,
        cond: AlertCondition,
    ) -> Optional[AlertEvent]:
        """检查VIX条件"""
        if state.current_vix is None:
            return None

        if cond.condition_type == "vix_spike" and state.current_vix > cond.threshold:
            return AlertEvent(
                alert_id=cond.condition_id,
                ticker=cond.ticker,
                condition_type=cond.condition_type,
                triggered_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                severity="critical" if state.current_vix > 35 else "warning",
                message=f"VIX飙升: {state.current_vix:.1f} (阈值{cond.threshold}) — 市场恐慌情绪上升",
                current_value=state.current_vix,
                threshold=cond.threshold,
            )
        return None

    def check_all_conditions(
        self,
        ticker: str,
        current_data: dict,
    ) -> list[AlertEvent]:
        """
        检查所有条件并返回触发的事件。

        参数:
            ticker: 标的代码
            current_data: 当前市场数据字典
                {
                    "price": float,
                    "rsi": float,
                    "macd_hist": float,
                    "volume_ratio": float,
                    "atr_pct": float,
                    "vix": float,
                }
        """
        if ticker not in self.state:
            self.state[ticker] = AlertSystemState(ticker=ticker)

        state = self.state[ticker]
        state.current_price = current_data.get("price", 0.0)
        state.current_rsi = current_data.get("rsi")
        state.current_macd_hist = current_data.get("macd_hist")
        state.current_volume_ratio = current_data.get("volume_ratio")
        state.current_atr_pct = current_data.get("atr_pct")
        state.current_vix = current_data.get("vix")
        state.last_check = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        triggered = []

        for cond in self.conditions.values():
            if not cond.enabled or cond.ticker != ticker:
                continue

            # 冷却时间检查
            if cond.last_triggered:
                last_dt = datetime.fromisoformat(cond.last_triggered.replace(" UTC", "+00:00"))
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
                if elapsed < cond.cooldown_minutes:
                    continue

            event = None
            if cond.condition_type in ("price_above", "price_below"):
                event = self._check_price_condition(state, cond)
            elif cond.condition_type in ("rsi_above", "rsi_below"):
                event = self._check_rsi_condition(state, cond)
            elif cond.condition_type == "volume_spike":
                event = self._check_volume_condition(state, cond)
            elif cond.condition_type == "vix_spike":
                event = self._check_vix_condition(state, cond)

            if event:
                event.context = {
                    "price": state.current_price,
                    "rsi": state.current_rsi,
                    "volume_ratio": state.current_volume_ratio,
                    "vix": state.current_vix,
                }
                triggered.append(event)
                cond.last_triggered = event.triggered_at
                cond.trigger_count += 1
                state.alert_history.append(event)
                state.total_triggers += 1
                state.triggers_today += 1

                # 执行回调
                for cb in self.callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        logger.warning(f"[Alert] 回调执行失败: {e}")

        state.active_alerts = triggered
        return triggered

    def get_alert_summary(self, ticker: str) -> dict:
        """获取预警摘要"""
        state = self.state.get(ticker)
        if not state:
            return {"error": "无该标的的预警状态"}

        recent_alerts = state.alert_history[-10:] if state.alert_history else []

        return {
            "标的": ticker,
            "当前价格": state.current_price,
            "当前RSI": state.current_rsi,
            "当前VIX": state.current_vix,
            "总触发次数": state.total_triggers,
            "今日触发": state.triggers_today,
            "最近检查": state.last_check,
            "活跃预警": [
                {
                    "类型": a.condition_type,
                    "严重度": a.severity,
                    "消息": a.message,
                    "触发时间": a.triggered_at,
                }
                for a in state.active_alerts
            ],
            "历史预警(最近10条)": [
                {
                    "类型": a.condition_type,
                    "严重度": a.severity,
                    "消息": a.message,
                    "触发时间": a.triggered_at,
                }
                for a in recent_alerts
            ],
        }

    def reset_daily_counters(self) -> None:
        """重置每日计数器（应在每天开盘前调用）"""
        for state in self.state.values():
            state.triggers_today = 0


def create_default_alerts(ticker: str) -> list[AlertCondition]:
    """
    为标的创建默认预警条件集。
    适用于SPY/GLD等主要标的。
    """
    conditions = [
        # RSI极端值
        AlertCondition(f"{ticker}_rsi_overbought", ticker, "rsi_above", 70, cooldown_minutes=120),
        AlertCondition(f"{ticker}_rsi_oversold", ticker, "rsi_below", 30, cooldown_minutes=120),
        # 成交量异常
        AlertCondition(f"{ticker}_volume_spike", ticker, "volume_spike", 2.5, cooldown_minutes=60),
        # VIX恐慌（针对SPY）
    ]

    if ticker in ("SPY", "QQQ", "IWM"):
        conditions.append(AlertCondition(f"{ticker}_vix_spike", ticker, "vix_spike", 30, cooldown_minutes=180))

    return conditions


def check_alerts_for_ticker(
    ticker: str,
    fetcher,
    alert_system: Optional[AlertSystem] = None,
) -> dict:
    """
    对指定标的执行完整预警检查。
    获取最新数据并检查所有条件。
    """
    if alert_system is None:
        alert_system = AlertSystem()
        for cond in create_default_alerts(ticker):
            alert_system.add_condition(cond)

    # 获取最新数据
    current_data = {"price": 0.0}
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        current_data["price"] = info.last_price or 0

        # 获取历史数据计算指标
        df = fetcher.download_history(ticker, period="60d", interval="1d")
        if not df.empty and len(df) >= 20:
            close = df["Close"]
            volume = df.get("Volume", pd.Series(np.ones(len(close)), index=close.index))

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_data["rsi"] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

            # Volume ratio
            vol_sma20 = volume.rolling(20).mean()
            vol_ratio = volume.iloc[-1] / vol_sma20.iloc[-1] if vol_sma20.iloc[-1] > 0 else 1.0
            current_data["volume_ratio"] = float(vol_ratio) if not pd.isna(vol_ratio) else None

            # ATR%
            tr1 = df["High"] - df["Low"]
            tr2 = abs(df["High"] - close.shift())
            tr3 = abs(df["Low"] - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_pct = atr.iloc[-1] / close.iloc[-1] * 100 if close.iloc[-1] > 0 else None
            current_data["atr_pct"] = float(atr_pct) if not pd.isna(atr_pct) else None

            # MACD hist
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd.iloc[-1] - signal.iloc[-1]
            current_data["macd_hist"] = float(macd_hist) if not pd.isna(macd_hist) else None
    except Exception as e:
        logger.warning(f"[Alert] {ticker} 获取指标数据失败: {e}")

    # VIX（针对SPY）
    if ticker in ("SPY", "QQQ", "IWM"):
        try:
            vix = yf.Ticker("^VIX")
            vix_info = vix.fast_info
            current_data["vix"] = vix_info.last_price or None
        except Exception:
            pass

    # 检查所有条件
    events = alert_system.check_all_conditions(ticker, current_data)

    return {
        "标的": ticker,
        "当前数据": current_data,
        "触发预警": [
            {
                "类型": e.condition_type,
                "严重度": e.severity,
                "消息": e.message,
                "触发时间": e.triggered_at,
            }
            for e in events
        ],
        "预警统计": alert_system.get_alert_summary(ticker),
    }
