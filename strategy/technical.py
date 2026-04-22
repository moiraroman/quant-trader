# ============================================================
# strategy/technical.py — 技术指标策略
# 包含：MA 交叉、RSI、MACD、布林带、均线 + AI 加权
# ============================================================
import logging
from typing import Optional

import pandas as pd
import numpy as np

from strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

# ============================================================
# 指标计算工具
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 OHLCV DataFrame 上计算所有技术指标。
    使用 pandas_ta（纯 Python，无需编译）。
    """
    df = df.copy()

    # ---- 均线 ----
    for window in [5, 10, 20, 50, 200]:
        if len(df) >= window:
            df[f"SMA_{window}"] = df["Close"].rolling(window).mean()
            df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

    # ---- RSI ----
    if len(df) >= 14:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["RSI_14"] = 100 - (100 / (1 + rs))

    # ---- MACD ----
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ---- 布林带 ----
    if len(df) >= 20:
        bb = df["Close"].rolling(20).agg(["mean", "std"])
        df["BB_mid"] = bb["mean"]
        df["BB_upper"] = bb["mean"] + 2 * bb["std"]
        df["BB_lower"] = bb["mean"] - 2 * bb["std"]
        df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-10)

    # ---- 成交量指标 ----
    if "Volume" in df.columns and len(df) >= 20:
        df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()
        df["Vol_ratio"] = df["Volume"] / (df["Vol_SMA_20"] + 1)

    # ---- 价格动量 ----
    for n in [5, 10, 20]:
        if len(df) >= n:
            df[f"Return_{n}d"] = df["Close"].pct_change(n)
            df[f"Momentum_{n}d"] = df["Close"] - df["Close"].shift(n)

    # ---- ATR（真实波幅）----
    if len(df) >= 14:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()

    return df


# ============================================================
# 策略1：MA 交叉策略
# ============================================================

class MAStrategy(BaseStrategy):
    """
    均线金叉死叉策略。

    参数:
        short_window: 短期均线周期（默认 10）
        long_window:  长期均线周期（默认 50）
    """

    name = "MA_Crossover"

    @property
    def default_params(self):
        return {
            "short_window": 10,
            "long_window": 50,
        }

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        sw, lw = p["short_window"], p["long_window"]

        try:
            df_ind = add_indicators(df)
        except Exception:
            # add_indicators may fail on edge-case data
            df_ind = df.copy()

        if len(df_ind) < lw or "Close" not in df_ind.columns:
            return pd.DataFrame({"signal": [], "strength": [], "confidence": [], "reason": []})

        close = df_ind["Close"]
        sma_s = df_ind.get(f"SMA_{sw}")
        sma_l = df_ind.get(f"SMA_{lw}")

        # Compute missing SMAs if not in pre-computed indicators
        if sma_s is None:
            sma_s = close.rolling(window=sw).mean()
        if sma_l is None:
            sma_l = close.rolling(window=lw).mean()

        # 金叉：short 上穿 long → BUY
        # 死叉：short 下穿 long → SELL
        sigs = []
        for i in range(len(df_ind)):
            if pd.isna(sma_s.iloc[i]) or pd.isna(sma_l.iloc[i]):
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.0, "reason": ""})
                continue

            if i < 1:
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.0, "reason": ""})
                continue

            prev_s, prev_l = sma_s.iloc[i-1], sma_l.iloc[i-1]
            curr_s, curr_l = sma_s.iloc[i], sma_l.iloc[i]

            if prev_s <= prev_l and curr_s > curr_l:
                sigs.append({"signal": "BUY", "strength": 0.7, "confidence": 0.7,
                             "reason": f"MA{sw} 上穿 MA{lw} 金叉"})
            elif prev_s >= prev_l and curr_s < curr_l:
                sigs.append({"signal": "SELL", "strength": 0.7, "confidence": 0.7,
                             "reason": f"MA{sw} 下穿 MA{lw} 死叉"})
            else:
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.5, "reason": ""})

        result = pd.DataFrame(sigs, index=df_ind.index)
        result.index.name = "Date"
        return result


# ============================================================
# 策略2：RSI 策略
# ============================================================

class RSIStrategy(BaseStrategy):
    """
    RSI 超买超卖策略。

    参数:
        period: RSI 周期（默认 14）
        oversold: 超卖阈值（默认 30）
        overbought: 超买阈值（默认 70）
    """

    name = "RSI"

    @property
    def default_params(self):
        return {
            "period": 14,
            "oversold": 30,
            "overbought": 70,
        }

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df_ind = add_indicators(df)
        if "RSI_14" not in df_ind.columns or df_ind["RSI_14"].isna().all():
            return pd.DataFrame({"signal": [], "strength": [], "confidence": [], "reason": []})

        rsi = df_ind["RSI_14"]
        sigs = []
        prev_state = "normal"  # 正常 / oversold / overbought
        for i in range(len(df_ind)):
            v = rsi.iloc[i]
            if pd.isna(v):
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.0, "reason": ""})
                continue

            curr_state = "normal"
            if v < p["oversold"]:
                curr_state = "oversold"
            elif v > p["overbought"]:
                curr_state = "overbought"

            # 状态转换信号（只在边界切换时触发，避免每日重复）
            if prev_state == "oversold" and curr_state == "normal":
                # 从超卖区回归 → 买入信号（反弹确认）
                sigs.append({"signal": "BUY", "strength": 0.75,
                             "confidence": 0.75, "reason": f"RSI={v:.1f} 脱离超卖"})
            elif prev_state == "overbought" and curr_state == "normal":
                # 从超买区回归 → 卖出信号（回落确认）
                sigs.append({"signal": "SELL", "strength": 0.75,
                             "confidence": 0.75, "reason": f"RSI={v:.1f} 脱离超买"})
            elif prev_state == "normal" and curr_state == "oversold":
                # 首次进入超卖 → 轻仓试探
                sigs.append({"signal": "BUY", "strength": 0.6,
                             "confidence": 0.6, "reason": f"RSI={v:.1f} 进入超卖"})
            elif prev_state == "normal" and curr_state == "overbought":
                # 首次进入超买 → 预警
                sigs.append({"signal": "SELL", "strength": 0.6,
                             "confidence": 0.6, "reason": f"RSI={v:.1f} 进入超买"})
            else:
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.5, "reason": ""})

            prev_state = curr_state

        result = pd.DataFrame(sigs, index=df_ind.index)
        result.index.name = "Date"
        return result


# ============================================================
# 策略3：MACD 策略
# ============================================================

class MACDStrategy(BaseStrategy):
    """
    MACD 策略：MACD 线与信号线交叉 + 直方图变化。
    """

    name = "MACD"

    @property
    def default_params(self):
        return {
            "fast": 12,
            "slow": 26,
            "signal_period": 9,
        }

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ind = add_indicators(df)
        cols = ["MACD", "MACD_signal", "MACD_hist"]
        if not all(c in df_ind.columns for c in cols):
            return pd.DataFrame({"signal": [], "strength": [], "confidence": [], "reason": []})

        sigs = []
        for i in range(len(df_ind)):
            if i < 1:
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.0, "reason": ""})
                continue

            macd = df_ind["MACD"].iloc[i]
            sig = df_ind["MACD_signal"].iloc[i]
            hist = df_ind["MACD_hist"].iloc[i]
            prev_hist = df_ind["MACD_hist"].iloc[i-1]

            if pd.isna(macd) or pd.isna(sig):
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.0, "reason": ""})
                continue

            # MACD 金叉（转正）+ 直方图扩大 → 买入信号增强
            if macd > sig and prev_hist <= 0 and hist > 0:
                strength = 0.7 + 0.1 * min(abs(hist) / (df_ind["Close"].iloc[i] + 1e-10) * 100, 0.3)
                sigs.append({"signal": "BUY", "strength": min(strength, 1.0),
                             "confidence": 0.7, "reason": f"MACD 金叉 hist={hist:.3f}"})
            elif macd < sig and prev_hist >= 0 and hist < 0:
                strength = 0.7 + 0.1 * min(abs(hist) / (df_ind["Close"].iloc[i] + 1e-10) * 100, 0.3)
                sigs.append({"signal": "SELL", "strength": min(strength, 1.0),
                             "confidence": 0.7, "reason": f"MACD 死叉 hist={hist:.3f}"})
            else:
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.5, "reason": ""})

        result = pd.DataFrame(sigs, index=df_ind.index)
        result.index.name = "Date"
        return result


# ============================================================
# 策略4：布林带策略
# ============================================================

class BollingerStrategy(BaseStrategy):
    """
    布林带策略：价格触及下轨买入，触及上轨卖出。
    """

    name = "BollingerBand"

    @property
    def default_params(self):
        return {"period": 20, "std": 2}

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ind = add_indicators(df)
        required = ["BB_upper", "BB_lower", "BB_mid"]
        if not all(c in df_ind.columns for c in required):
            return pd.DataFrame({"signal": [], "strength": [], "confidence": [], "reason": []})

        sigs = []
        prev_state = "normal"   # normal / below_lower / above_upper
        for i in range(len(df_ind)):
            close = df_ind["Close"].iloc[i]
            upper = df_ind["BB_upper"].iloc[i]
            lower = df_ind["BB_lower"].iloc[i]

            if pd.isna(upper) or pd.isna(lower):
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.0, "reason": ""})
                prev_state = "normal"
                continue

            position = (close - lower) / (upper - lower + 1e-10)

            curr_state = "normal"
            if close <= lower:
                curr_state = "below_lower"
            elif close >= upper:
                curr_state = "above_upper"

            # 状态转换时触发，避免在极值区持续触发（耗尽现金）
            if prev_state == "below_lower" and curr_state == "normal":
                sigs.append({"signal": "BUY", "strength": 0.75,
                             "confidence": 0.75, "reason": "价格脱离布林下轨，RSI确认反弹"})
            elif prev_state == "above_upper" and curr_state == "normal":
                sigs.append({"signal": "SELL", "strength": 0.75,
                             "confidence": 0.75, "reason": "价格脱离布林上轨，回落确认"})
            elif prev_state == "normal" and curr_state == "below_lower":
                sigs.append({"signal": "BUY", "strength": 0.6,
                             "confidence": 0.6, "reason": "首次触及布林下轨，试探买入"})
            elif prev_state == "normal" and curr_state == "above_upper":
                sigs.append({"signal": "SELL", "strength": 0.6,
                             "confidence": 0.6, "reason": "首次触及布林上轨，预警卖出"})
            else:
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.5, "reason": ""})

            prev_state = curr_state

        result = pd.DataFrame(sigs, index=df_ind.index)
        result.index.name = "Date"
        return result


# ============================================================
# 多策略加权合成信号
# ============================================================

class EnsembleStrategy(BaseStrategy):
    """
    多策略加权合成：同时运行多个子策略，按权重合并信号。
    """

    name = "Ensemble"

    def __init__(self, strategies: list[BaseStrategy] = None, weights: list[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.strategies = strategies or []
        n = len(self.strategies)
        self.weights = weights or [1.0 / n] * n
        if len(self.weights) != n:
            self.weights = [1.0 / n] * n

    @property
    def default_params(self):
        return {
            "weights": self.weights,
            "threshold": 0.55,
        }

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.strategies:
            return pd.DataFrame({"signal": ["HOLD"] * len(df),
                                  "strength": [0.0] * len(df),
                                  "confidence": [0.5] * len(df),
                                  "reason": ["无子策略"] * len(df)})

        # 收集各子策略信号
        all_sigs = {}
        for strat in self.strategies:
            sig_df = strat._compute_signals(df)
            all_sigs[strat.name] = sig_df

        # 加权合并
        action_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        score = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for strat, w in zip(self.strategies, self.weights):
            sig_df = all_sigs.get(strat.name)
            if sig_df is None or sig_df.empty:
                continue
            action_scores = sig_df["signal"].map(action_map).fillna(0)
            strength_scores = sig_df["strength"].fillna(0)
            score += (action_scores * strength_scores) * w
            total_weight += w

        if total_weight > 0:
            score /= total_weight

        # 决策
        threshold = self.params.get("threshold", 0.55)
        final_signals = []
        for s in score:
            if s >= threshold:
                final_signals.append({"signal": "BUY", "strength": min(abs(s), 1.0),
                                       "confidence": min(abs(s) + 0.1, 1.0),
                                       "reason": "Ensemble 多策略合成信号"})
            elif s <= -threshold:
                final_signals.append({"signal": "SELL", "strength": min(abs(s), 1.0),
                                       "confidence": min(abs(s) + 0.1, 1.0),
                                       "reason": "Ensemble 多策略合成信号"})
            else:
                final_signals.append({"signal": "HOLD", "strength": 0.0,
                                       "confidence": 0.5, "reason": ""})

        result = pd.DataFrame(final_signals, index=df.index)
        result.index.name = "Date"
        return result
