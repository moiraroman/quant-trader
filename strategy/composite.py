# ============================================================
# strategy/composite.py — 复合指标策略
# 入场条件：5选3（RSI/SuperTrend/MACD/Momentum/ADX）
# 风控：2x ATR 止损，8x ATR 止盈，RSI>70 自动止盈
# ============================================================
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


# ============================================================
# 扩展指标计算
# ============================================================

def add_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 OHLCV DataFrame 上计算扩展技术指标。
    包括：SuperTrend, ADX, ATR, RSI, MACD, Momentum
    """
    df = df.copy()
    
    # ---- ATR（真实波幅）----
    if len(df) >= 14:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()
    
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
    
    # ---- SuperTrend ----
    if "ATR_14" in df.columns and len(df) >= 14:
        atr = df["ATR_14"]
        hl2 = (df["High"] + df["Low"]) / 2
        
        # 默认因子 3.0
        factor = 3.0
        upper_band = hl2 + factor * atr
        lower_band = hl2 - factor * atr
        
        # SuperTrend 计算
        super_trend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(len(df)):
            if i == 0:
                super_trend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1
            else:
                # 上轨调整：只能下降，不能上升
                if upper_band.iloc[i] < super_trend.iloc[i-1] or df["Close"].iloc[i-1] > super_trend.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
                
                # 下轨调整：只能上升，不能下降
                if lower_band.iloc[i] > super_trend.iloc[i-1] or df["Close"].iloc[i-1] < super_trend.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                
                # 趋势方向
                if df["Close"].iloc[i] > upper_band.iloc[i]:
                    direction.iloc[i] = 1  # 上升趋势
                    super_trend.iloc[i] = lower_band.iloc[i]
                elif df["Close"].iloc[i] < lower_band.iloc[i]:
                    direction.iloc[i] = -1  # 下降趋势
                    super_trend.iloc[i] = upper_band.iloc[i]
                else:
                    direction.iloc[i] = direction.iloc[i-1]
                    super_trend.iloc[i] = super_trend.iloc[i-1]
        
        df["SuperTrend"] = super_trend
        df["SuperTrend_direction"] = direction
        df["SuperTrend_upper"] = upper_band
        df["SuperTrend_lower"] = lower_band
    
    # ---- ADX（平均方向指数）----
    if len(df) >= 14:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        # +DM 和 -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # TR
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        
        # 平滑
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        
        # DX 和 ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        df["ADX"] = dx.rolling(14).mean()
        df["Plus_DI"] = plus_di
        df["Minus_DI"] = minus_di
    
    # ---- 动量 ----
    for n in [5, 10, 20]:
        if len(df) >= n:
            df[f"Momentum_{n}d"] = df["Close"] - df["Close"].shift(n)
    
    return df


# ============================================================
# 复合信号数据结构
# ============================================================

@dataclass
class CompositeSignal(Signal):
    """
    复合策略信号，继承基础 Signal，增加风控参数。
    """
    stop_loss_price: Optional[float] = None     # 止损价
    take_profit_price: Optional[float] = None   # 止盈价
    risk_amount: Optional[float] = None         # 风险金额
    position_size: Optional[float] = None       # 建议仓位
    atr_value: Optional[float] = None           # ATR 值
    conditions_met: List[str] = field(default_factory=list)  # 满足的条件列表


# ============================================================
# 复合指标策略
# ============================================================

class CompositeStrategy(BaseStrategy):
    """
    复合指标策略：满足 3/5 条件时入场。
    
    入场条件（需至少满足 3 个）：
        1. RSI < 40 (超卖)
        2. SuperTrend 金叉（方向转正）
        3. MACD 金叉（MACD > Signal 且直方图转正）
        4. 动量转正（5日动量 > 0）
        5. ADX 趋势过滤（<30 震荡区间 或 >40 强趋势）
    
    风险管理：
        - 6% 风险仓位
        - 2x ATR 止损
        - 8x ATR 止盈 (4:1 盈亏比)
        - RSI > 70 自动止盈
    """
    
    name = "Composite"
    
    @property
    def default_params(self):
        return {
            "min_conditions": 3,          # 最少满足条件数
            "risk_per_trade": 0.06,       # 单笔风险 6%
            "stop_loss_atr": 2.0,         # 止损 ATR 倍数
            "take_profit_atr": 8.0,       # 止盈 ATR 倍数
            "rsi_oversold": 40,           # RSI 超卖阈值
            "rsi_overbought": 70,         # RSI 超买阈值（止盈）
            "adx_range_max": 30,          # ADX 震荡区上限
            "adx_trend_min": 40,          # ADX 强趋势阈值
            "min_confidence": 0.6,        # 最小置信度
        }
    
    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算复合信号。
        """
        df_ind = add_extended_indicators(df)
        
        required_cols = ["RSI_14", "SuperTrend_direction", "MACD", "MACD_signal", 
                         "MACD_hist", "Momentum_5d", "ADX", "ATR_14"]
        if not all(c in df_ind.columns for c in required_cols):
            return pd.DataFrame({
                "signal": [], "strength": [], "confidence": [], "reason": [],
                "stop_loss": [], "take_profit": [], "atr": [], "conditions": []
            })
        
        p = self.params
        sigs = []
        
        for i in range(len(df_ind)):
            if i < 20:  # 需要足够历史数据
                sigs.append({
                    "signal": "HOLD", "strength": 0.0, "confidence": 0.0,
                    "reason": "数据不足", "stop_loss": None, "take_profit": None,
                    "atr": None, "conditions": ""
                })
                continue
            
            row = df_ind.iloc[i]
            prev = df_ind.iloc[i-1] if i > 0 else row
            
            close = row["Close"]
            atr = row["ATR_14"] if pd.notna(row["ATR_14"]) else 0
            rsi = row["RSI_14"] if pd.notna(row["RSI_14"]) else 50
            adx = row["ADX"] if pd.notna(row["ADX"]) else 25
            
            # 检查入场条件
            conditions = []
            condition_scores = []
            
            # 条件1：RSI < 40 (超卖)
            if rsi < p["rsi_oversold"]:
                conditions.append("RSI超卖")
                condition_scores.append(1.0)
            
            # 条件2：SuperTrend 金叉（方向从负转正）
            st_dir = row.get("SuperTrend_direction", 0)
            prev_st_dir = prev.get("SuperTrend_direction", 0) if i > 0 else 0
            if pd.notna(st_dir) and pd.notna(prev_st_dir):
                if prev_st_dir < 0 and st_dir > 0:
                    conditions.append("SuperTrend金叉")
                    condition_scores.append(1.0)
                elif st_dir > 0:
                    conditions.append("SuperTrend上升")
                    condition_scores.append(0.5)  # 已在上升趋势中，给部分分
            
            # 条件3：MACD 金叉
            macd = row.get("MACD", 0)
            macd_sig = row.get("MACD_signal", 0)
            macd_hist = row.get("MACD_hist", 0)
            prev_macd_hist = prev.get("MACD_hist", 0) if i > 0 else 0
            
            if pd.notna(macd) and pd.notna(macd_sig):
                if macd > macd_sig and prev_macd_hist <= 0 and macd_hist > 0:
                    conditions.append("MACD金叉")
                    condition_scores.append(1.0)
                elif macd > macd_sig:
                    conditions.append("MACD多头")
                    condition_scores.append(0.5)
            
            # 条件4：动量转正
            momentum = row.get("Momentum_5d", 0)
            prev_momentum = prev.get("Momentum_5d", 0) if i > 0 else 0
            if pd.notna(momentum):
                if prev_momentum <= 0 and momentum > 0:
                    conditions.append("动量转正")
                    condition_scores.append(1.0)
                elif momentum > 0:
                    conditions.append("动量正")
                    condition_scores.append(0.5)
            
            # 条件5：ADX 趋势过滤
            if pd.notna(adx):
                if adx < p["adx_range_max"]:
                    conditions.append("ADX震荡区")
                    condition_scores.append(1.0)  # 震荡区可反转
                elif adx > p["adx_trend_min"]:
                    conditions.append("ADX强趋势")
                    condition_scores.append(1.0)  # 强趋势顺势
            
            # 计算满足条件数
            full_conditions = sum(1 for s in condition_scores if s >= 1.0)
            partial_conditions = sum(condition_scores)
            
            # 生成信号
            if full_conditions >= p["min_conditions"]:
                # 计算止损止盈
                stop_loss = close - p["stop_loss_atr"] * atr if atr > 0 else None
                take_profit = close + p["take_profit_atr"] * atr if atr > 0 else None
                
                # 置信度基于条件数
                confidence = min(0.5 + 0.1 * full_conditions, 0.95)
                strength = min(partial_conditions / 3.0, 1.0)
                
                reason = f"{full_conditions}/5条件满足: {', '.join(conditions[:3])}"
                
                sigs.append({
                    "signal": "BUY",
                    "strength": strength,
                    "confidence": confidence,
                    "reason": reason,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "atr": atr,
                    "conditions": "|".join(conditions)
                })
            
            # RSI > 70 自动止盈信号
            elif rsi > p["rsi_overbought"]:
                sigs.append({
                    "signal": "SELL",
                    "strength": 0.8,
                    "confidence": 0.8,
                    "reason": f"RSI={rsi:.1f}>70 自动止盈",
                    "stop_loss": None,
                    "take_profit": close,
                    "atr": atr,
                    "conditions": "RSI超买止盈"
                })
            
            else:
                sigs.append({
                    "signal": "HOLD", "strength": 0.0, "confidence": 0.5,
                    "reason": f"仅{full_conditions}/5条件满足",
                    "stop_loss": None, "take_profit": None,
                    "atr": atr, "conditions": "|".join(conditions) if conditions else ""
                })
        
        result = pd.DataFrame(sigs, index=df_ind.index)
        result.index.name = "Date"
        return result
    
    def generate(self, df: pd.DataFrame, ticker: str = "", 
                 total_equity: float = 100000.0) -> CompositeSignal:
        """
        生成复合信号，包含止损止盈价格。
        
        参数:
            df: OHLCV DataFrame
            ticker: 标的代码
            total_equity: 总权益（用于计算仓位）
        
        返回:
            CompositeSignal 对象
        """
        if df.empty or len(df) < 30:
            return CompositeSignal(
                ticker=ticker, action="HOLD", 
                reason="数据不足（需至少30条）"
            )
        
        try:
            sig_df = self._compute_signals(df)
            latest = sig_df.iloc[-1]
            price = df["Close"].iloc[-1]
            
            p = self.params
            atr = latest.get("atr")
            
            # 计算仓位大小（基于风险）
            position_size = None
            risk_amount = None
            if atr and atr > 0 and total_equity > 0:
                risk_amount = total_equity * p["risk_per_trade"]
                # 每股风险 = ATR * 止损倍数
                risk_per_share = atr * p["stop_loss_atr"]
                if risk_per_share > 0:
                    position_size = risk_amount / risk_per_share
            
            return CompositeSignal(
                ticker=ticker or self.params.get("ticker", ""),
                action=latest.get("signal", "HOLD"),
                strength=float(latest.get("strength", 0.5)),
                confidence=float(latest.get("confidence", 0.5)),
                reason=str(latest.get("reason", "")),
                price=price,
                stop_loss_price=latest.get("stop_loss"),
                take_profit_price=latest.get("take_profit"),
                risk_amount=risk_amount,
                position_size=position_size,
                atr_value=atr,
                conditions_met=latest.get("conditions", "").split("|") if latest.get("conditions") else [],
                metadata={"raw": latest.to_dict()},
            )
        
        except Exception as e:
            self._logger.error(f"复合信号计算异常: {e}")
            return CompositeSignal(ticker=ticker, action="HOLD", reason=f"计算异常: {e}")


# ============================================================
# 止盈止损管理器
# ============================================================

class StopLossTakeProfitManager:
    """
    止盈止损管理器。
    
    功能：
    - 追踪持仓的止损止盈价格
    - 实时检查是否触发
    - 支持移动止损（trailing stop）
    """
    
    def __init__(self):
        # 持仓记录：{ticker: {entry_price, stop_loss, take_profit, atr, highest_since_entry}}
        self.positions = {}
    
    def add_position(self, ticker: str, entry_price: float, 
                     stop_loss: float, take_profit: float, atr: float):
        """添加新持仓"""
        self.positions[ticker] = {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr,
            "highest_since_entry": entry_price,
            "trailing_activated": False,
        }
        logger.info(f"[SL/TP] 添加持仓 {ticker}: 入场=${entry_price:.2f}, "
                   f"止损=${stop_loss:.2f}, 止盈=${take_profit:.2f}")
    
    def check_stop(self, ticker: str, current_price: float, 
                   current_rsi: float = None, rsi_threshold: float = 70) -> Tuple[bool, str]:
        """
        检查是否触发止损或止盈。
        
        返回:
            (是否触发, 原因)
        """
        if ticker not in self.positions:
            return False, ""
        
        pos = self.positions[ticker]
        
        # 更新最高价（用于移动止损）
        if current_price > pos["highest_since_entry"]:
            pos["highest_since_entry"] = current_price
        
        # 检查止盈
        if pos["take_profit"] and current_price >= pos["take_profit"]:
            return True, f"止盈触发: ${current_price:.2f} >= ${pos['take_profit']:.2f}"
        
        # RSI > 70 自动止盈
        if current_rsi is not None and current_rsi > rsi_threshold:
            return True, f"RSI止盈: RSI={current_rsi:.1f} > {rsi_threshold}"
        
        # 检查止损
        if pos["stop_loss"] and current_price <= pos["stop_loss"]:
            return True, f"止损触发: ${current_price:.2f} <= ${pos['stop_loss']:.2f}"
        
        return False, ""
    
    def update_trailing_stop(self, ticker: str, atr_mult: float = 2.0):
        """
        更新移动止损。
        当价格创新高时，将止损上移。
        """
        if ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        atr = pos.get("atr", 0)
        
        if atr <= 0:
            return
        
        # 计算新止损：最高价 - ATR * 倍数
        new_stop = pos["highest_since_entry"] - atr_mult * atr
        
        # 只能上移，不能下移
        if new_stop > pos["stop_loss"]:
            old_stop = pos["stop_loss"]
            pos["stop_loss"] = new_stop
            pos["trailing_activated"] = True
            logger.info(f"[SL/TP] {ticker} 移动止损: ${old_stop:.2f} -> ${new_stop:.2f}")
    
    def remove_position(self, ticker: str):
        """移除持仓"""
        if ticker in self.positions:
            del self.positions[ticker]
    
    def get_position(self, ticker: str) -> dict:
        """获取持仓信息"""
        return self.positions.get(ticker)
    
    def get_all_positions(self) -> dict:
        """获取所有持仓"""
        return self.positions.copy()
