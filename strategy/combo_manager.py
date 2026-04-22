# ============================================================
# strategy/combo_manager.py — 策略组合管理器
# 支持多策略组合、参数配置、保存/加载
# ============================================================
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 策略参数配置
# ============================================================

@dataclass
class StrategyConfig:
    """单个策略的配置"""
    name: str                          # 策略名称
    enabled: bool = True               # 是否启用
    weight: float = 1.0                # 权重（用于投票）
    params: dict = field(default_factory=dict)  # 策略参数
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "StrategyConfig":
        return cls(**data)


# ============================================================
# 策略组合配置
# ============================================================

@dataclass
class ComboConfig:
    """策略组合配置"""
    name: str                                    # 组合名称
    strategies: list[StrategyConfig]             # 策略列表
    created_at: str = ""                         # 创建时间
    description: str = ""                        # 描述
    # 风控参数
    risk_per_trade: float = 0.06                 # 单笔风险
    stop_loss_atr: float = 2.0                   # 止损 ATR 倍数
    take_profit_atr: float = 8.0                 # 止盈 ATR 倍数
    min_vote_count: int = 2                      # 最少投票数
    vote_threshold: float = 0.6                  # 投票阈值
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "strategies": [s.to_dict() for s in self.strategies],
            "created_at": self.created_at,
            "description": self.description,
            "risk_per_trade": self.risk_per_trade,
            "stop_loss_atr": self.stop_loss_atr,
            "take_profit_atr": self.take_profit_atr,
            "min_vote_count": self.min_vote_count,
            "vote_threshold": self.vote_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ComboConfig":
        strategies = [StrategyConfig.from_dict(s) for s in data.get("strategies", [])]
        return cls(
            name=data.get("name", "未命名"),
            strategies=strategies,
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
            risk_per_trade=data.get("risk_per_trade", 0.06),
            stop_loss_atr=data.get("stop_loss_atr", 2.0),
            take_profit_atr=data.get("take_profit_atr", 8.0),
            min_vote_count=data.get("min_vote_count", 2),
            vote_threshold=data.get("vote_threshold", 0.6),
        )


# ============================================================
# 策略组合管理器
# ============================================================

class ComboManager:
    """
    管理策略组合的保存、加载、删除。
    配置文件存储在 data/combos/ 目录下。
    """
    
    # 可用策略列表及其默认参数
    AVAILABLE_STRATEGIES = {
        "RSI": {
            "period": 14,
            "oversold": 30,
            "overbought": 70,
        },
        "MA交叉": {
            "short_window": 10,
            "long_window": 30,
        },
        "MACD": {
            "fast": 12,
            "slow": 26,
            "signal": 9,
        },
        "布林带": {
            "period": 20,
            "std_dev": 2.0,
        },
        "复合指标": {
            "min_conditions": 3,
            "use_rsi": True,
            "use_supertrend": True,
            "use_macd": True,
            "use_momentum": True,
            "use_adx": True,
        },
    }
    
    def __init__(self, config_dir: str = "data/combos"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def list_combos(self) -> list[str]:
        """列出所有已保存的组合"""
        combos = list(self.config_dir.glob("*.json"))
        return [f.stem for f in combos]
    
    def load_combo(self, name: str) -> Optional[ComboConfig]:
        """加载组合配置"""
        path = self.config_dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ComboConfig.from_dict(data)
        except Exception as e:
            logger.error(f"加载组合 {name} 失败: {e}")
            return None
    
    def save_combo(self, combo: ComboConfig) -> bool:
        """保存组合配置"""
        if not combo.created_at:
            combo.created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        path = self.config_dir / f"{combo.name}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(combo.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"组合 {combo.name} 已保存")
            return True
        except Exception as e:
            logger.error(f"保存组合 {combo.name} 失败: {e}")
            return False
    
    def delete_combo(self, name: str) -> bool:
        """删除组合配置"""
        path = self.config_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            logger.info(f"组合 {name} 已删除")
            return True
        return False
    
    def get_default_combo(self) -> ComboConfig:
        """获取默认组合（MACD + RSI）"""
        return ComboConfig(
            name="默认组合",
            strategies=[
                StrategyConfig(name="MACD", enabled=True, weight=1.0, params={"fast": 12, "slow": 26, "signal": 9}),
                StrategyConfig(name="RSI", enabled=True, weight=1.0, params={"period": 14, "oversold": 30, "overbought": 70}),
            ],
            description="MACD + RSI 双策略组合",
            risk_per_trade=0.06,
            stop_loss_atr=2.0,
            take_profit_atr=8.0,
            min_vote_count=1,  # 只要1个策略触发就执行
        )


# ============================================================
# 组合策略信号生成器
# ============================================================

class ComboSignalGenerator:
    """
    根据组合配置生成综合信号。
    支持投票机制：多个策略投票决定最终信号。
    """
    
    def __init__(self, combo: ComboConfig):
        self.combo = combo
        self._strategies = {}
        self._init_strategies()
    
    def _init_strategies(self):
        """初始化策略实例"""
        from strategy.technical import MAStrategy, RSIStrategy, MACDStrategy, BollingerStrategy
        from strategy.composite import CompositeStrategy
        
        strategy_classes = {
            "RSI": RSIStrategy,
            "MA交叉": MAStrategy,
            "MACD": MACDStrategy,
            "布林带": BollingerStrategy,
            "复合指标": CompositeStrategy,
        }
        
        for strat_config in self.combo.strategies:
            if strat_config.enabled and strat_config.name in strategy_classes:
                cls = strategy_classes[strat_config.name]
                self._strategies[strat_config.name] = cls(**strat_config.params)
    
    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算组合信号。
        
        投票机制：
        1. 每个启用的策略计算自己的信号
        2. BUY = +1, SELL = -1, HOLD = 0
        3. 加权求和后归一化
        4. 超过阈值则执行
        """
        if not self._strategies:
            return pd.DataFrame(index=df.index, columns=["signal", "confidence", "strength", "reason"])
        
        # 收集所有策略的信号
        all_signals = {}
        for name, strat in self._strategies.items():
            try:
                sig_df = strat._compute_signals(df)
                all_signals[name] = sig_df
            except Exception as e:
                logger.warning(f"策略 {name} 计算失败: {e}")
        
        if not all_signals:
            return pd.DataFrame(index=df.index, columns=["signal", "confidence", "strength", "reason"])
        
        # 投票
        result = pd.DataFrame(index=df.index)
        result["vote_score"] = 0.0
        result["vote_count"] = 0
        result["buy_count"] = 0
        result["sell_count"] = 0
        result["reasons"] = ""
        
        # 找到对应的权重
        weights = {}
        for strat_config in self.combo.strategies:
            weights[strat_config.name] = strat_config.weight
        
        for name, sig_df in all_signals.items():
            weight = weights.get(name, 1.0)
            for idx in sig_df.index:
                signal = sig_df.loc[idx, "signal"] if "signal" in sig_df.columns else "HOLD"
                confidence = sig_df.loc[idx, "confidence"] if "confidence" in sig_df.columns else 0.5
                reason = sig_df.loc[idx, "reason"] if "reason" in sig_df.columns else ""
                
                # 投票
                vote = 0
                if signal == "BUY":
                    vote = weight * confidence
                    result.loc[idx, "buy_count"] += 1
                elif signal == "SELL":
                    vote = -weight * confidence
                    result.loc[idx, "sell_count"] += 1
                
                result.loc[idx, "vote_score"] += vote
                result.loc[idx, "vote_count"] += 1
                
                # 记录原因
                if signal in ("BUY", "SELL"):
                    current_reason = result.loc[idx, "reasons"]
                    result.loc[idx, "reasons"] = f"{current_reason}{name}:{signal}; "
        
        # 确定最终信号
        def determine_signal(row):
            vote_score = row["vote_score"]
            vote_count = row["vote_count"]
            buy_count = row["buy_count"]
            sell_count = row["sell_count"]
            
            if vote_count == 0:
                return "HOLD", 0.0, 0.0, "无有效信号"
            
            # 归一化投票分数
            normalized = vote_score / vote_count
            
            # 多数投票模式：买方多则买，卖方多则卖
            # 同时检查最小投票数
            if buy_count >= self.combo.min_vote_count and buy_count > sell_count:
                confidence = buy_count / vote_count
                return "BUY", confidence, confidence, row["reasons"]
            elif sell_count >= self.combo.min_vote_count and sell_count > buy_count:
                confidence = sell_count / vote_count
                return "SELL", confidence, confidence, row["reasons"]
            elif buy_count >= self.combo.min_vote_count and buy_count == sell_count and buy_count > 0:
                # 平票时看 vote_score 方向
                if normalized > 0:
                    confidence = buy_count / vote_count
                    return "BUY", confidence, confidence, f"平票决胜(加权): {row['reasons']}"
                elif normalized < 0:
                    confidence = sell_count / vote_count
                    return "SELL", confidence, confidence, f"平票决胜(加权): {row['reasons']}"
            
            return "HOLD", abs(normalized), max(buy_count, sell_count) / vote_count, f"投票不足 (买:{buy_count}/卖:{sell_count})"
        
        signals_data = result.apply(determine_signal, axis=1)
        result["signal"] = signals_data.apply(lambda x: x[0])
        result["strength"] = signals_data.apply(lambda x: x[1])
        result["confidence"] = signals_data.apply(lambda x: x[2])
        result["reason"] = signals_data.apply(lambda x: x[3])
        
        return result[["signal", "confidence", "strength", "reason"]]
