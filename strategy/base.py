# ============================================================
# strategy/base.py — 策略基类
# 所有策略必须继承 BaseStrategy，统一信号格式，方便切换
# ============================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 信号数据结构
# ============================================================

@dataclass
class Signal:
    """
    策略产生的交易信号。

    Attributes:
        ticker: 标的代码
        action: BUY / SELL / HOLD
        strength: 信号强度 0.0~1.0（用于多策略加权）
        confidence: 置信度 0.0~1.0（用于阈值过滤）
        reason: 信号原因说明
        price: 参考价格（可选）
        metadata: 额外元数据
    """
    ticker: str
    action: str            # BUY / SELL / HOLD
    strength: float = 0.5  # 0.0 ~ 1.0
    confidence: float = 0.5
    reason: str = ""
    price: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def is_actionable(self, threshold: float = 0.6) -> bool:
        """信号是否值得执行（置信度 > 阈值）"""
        return self.confidence >= threshold and self.action != "HOLD"

    def __repr__(self):
        return (f"Signal({self.ticker} {self.action} "
                f"conf={self.confidence:.2f} reason={self.reason[:30]})")


# ============================================================
# 策略基类
# ============================================================

class BaseStrategy(ABC):
    """
    所有策略的抽象基类。

    子类必须实现：
        _compute_signals() -> pd.Series

    通用接口：
        generate(df: pd.DataFrame) -> Signal
        get_params() -> dict
        set_params(**kwargs)
    """

    name: str = "BaseStrategy"

    def __init__(self, **kwargs):
        """
        初始化参数。

        子类在 __init__ 中应先调用 super().__init__(**kwargs)，
        本方法会自动合并 default_params 与传入参数。
        """
        defaults = getattr(self, 'default_params', {})
        self.params = {**defaults, **kwargs}
        self._logger = logging.getLogger(f"{__name__}.{self.name}")

    # ---- 抽象方法 ----

    @abstractmethod
    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        核心计算逻辑。接收 OHLCV DataFrame，返回信号 DataFrame。

        返回格式（列名固定）：
            signal: BUY / SELL / HOLD
            strength: 0.0 ~ 1.0
            confidence: 0.0 ~ 1.0
            reason: str
        """
        raise NotImplementedError

    # ---- 通用接口 ----

    def generate(self, df: pd.DataFrame, ticker: str = "") -> Signal:
        """
        主入口：给定最新行情，返回一个 Signal 对象。

        参数:
            df: OHLCV DataFrame，列: Open/High/Low/Close/Volume
            ticker: 标的代码

        返回:
            Signal 对象
        """
        if df.empty or len(df) < 20:
            return Signal(ticker=ticker, action="HOLD", reason="数据不足")

        try:
            sig_df = self._compute_signals(df)
            latest = sig_df.iloc[-1]
            price = df["Close"].iloc[-1] if "Close" in df.columns else None

            return Signal(
                ticker=ticker or self.params.get("ticker", ""),
                action=latest.get("signal", "HOLD"),
                strength=float(latest.get("strength", 0.5)),
                confidence=float(latest.get("confidence", 0.5)),
                reason=str(latest.get("reason", "")),
                price=price,
                metadata={"raw": latest.to_dict()},
            )
        except Exception as e:
            self._logger.error(f"信号计算异常: {e}")
            return Signal(ticker=ticker, action="HOLD", reason=f"计算异常: {e}")

    def batch_generate(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """批量对多个标的生成信号"""
        return {ticker: self.generate(df, ticker) for ticker, df in data.items()}

    # ---- 参数管理 ----

    @property
    def default_params(self) -> dict:
        """子类覆盖：返回默认参数 dict"""
        return {}

    def get_params(self) -> dict:
        return self.params.copy()

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def __repr__(self):
        return f"{self.name}({self.params})"
