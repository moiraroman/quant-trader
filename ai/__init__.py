# ============================================================
# ai/__init__.py
# ============================================================
from .market_analyzer import MarketAnalyzer
from .stock_screener import StockScreener
from .orchestrator import AIQuantAnalyst

# 情绪分析模块（可选，确保可导入）
try:
    from . import sentiment
except Exception:
    sentiment = None

__all__ = ["MarketAnalyzer", "StockScreener", "AIQuantAnalyst", "sentiment"]
