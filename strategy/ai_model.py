# ============================================================
# strategy/ai_model.py — AI 量化信号模型
# 机器学习（LGBM）+ 特征工程 + 信号输出
# ============================================================
import os
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from strategy.base import BaseStrategy

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ============================================================
# 特征工程
# ============================================================

def build_features(df: pd.DataFrame, target_horizon: int = 5) -> pd.DataFrame:
    """
    从 OHLCV 构建机器学习特征。

    参数:
        df: OHLCV DataFrame
        target_horizon: 预测未来 N 天涨跌

    返回:
        DataFrame，含特征列和 label 列
    """
    df = df.copy()
    if len(df) < 60:
        return pd.DataFrame()

    # ---- 收益率特征 ----
    for n in [1, 3, 5, 10, 20]:
        df[f"return_{n}d"] = df["Close"].pct_change(n)
        df[f"high_low_{n}d"] = (df["High"].rolling(n).max() - df["Low"].rolling(n).min()) / df["Close"]

    # ---- 均线乖离率 ----
    for w in [5, 10, 20, 50]:
        ma = df["Close"].rolling(w).mean()
        df[f"ma_dev_{w}"] = (df["Close"] - ma) / ma

    # ---- 成交量特征 ----
    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / vol_ma
        df["vol_ma5"] = df["Volume"].rolling(5).mean()

    # ---- 波动率 ----
    ret = df["Close"].pct_change()
    for w in [5, 10, 20]:
        df[f"volatility_{w}d"] = ret.rolling(w).std()

    # ---- RSI ----
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ---- MACD ----
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ---- 布林带 ----
    bb_mean = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_pct"] = (df["Close"] - (bb_mean - 2*bb_std)) / (4*bb_std + 1e-10)

    # ---- 目标变量：未来 N 天收益率（分类标签）----
    df["future_return"] = df["Close"].shift(-target_horizon) / df["Close"] - 1
    df["label"] = (df["future_return"] > 0).astype(int)  # 1=涨，0=跌

    # ---- 去噪：去除 NaN ----
    drop_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
                 "Capital Gains", "future_return"]
    feature_cols = [c for c in df.columns if c not in drop_cols and c != "label"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["label"])

    return df[feature_cols + ["label", "Close"]]


# ============================================================
# AI 策略
# ============================================================

class AIStrategy(BaseStrategy):
    """
    基于 LightGBM 的涨跌预测策略。

    参数:
        model_path: 模型保存路径（None 则每次重新训练）
        feature_window: 训练特征窗口（天数）
        prediction_horizon: 预测 N 天后涨跌
        signal_threshold: 置信度阈值（超过才下单）
        retrain_days: 每隔 N 天自动重训练
    """

    name = "AI_LightGBM"

    @property
    def default_params(self):
        return {
            "model_path": "models/lgbm_model.pkl",
            "feature_window": 60,
            "prediction_horizon": 5,
            "signal_threshold": 0.6,
            "retrain_days": 30,
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: Optional[object] = None
        self.feature_cols: list[str] = []
        self._last_train_date: Optional[str] = None
        self._load_or_train(None)

    def _load_or_train(self, df: Optional[pd.DataFrame]):
        """尝试加载已有模型，若无则准备训练"""
        model_path = self.params.get("model_path")
        if model_path:
            p = Path(model_path)
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        model_data = pickle.load(f)
                    self.model = model_data["model"]
                    self.feature_cols = model_data.get("feature_cols", [])
                    self._last_train_date = model_data.get("train_date", "")
                    logger.info(f"[AI] 加载已有模型: {model_path}")
                except Exception as e:
                    logger.warning(f"[AI] 模型加载失败，将重新训练: {e}")
                    self.model = None

    def _train(self, df: pd.DataFrame):
        """训练 LightGBM 模型"""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("[AI] LightGBM 未安装，请运行: pip install lightgbm")
            return

        df_feat = build_features(df, target_horizon=self.params["prediction_horizon"])
        if df_feat.empty or len(df_feat) < 200:
            logger.warning(f"[AI] 训练数据不足: {len(df_feat)} 条")
            return

        self.feature_cols = [c for c in df_feat.columns if c not in ("label", "Close")]
        X = df_feat[self.feature_cols]
        y = df_feat["label"]

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # 保存模型
        model_path = self.params.get("model_path")
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "feature_cols": self.feature_cols,
                    "train_date": str(pd.Timestamp("today").date()),
                }, f)
            logger.info(f"[AI] 模型已保存: {model_path}")

        # 验证集表现
        proba = self.model.predict(X_test)
        auc = self._auc_score(y_test, proba)
        logger.info(f"[AI] 训练完成，验证集 AUC={auc:.4f}")

    @staticmethod
    def _auc_score(y_true, y_pred) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)
        except Exception:
            return 0.5

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            logger.warning("[AI] 模型未加载，跳过信号生成")
            return pd.DataFrame({
                "signal": ["HOLD"] * len(df),
                "strength": [0.0] * len(df),
                "confidence": [0.5] * len(df),
                "reason": ["模型未就绪"] * len(df),
            })

        df_feat = build_features(df, target_horizon=self.params["prediction_horizon"])
        if df_feat.empty:
            return pd.DataFrame({
                "signal": ["HOLD"] * len(df),
                "strength": [0.0] * len(df),
                "confidence": [0.5] * len(df),
                "reason": ["特征不足"] * len(df),
            })

        # 只对有特征的行做预测
        feat_data = df_feat[self.feature_cols]
        proba = self.model.predict(feat_data)

        # 与 df_feat 对齐（dropna 后行数可能减少）
        sigs = []
        proba_index = 0
        for i in range(len(df)):
            # 找到 df_feat 中对应的 index
            date_idx = df.index[i]
            if date_idx in df_feat.index:
                p = proba[list(df_feat.index).index(date_idx)]
                threshold = self.params["signal_threshold"]
                if p > threshold:
                    sigs.append({
                        "signal": "BUY",
                        "strength": float(p),
                        "confidence": float(p),
                        "reason": f"AI预测看涨概率={p:.3f}",
                    })
                elif p < (1 - threshold):
                    sigs.append({
                        "signal": "SELL",
                        "strength": float(1 - p),
                        "confidence": float(1 - p),
                        "reason": f"AI预测看跌概率={1-p:.3f}",
                    })
                else:
                    sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.5, "reason": ""})
            else:
                sigs.append({"signal": "HOLD", "strength": 0.0, "confidence": 0.5, "reason": ""})

        result = pd.DataFrame(sigs, index=df.index)
        result.index.name = "Date"
        return result

    def train(self, df: pd.DataFrame):
        """手动触发训练"""
        self._train(df)
