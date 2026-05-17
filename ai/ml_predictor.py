# ============================================================
# ai/ml_predictor.py — 机器学习预测模块
# 基于技术指标特征集训练简单模型，输出分类/回归预测
# ============================================================
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入sklearn，如未安装则降级为简单统计方法
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("[ML] sklearn未安装，使用统计降级方案")


@dataclass
class MLPredictionResult:
    """机器学习预测结果"""
    ticker: str
    current_price: float
    # 分类预测（上涨/下跌/横盘）
    direction_prediction: str = ""  # "up" / "down" / "neutral"
    direction_confidence: float = 0.0
    direction_probs: dict = field(default_factory=dict)  # {"up": 0.5, "down": 0.3, "neutral": 0.2}
    # 回归预测（目标价）
    target_price_5d: float = 0.0
    target_price_20d: float = 0.0
    target_confidence: float = 0.0
    # 特征重要性
    feature_importance: dict = field(default_factory=dict)
    # 模型性能
    model_accuracy: float = 0.0
    model_precision: float = 0.0
    model_recall: float = 0.0
    # 使用的特征
    features_used: list = field(default_factory=list)
    # 训练数据范围
    train_start: str = ""
    train_end: str = ""
    # 缺失数据
    missing_data: list = field(default_factory=list)
    # 免责声明
    disclaimer: str = "机器学习预测基于历史模式，不构成投资建议，市场结构变化可能导致模型失效"


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建技术指标特征集。
    返回特征DataFrame（与价格数据对齐）。
    """
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data.get("Volume", pd.Series(np.ones(len(close)), index=close.index))

    features = pd.DataFrame(index=data.index)

    # 价格动量
    features["returns_1d"] = close.pct_change()
    features["returns_5d"] = close.pct_change(5)
    features["returns_10d"] = close.pct_change(10)
    features["returns_20d"] = close.pct_change(20)

    # 均线位置
    sma5 = close.rolling(5).mean()
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    features["close_sma20"] = close / sma20 - 1
    features["close_sma50"] = close / sma50 - 1
    features["sma5_sma20"] = sma5 / sma20 - 1
    features["sma10_sma50"] = sma10 / sma50 - 1

    # 波动率
    features["volatility_5d"] = close.pct_change().rolling(5).std()
    features["volatility_20d"] = close.pct_change().rolling(20).std()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    # 布林带
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features["bb_position"] = (close - sma20) / (2 * std20)

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean() / close

    # 成交量
    vol_sma20 = volume.rolling(20).mean()
    features["volume_ratio"] = volume / vol_sma20
    features["volume_change"] = volume.pct_change()

    # 价格位置（日内）
    features["intraday_position"] = (close - low) / (high - low + 1e-10)

    return features.dropna()


def _build_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    构建标签：未来N天收益率分类。
    >2%: up, <-2%: down, else: neutral
    """
    future_returns = df["Close"].shift(-horizon) / df["Close"] - 1
    labels = future_returns.apply(lambda x: "up" if x > 0.02 else ("down" if x < -0.02 else "neutral"))
    return labels


def predict_with_ml(
    ticker: str,
    current_price: float,
    fetcher,
    prediction_horizon: int = 5,  # 5天或20天
    model_type: str = "random_forest",  # "random_forest" / "logistic" / "statistical"
) -> MLPredictionResult:
    """
    使用机器学习模型预测未来价格方向。

    参数:
        ticker: 标的代码
        current_price: 当前价格
        fetcher: YFinanceFetcher实例
        prediction_horizon: 预测周期（5或20天）
        model_type: 模型类型

    返回:
        MLPredictionResult
    """
    result = MLPredictionResult(ticker=ticker, current_price=current_price)

    # 获取历史数据（至少2年）
    try:
        df = fetcher.download_history(ticker, period="3y", interval="1d")
        if df.empty or len(df) < 252:
            result.missing_data.append("历史数据不足（需至少1年）")
            return result
    except Exception as e:
        logger.warning(f"[ML] {ticker} 获取历史数据失败: {e}")
        result.missing_data.append(f"历史数据获取失败: {e}")
        return result

    result.train_start = str(df.index[0])
    result.train_end = str(df.index[-1])

    # 构建特征和标签
    features = _build_features(df)
    labels = _build_labels(df, horizon=prediction_horizon)

    # 对齐
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # 移除缺失值
    valid_mask = ~features.isnull().any(axis=1) & ~labels.isnull()
    features = features[valid_mask]
    labels = labels[valid_mask]

    if len(features) < 100:
        result.missing_data.append("有效训练样本不足（需至少100条）")
        return result

    result.features_used = list(features.columns)

    # 划分训练/测试集
    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = labels.iloc[:split_idx]
    y_test = labels.iloc[split_idx:]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    if not SKLEARN_AVAILABLE or model_type == "statistical":
        # 降级方案：基于近期动量的简单统计
        return _statistical_prediction(result, features, labels, current_price, prediction_horizon)

    try:
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        result.model_accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)

        # 特征重要性
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(features.columns, model.feature_importances_))
            result.feature_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        elif hasattr(model, "coef_"):
            # LogisticRegression
            importance = dict(zip(features.columns, np.abs(model.coef_[0])))
            result.feature_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        # 预测当前状态
        latest_features = scaler.transform(features.iloc[-1:].values)
        pred_proba = model.predict_proba(latest_features)[0]
        pred_class = model.predict(latest_features)[0]

        classes = model.classes_
        result.direction_probs = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, pred_proba)}
        result.direction_prediction = pred_class
        result.direction_confidence = round(float(max(pred_proba)) * 100, 1)

        # 回归预测目标价（使用RandomForestRegressor）
        future_returns = df["Close"].shift(-prediction_horizon) / df["Close"] - 1
        future_returns = future_returns.loc[common_idx][valid_mask]
        y_reg_train = future_returns.iloc[:split_idx]
        y_reg_test = future_returns.iloc[split_idx:]

        reg_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        reg_model.fit(X_train_scaled, y_reg_train)
        pred_return = reg_model.predict(latest_features)[0]

        if prediction_horizon == 5:
            result.target_price_5d = round(current_price * (1 + pred_return), 2)
        else:
            result.target_price_20d = round(current_price * (1 + pred_return), 2)
        result.target_confidence = result.direction_confidence

    except Exception as e:
        logger.warning(f"[ML] {ticker} 模型训练失败: {e}")
        result.missing_data.append(f"模型训练失败: {e}")
        return _statistical_prediction(result, features, labels, current_price, prediction_horizon)

    logger.info(f"[ML] {ticker} 预测完成: 方向{result.direction_prediction}, 置信度{result.direction_confidence}%")
    return result


def _statistical_prediction(
    result: MLPredictionResult,
    features: pd.DataFrame,
    labels: pd.Series,
    current_price: float,
    horizon: int,
) -> MLPredictionResult:
    """统计降级方案：基于近期动量和RSI的简单预测"""
    latest = features.iloc[-1]

    # 基于RSI和动量综合评分
    rsi = latest.get("rsi", 50)
    ret_5d = latest.get("returns_5d", 0)
    ret_20d = latest.get("returns_20d", 0)
    macd_hist = latest.get("macd_hist", 0)
    bb_pos = latest.get("bb_position", 0)

    bull_score = 0
    bear_score = 0

    if rsi < 30:
        bull_score += 3
    elif rsi < 40:
        bull_score += 1
    elif rsi > 70:
        bear_score += 3
    elif rsi > 60:
        bear_score += 1

    if ret_5d > 0.05:
        bear_score += 2  # 短期涨幅过大，回调风险
    elif ret_5d < -0.05:
        bull_score += 2  # 短期跌幅过大，反弹机会

    if ret_20d > 0.1:
        bear_score += 1
    elif ret_20d < -0.1:
        bull_score += 1

    if macd_hist > 0:
        bull_score += 1
    else:
        bear_score += 1

    if bb_pos > 0.8:
        bear_score += 1
    elif bb_pos < -0.8:
        bull_score += 1

    total = bull_score + bear_score
    if total == 0:
        up_prob = 33.3
        down_prob = 33.3
        neutral_prob = 33.4
        pred = "neutral"
    else:
        up_prob = round(bull_score / total * 100, 1)
        down_prob = round(bear_score / total * 100, 1)
        neutral_prob = round(max(0, 100 - up_prob - down_prob), 1)
        if bull_score > bear_score + 1:
            pred = "up"
        elif bear_score > bull_score + 1:
            pred = "down"
        else:
            pred = "neutral"

    result.direction_prediction = pred
    result.direction_probs = {"up": up_prob, "down": down_prob, "neutral": neutral_prob}
    result.direction_confidence = round(max(up_prob, down_prob, neutral_prob), 1)
    result.model_accuracy = 0.0  # 统计方法无准确率
    result.features_used = ["rsi", "returns_5d", "returns_20d", "macd_hist", "bb_position"]
    result.feature_importance = {
        "rsi": 0.25,
        "returns_5d": 0.20,
        "returns_20d": 0.15,
        "macd_hist": 0.20,
        "bb_position": 0.20,
    }

    # 简单目标价估算（基于历史平均动量）
    avg_momentum = features["returns_5d"].mean() if horizon == 5 else features["returns_20d"].mean()
    target = current_price * (1 + avg_momentum)
    if horizon == 5:
        result.target_price_5d = round(target, 2)
    else:
        result.target_price_20d = round(target, 2)

    result.missing_data.append("使用统计降级方案（sklearn未安装或模型失败）")
    return result


def format_ml_result(result: MLPredictionResult) -> dict:
    """格式化ML预测结果为UI展示格式"""
    return {
        "标的": result.ticker,
        "当前价格": result.current_price,
        "方向预测": {
            "预测方向": result.direction_prediction,
            "置信度": f"{result.direction_confidence}%",
            "概率分布": result.direction_probs,
        },
        "目标价预测": {
            "5日目标价": result.target_price_5d if result.target_price_5d > 0 else "N/A",
            "20日目标价": result.target_price_20d if result.target_price_20d > 0 else "N/A",
        },
        "模型性能": {
            "测试集准确率": f"{result.model_accuracy}%" if result.model_accuracy > 0 else "统计方法无准确率",
        },
        "特征重要性": result.feature_importance,
        "使用特征": result.features_used[:10],
        "训练数据": f"{result.train_start} 至 {result.train_end}",
        "免责声明": result.disclaimer,
        "缺少数据": result.missing_data,
    }
