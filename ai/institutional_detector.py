# ============================================================
# ai/institutional_detector.py — 机构异动检测
# 功能：尾盘异动、大单冲击、成交量异常、价格跳空检测
# 数据原则：真实数据优先，付费数据标注"缺少[数据源]"
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class IntradayAnomaly:
    """盘中异动事件"""
    time: str
    anomaly_type: str  # "volume_spike" / "price_gap" / "close_push" / "opening_drive"
    description: str
    severity: str  # "high" / "medium" / "low"
    volume_ratio: float = 0.0  # 相对近期均量倍数
    price_change_pct: float = 0.0


@dataclass
class InstitutionalResult:
    """机构异动分析结果"""
    ticker: str
    latest_price: float
    analysis_time: str

    # 尾盘分析（最后30/60分钟）
    close_push_detected: bool = False
    close_push_strength: float = 0.0  # 0~100
    close_volume_ratio: float = 0.0
    close_price_change_pct: float = 0.0

    # 开盘分析（前30分钟）
    opening_drive_detected: bool = False
    opening_drive_strength: float = 0.0
    opening_volume_ratio: float = 0.0

    # 全天异常
    volume_anomaly_detected: bool = False
    volume_anomaly_ratio: float = 0.0  # 成交量/20日均量
    price_gap_detected: bool = False
    gap_size_pct: float = 0.0

    # 事件列表
    anomalies: list = field(default_factory=list)

    # 综合判断
    institutional_activity_score: float = 0.0  # 0~100，越高越像机构行为
    activity_direction: str = "neutral"  # "buying" / "selling" / "neutral"
    confidence: float = 0.0

    # 缺少的数据
    missing_data: list = field(default_factory=list)


# ============================================================
# 核心分析函数
# ============================================================

def _get_intraday_data(ticker: str, fetcher) -> Optional[pd.DataFrame]:
    """
    获取日内数据（1分钟或5分钟）。
    yfinance免费版：最近7天1分钟，最近60天5分钟
    """
    try:
        # 尝试1分钟（最近7天）
        df = fetcher.download_history(ticker, period="5d", interval="1m")
        if df is not None and not df.empty and len(df) > 100:
            return df
    except Exception:
        pass

    try:
        # 回退到5分钟（最近60天）
        df = fetcher.download_history(ticker, period="30d", interval="5m")
        if df is not None and not df.empty and len(df) > 50:
            return df
    except Exception:
        pass

    try:
        # 最后回退到15分钟
        df = fetcher.download_history(ticker, period="60d", interval="15m")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    return None


def _analyze_close_push(df: pd.DataFrame) -> tuple[bool, float, float, float]:
    """
    分析尾盘拉升/打压（最后30分钟或最后10%的K线）。
    返回: (是否检测到, 强度0-100, 成交量倍数, 价格变化%)
    """
    if df is None or len(df) < 20:
        return False, 0.0, 0.0, 0.0

    # 取最后30根K线（约30分钟1m / 150分钟5m）
    close_bars = min(30, max(5, len(df) // 10))
    last = df.tail(close_bars)
    before = df.head(len(df) - close_bars).tail(max(20, len(df) // 3))

    if before.empty or last.empty:
        return False, 0.0, 0.0, 0.0

    # 成交量对比
    avg_vol = before["Volume"].mean()
    last_vol = last["Volume"].mean()
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0

    # 价格变化
    start_price = float(last["Close"].iloc[0])
    end_price = float(last["Close"].iloc[-1])
    price_change = (end_price - start_price) / start_price * 100 if start_price > 0 else 0

    # 强度计算
    strength = 0.0
    if vol_ratio > 2.0 and abs(price_change) > 0.3:
        strength = min(100, vol_ratio * 15 + abs(price_change) * 10)
    elif vol_ratio > 1.5 and abs(price_change) > 0.15:
        strength = min(80, vol_ratio * 10 + abs(price_change) * 8)
    elif vol_ratio > 1.2 and abs(price_change) > 0.1:
        strength = min(50, vol_ratio * 8 + abs(price_change) * 5)

    detected = strength > 30
    return detected, strength, vol_ratio, price_change


def _analyze_opening_drive(df: pd.DataFrame) -> tuple[bool, float, float]:
    """
    分析开盘冲击（前30分钟或前10%的K线）。
    返回: (是否检测到, 强度, 成交量倍数)
    """
    if df is None or len(df) < 20:
        return False, 0.0, 0.0

    open_bars = min(30, max(5, len(df) // 10))
    first = df.head(open_bars)
    rest = df.tail(len(df) - open_bars).head(max(20, len(df) // 3))

    if rest.empty or first.empty:
        return False, 0.0, 0.0

    avg_vol = rest["Volume"].mean()
    first_vol = first["Volume"].mean()
    vol_ratio = first_vol / avg_vol if avg_vol > 0 else 0

    start_price = float(first["Open"].iloc[0])
    end_price = float(first["Close"].iloc[-1])
    price_change = (end_price - start_price) / start_price * 100 if start_price > 0 else 0

    strength = 0.0
    if vol_ratio > 2.5 and abs(price_change) > 0.4:
        strength = min(100, vol_ratio * 12 + abs(price_change) * 8)
    elif vol_ratio > 1.8 and abs(price_change) > 0.2:
        strength = min(70, vol_ratio * 10 + abs(price_change) * 6)

    detected = strength > 30
    return detected, strength, vol_ratio


def _analyze_volume_anomaly(df: pd.DataFrame) -> tuple[bool, float]:
    """
    全天成交量异常检测。
    返回: (是否异常, 相对20日均量倍数)
    """
    if df is None or len(df) < 20:
        return False, 0.0

    total_vol = df["Volume"].sum()
    # 用历史日数据估算"正常"日成交量（这里只有日内数据，粗略估算）
    # 假设最近20根K线代表"近期正常"
    recent_avg = df.tail(20)["Volume"].mean() * (len(df) / 20)
    ratio = total_vol / recent_avg if recent_avg > 0 else 0

    # 由于没有历史日均量，标注缺少
    detected = ratio > 2.0
    return detected, ratio


def _detect_price_gap(df: pd.DataFrame) -> tuple[bool, float]:
    """
    检测开盘跳空。
    返回: (是否有跳空, 跳空幅度%)
    """
    if df is None or len(df) < 2:
        return False, 0.0

    # 找到每天的第一根K线
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["date"] = df.index.date

    gaps = []
    prev_close = None
    for date, group in df.groupby("date"):
        if prev_close is not None:
            today_open = float(group["Open"].iloc[0])
            gap = (today_open - prev_close) / prev_close * 100
            gaps.append(gap)
        prev_close = float(group["Close"].iloc[-1])

    if gaps:
        latest_gap = gaps[-1]
        detected = abs(latest_gap) > 0.3
        return detected, latest_gap
    return False, 0.0


# ============================================================
# 主分析入口
# ============================================================

def analyze_institutional_activity(
    ticker: str,
    latest_price: float,
    fetcher,
) -> InstitutionalResult:
    """
    分析机构异动。

    数据源：
      - yfinance 日内数据（1m/5m/15m，免费，最近7-60天）
      - 缺失：Level 2订单簿、逐笔成交、暗池数据（均付费）
    """
    result = InstitutionalResult(
        ticker=ticker,
        latest_price=latest_price,
        analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # 获取日内数据
    df = _get_intraday_data(ticker, fetcher)
    if df is None or df.empty:
        result.missing_data.append("日内数据（yfinance 1m/5m/15m均不可用）")
        result.missing_data.append("Level 2订单簿（付费：CBOE/ Nasdaq TotalView）")
        result.missing_data.append("暗池交易数据（付费：FINRA ATS）")
        return result

    anomalies = []

    # 1. 尾盘分析
    cp_detected, cp_strength, cp_vol_ratio, cp_price = _analyze_close_push(df)
    result.close_push_detected = cp_detected
    result.close_push_strength = round(cp_strength, 1)
    result.close_volume_ratio = round(cp_vol_ratio, 2)
    result.close_price_change_pct = round(cp_price, 2)

    if cp_detected:
        direction = "拉升" if cp_price > 0 else "打压"
        anomalies.append(IntradayAnomaly(
            time="尾盘",
            anomaly_type="close_push",
            description=f"尾盘{direction}，成交量{cp_vol_ratio:.1f}倍均量，价格变动{cp_price:+.2f}%",
            severity="high" if cp_strength > 60 else "medium",
            volume_ratio=cp_vol_ratio,
            price_change_pct=cp_price,
        ))

    # 2. 开盘分析
    od_detected, od_strength, od_vol_ratio = _analyze_opening_drive(df)
    result.opening_drive_detected = od_detected
    result.opening_drive_strength = round(od_strength, 1)
    result.opening_volume_ratio = round(od_vol_ratio, 2)

    if od_detected:
        anomalies.append(IntradayAnomaly(
            time="开盘",
            anomaly_type="opening_drive",
            description=f"开盘冲击，成交量{od_vol_ratio:.1f}倍均量",
            severity="high" if od_strength > 60 else "medium",
            volume_ratio=od_vol_ratio,
            price_change_pct=0.0,
        ))

    # 3. 成交量异常
    va_detected, va_ratio = _analyze_volume_anomaly(df)
    result.volume_anomaly_detected = va_detected
    result.volume_anomaly_ratio = round(va_ratio, 2)

    if va_detected:
        anomalies.append(IntradayAnomaly(
            time="全天",
            anomaly_type="volume_spike",
            description=f"成交量异常，估算比率{va_ratio:.1f}x",
            severity="medium",
            volume_ratio=va_ratio,
        ))

    # 4. 跳空
    gap_detected, gap_size = _detect_price_gap(df)
    result.price_gap_detected = gap_detected
    result.gap_size_pct = round(gap_size, 2)

    if gap_detected:
        gap_type = "向上跳空" if gap_size > 0 else "向下跳空"
        anomalies.append(IntradayAnomaly(
            time="开盘",
            anomaly_type="price_gap",
            description=f"{gap_type} {abs(gap_size):.2f}%",
            severity="high" if abs(gap_size) > 1.0 else "medium",
            price_change_pct=gap_size,
        ))

    result.anomalies = anomalies

    # 综合评分
    score = 0.0
    if cp_detected:
        score += cp_strength * 0.4
    if od_detected:
        score += od_strength * 0.3
    if va_detected:
        score += min(20, va_ratio * 5)
    if gap_detected:
        score += min(15, abs(gap_size) * 5)

    result.institutional_activity_score = round(min(100, score), 1)

    # 方向判断
    net_price = cp_price + gap_size
    if net_price > 0.3 and score > 30:
        result.activity_direction = "buying"
    elif net_price < -0.3 and score > 30:
        result.activity_direction = "selling"
    else:
        result.activity_direction = "neutral"

    result.confidence = min(100, len(anomalies) * 25 + 20)

    # 标注付费缺失数据
    result.missing_data.append("Level 2订单簿（付费：CBOE/Nasdaq TotalView）")
    result.missing_data.append("逐笔成交数据（付费：NYSE OpenBook）")
    result.missing_data.append("暗池交易数据（付费：FINRA ATS）")

    return result


# ============================================================
# 格式化输出
# ============================================================

def format_institutional_result(result: InstitutionalResult) -> dict:
    """格式化机构异动结果为字典（供WebUI）"""
    return {
        "标的": result.ticker,
        "机构活跃度评分": f"{result.institutional_activity_score:.0f}/100",
        "活跃方向": result.activity_direction,
        "置信度": f"{result.confidence:.0f}%",
        "尾盘异动": {
            "检测到": result.close_push_detected,
            "强度": result.close_push_strength,
            "成交量比": result.close_volume_ratio,
            "价格变动": f"{result.close_price_change_pct:+.2f}%",
        },
        "开盘冲击": {
            "检测到": result.opening_drive_detected,
            "强度": result.opening_drive_strength,
            "成交量比": result.opening_volume_ratio,
        },
        "成交量异常": {
            "检测到": result.volume_anomaly_detected,
            "比率": result.volume_anomaly_ratio,
        },
        "开盘跳空": {
            "检测到": result.price_gap_detected,
            "幅度": f"{result.gap_size_pct:+.2f}%",
        },
        "异常事件": [
            {
                "时间": a.time,
                "类型": a.anomaly_type,
                "描述": a.description,
                "严重度": a.severity,
            }
            for a in result.anomalies
        ],
        "缺少数据": result.missing_data,
    }
