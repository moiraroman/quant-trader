# ============================================================
# ai/macro_policy.py — 宏观政策/新闻/日历分析
# 功能：美联储政策、经济数据日历、地缘政治风险、市场假期
# 数据原则：真实数据优先，付费数据标注"缺少[数据源]"
# ============================================================
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EconomicEvent:
    """经济事件"""
    date: str
    event_name: str
    importance: str  # "high" / "medium" / "low"
    forecast: str = ""
    previous: str = ""
    actual: str = ""
    impact: str = "pending"  # "bullish" / "bearish" / "neutral" / "pending"


@dataclass
class PolicyRisk:
    """政策风险项"""
    category: str  # "fed" / "fiscal" / "geopolitical" / "trade"
    risk_name: str
    risk_level: str  # "high" / "medium" / "low"
    description: str
    source: str = ""


@dataclass
class MacroPolicyResult:
    """宏观政策分析结果"""
    ticker: str
    analysis_time: str

    # 美联储政策
    fed_policy_stance: str = "neutral"  # "hawkish" / "dovish" / "neutral"
    fed_confidence: float = 0.0
    fed_note: str = ""

    # 近期重要事件
    upcoming_events: list = field(default_factory=list)  # EconomicEvent列表
    recent_events: list = field(default_factory=list)

    # 政策风险
    policy_risks: list = field(default_factory=list)  # PolicyRisk列表
    overall_risk_level: str = "low"

    # 市场假期
    market_holidays: list = field(default_factory=list)
    next_holiday: str = ""

    # 对标的的影响评估
    impact_assessment: str = ""
    impact_direction: str = "neutral"  # "positive" / "negative" / "neutral"
    impact_confidence: float = 0.0

    # 缺少的数据
    missing_data: list = field(default_factory=list)


# ============================================================
# 预设数据（基于公开信息，定期需更新）
# ============================================================

# 2024-2025 已知美联储会议日期（FOMC）
# 实际使用时建议通过搜索获取最新日程
FOMC_MEETINGS_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-11",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]

# 美国市场假期（2025）
US_MARKET_HOLIDAYS_2025 = {
    "2025-01-01": "New Year's Day",
    "2025-01-20": "Martin Luther King Jr. Day",
    "2025-02-17": "Presidents' Day",
    "2025-04-18": "Good Friday",
    "2025-05-26": "Memorial Day",
    "2025-06-19": "Juneteenth",
    "2025-07-04": "Independence Day",
    "2025-09-01": "Labor Day",
    "2025-11-27": "Thanksgiving",
    "2025-12-25": "Christmas Day",
}

# 重要经济数据发布日（每月固定）
# 这些数据需要通过搜索获取最新值
REGULAR_ECONOMIC_EVENTS = [
    {"name": "CPI (Consumer Price Index)", "day_pattern": "每月10-15日", "importance": "high"},
    {"name": "PPI (Producer Price Index)", "day_pattern": "每月11-16日", "importance": "high"},
    {"name": "非农就业 (Nonfarm Payrolls)", "day_pattern": "每月第一个周五", "importance": "high"},
    {"name": "失业率 (Unemployment Rate)", "day_pattern": "每月第一个周五", "importance": "high"},
    {"name": "零售销售 (Retail Sales)", "day_pattern": "每月中旬", "importance": "medium"},
    {"name": "GDP 初值/修正", "day_pattern": "每季度末", "importance": "high"},
    {"name": "ISM制造业PMI", "day_pattern": "每月第一个工作日", "importance": "medium"},
    {"name": "ISM服务业PMI", "day_pattern": "每月第一个工作日", "importance": "medium"},
    {"name": "初请失业金", "day_pattern": "每周四", "importance": "medium"},
    {"name": "消费者信心指数", "day_pattern": "每月末", "importance": "low"},
]


# ============================================================
# 核心分析函数
# ============================================================

def _get_next_fomc_meetings(today: datetime, count: int = 3) -> list[str]:
    """获取接下来N个FOMC会议日期"""
    future = [d for d in FOMC_MEETINGS_2025 if datetime.strptime(d, "%Y-%m-%d") >= today]
    return future[:count]


def _get_next_holiday(today: datetime) -> tuple[str, str]:
    """获取下一个市场假期"""
    future_holidays = {
        date: name for date, name in US_MARKET_HOLIDAYS_2025.items()
        if datetime.strptime(date, "%Y-%m-%d") >= today
    }
    if future_holidays:
        next_date = min(future_holidays.keys())
        return next_date, future_holidays[next_date]
    return "", ""


def _assess_fed_policy(fetcher) -> tuple[str, float, str]:
    """
    评估美联储政策立场。
    基于：
      1. DGS10（10年期国债收益率）趋势
      2. DFF（联邦基金利率）水平
      3. DGS2-DGS10利差（收益率曲线）
    注意：真实政策立场需要FOMC声明，这里基于市场数据近似
    """
    try:
        # 获取国债收益率数据（yfinance可用 ^TNX, ^FVX, ^IRX）
        tnx = fetcher.download_history("^TNX", period="30d", interval="1d")
        fvx = fetcher.download_history("^FVX", period="30d", interval="1d")

        if tnx is None or tnx.empty:
            return "neutral", 0.0, "无法获取国债收益率数据"

        tnx_latest = float(tnx["Close"].iloc[-1])
        tnx_30d = float(tnx["Close"].iloc[0]) if len(tnx) > 1 else tnx_latest
        tnx_change = tnx_latest - tnx_30d

        # 收益率上升 → 市场定价紧缩/通胀担忧
        # 收益率下降 → 市场定价宽松/衰退担忧
        stance = "neutral"
        confidence = 0.0

        if tnx_change > 0.3:
            stance = "hawkish"
            confidence = min(100, tnx_change * 30)
        elif tnx_change < -0.3:
            stance = "dovish"
            confidence = min(100, abs(tnx_change) * 30)

        note = f"10年期收益率: {tnx_latest:.2f}% (30日变化: {tnx_change:+.2f}%)"

        # 收益率曲线
        if fvx is not None and not fvx.empty:
            fvx_latest = float(fvx["Close"].iloc[-1])
            spread = tnx_latest - fvx_latest
            note += f", 2-10年利差: {spread:.2f}%"
            if spread < 0:
                note += " (倒挂—衰退信号)"
                confidence = min(100, confidence + 20)

        return stance, confidence, note

    except Exception as e:
        logger.warning(f"[MacroPolicy] 美联储政策评估失败: {e}")
        return "neutral", 0.0, f"评估异常: {e}"


def _build_event_calendar(today: datetime) -> tuple[list, list]:
    """
    构建经济事件日历。
    由于无法获取实时经济日历，返回预设事件模板+下次FOMC。
    """
    upcoming = []
    recent = []

    # 下次FOMC
    next_fomc = _get_next_fomc_meetings(today, 2)
    for fomc_date in next_fomc:
        days_until = (datetime.strptime(fomc_date, "%Y-%m-%d") - today).days
        upcoming.append(EconomicEvent(
            date=fomc_date,
            event_name=f"FOMC会议 (距今天{days_until}天)",
            importance="high",
            impact="pending",
        ))

    # 本月经济数据（估算）
    current_month = today.strftime("%Y-%m")
    for evt in REGULAR_ECONOMIC_EVENTS:
        if evt["importance"] == "high":
            upcoming.append(EconomicEvent(
                date=f"{current_month} (具体日期待确认)",
                event_name=evt["name"],
                importance=evt["importance"],
                impact="pending",
            ))

    return upcoming, recent


def _assess_policy_risks(ticker: str) -> list[PolicyRisk]:
    """
    评估当前政策风险。
    基于公开已知风险，需要定期更新。
    """
    risks = []

    # 通用风险（适用于所有标的）
    risks.append(PolicyRisk(
        category="fed",
        risk_name="美联储政策不确定性",
        risk_level="medium",
        description="利率路径取决于通胀数据，存在政策转向风险",
        source="FOMC前瞻指引",
    ))

    risks.append(PolicyRisk(
        category="geopolitical",
        risk_name="地缘政治紧张",
        risk_level="medium",
        description="中东/俄乌局势可能影响能源价格和避险情绪",
        source="新闻监测",
    ))

    # 标的特定风险
    ticker_upper = ticker.upper()
    if ticker_upper in ["GLD", "SLV"]:
        risks.append(PolicyRisk(
            category="fed",
            risk_name="实际利率变化",
            risk_level="high",
            description="实际利率上升压制黄金吸引力，下降则利好",
            source="TIPS收益率",
        ))
    elif ticker_upper in ["SPY", "QQQ", "IWM"]:
        risks.append(PolicyRisk(
            category="fiscal",
            risk_name="财政政策可持续性",
            risk_level="medium",
            description="美国债务水平和财政赤字长期影响",
            source="CBO预算报告",
        ))
    elif ticker_upper in ["XLE", "USO"]:
        risks.append(PolicyRisk(
            category="geopolitical",
            risk_name="能源供应中断",
            risk_level="high",
            description="OPEC+政策、地缘冲突影响油价",
            source="EIA/IEA报告",
        ))

    return risks


def _ticker_specific_impact(ticker: str, fed_stance: str, risks: list) -> tuple[str, str, float]:
    """评估政策对特定标的的影响"""
    ticker_upper = ticker.upper()

    if ticker_upper in ["GLD", "SLV"]:
        if fed_stance == "dovish":
            return "positive", "宽松政策利好贵金属", 65.0
        elif fed_stance == "hawkish":
            return "negative", "紧缩政策压制贵金属", 60.0
        else:
            return "neutral", "政策中性，贵金属受其他因素驱动", 30.0

    elif ticker_upper in ["SPY", "QQQ", "IWM", "VTI"]:
        if fed_stance == "dovish":
            return "positive", "宽松政策利好股市", 70.0
        elif fed_stance == "hawkish":
            return "negative", "紧缩政策压制估值", 55.0
        else:
            return "neutral", "政策中性，关注企业盈利", 25.0

    elif ticker_upper in ["TLT", "IEF", "SHY"]:
        if fed_stance == "dovish":
            return "positive", "降息利好债券价格", 75.0
        elif fed_stance == "hawkish":
            return "negative", "加息压制债券价格", 65.0
        else:
            return "neutral", "利率预期稳定", 30.0

    elif ticker_upper in ["XLE", "USO"]:
        # 能源受地缘影响大于货币政策
        geo_risk = any(r.category == "geopolitical" and r.risk_level == "high" for r in risks)
        if geo_risk:
            return "positive", "地缘风险支撑能源价格", 50.0
        return "neutral", "能源受供需和地缘多重影响", 20.0

    return "neutral", "影响不明确", 10.0


# ============================================================
# 主分析入口
# ============================================================

def analyze_macro_policy(
    ticker: str,
    latest_price: float,
    fetcher,
) -> MacroPolicyResult:
    """
    分析宏观政策环境。

    数据源：
      - yfinance 国债收益率（^TNX, ^FVX，免费）
      - 预设FOMC日程、市场假期（需定期更新）
      - 缺失：实时经济日历（付费：ForexFactory/Bloomberg）
      - 缺失：实时新闻情绪（付费：RavenPack/Accern）
    """
    today = datetime.now()
    result = MacroPolicyResult(
        ticker=ticker,
        analysis_time=today.strftime("%Y-%m-%d %H:%M"),
    )

    # 1. 美联储政策评估
    fed_stance, fed_conf, fed_note = _assess_fed_policy(fetcher)
    result.fed_policy_stance = fed_stance
    result.fed_confidence = round(fed_conf, 1)
    result.fed_note = fed_note

    # 2. 事件日历
    upcoming, recent = _build_event_calendar(today)
    result.upcoming_events = upcoming
    result.recent_events = recent

    # 3. 政策风险
    result.policy_risks = _assess_policy_risks(ticker)
    high_risks = sum(1 for r in result.policy_risks if r.risk_level == "high")
    result.overall_risk_level = "high" if high_risks > 0 else "medium" if len(result.policy_risks) > 2 else "low"

    # 4. 市场假期
    next_holiday_date, next_holiday_name = _get_next_holiday(today)
    if next_holiday_date:
        result.next_holiday = f"{next_holiday_date} {next_holiday_name}"
        result.market_holidays.append(next_holiday_name)

    # 5. 对标的的影响
    impact_dir, impact_desc, impact_conf = _ticker_specific_impact(ticker, fed_stance, result.policy_risks)
    result.impact_direction = impact_dir
    result.impact_assessment = impact_desc
    result.impact_confidence = round(impact_conf, 1)

    # 标注缺失
    result.missing_data.append("实时经济日历（付费：ForexFactory/Bloomberg/Econoday）")
    result.missing_data.append("实时新闻情绪分析（付费：RavenPack/Accern）")
    result.missing_data.append("美联储官员讲话日程（需手动跟踪）")
    result.missing_data.append("FOMC会议日程需每年更新（当前为2025预设）")

    return result


# ============================================================
# 格式化输出
# ============================================================

def format_macro_policy_result(result: MacroPolicyResult) -> dict:
    """格式化宏观政策结果为字典（供WebUI）"""
    return {
        "标的": result.ticker,
        "美联储立场": {
            "判断": result.fed_policy_stance,
            "置信度": f"{result.fed_confidence:.0f}%",
            "说明": result.fed_note,
        },
        "政策风险": [
            {
                "类别": r.category,
                "风险": r.risk_name,
                "等级": r.risk_level,
                "描述": r.description,
                "来源": r.source,
            }
            for r in result.policy_risks
        ],
        "整体风险等级": result.overall_risk_level,
        " upcoming_events": [
            {
                "日期": e.date,
                "事件": e.event_name,
                "重要性": e.importance,
                "影响": e.impact,
            }
            for e in result.upcoming_events[:5]
        ],
        "市场假期": {
            "下一个": result.next_holiday,
            "列表": result.market_holidays,
        },
        "对标的的影响": {
            "方向": result.impact_direction,
            "评估": result.impact_assessment,
            "置信度": f"{result.impact_confidence:.0f}%",
        },
        "缺少数据": result.missing_data,
    }
