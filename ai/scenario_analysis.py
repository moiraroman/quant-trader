# ============================================================
# ai/scenario_analysis.py — 情景分析（基准/乐观/悲观）
# 功能：基于当前技术面和宏观面，构建三情景概率分布
# ============================================================
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """单个情景"""
    name: str                # "基准"/"乐观"/"悲观"
    probability: float       # 概率 (%)
    description: str         # 情景描述
    # 价格目标
    target_5d: float = 0.0   # 5日目标价
    target_20d: float = 0.0  # 20日目标价
    target_5d_pct: float = 0.0
    target_20d_pct: float = 0.0
    # 关键假设
    assumptions: list = field(default_factory=list)
    # 触发条件
    triggers: list = field(default_factory=list)
    # 风险
    risks: list = field(default_factory=list)


@dataclass
class ScenarioResult:
    """情景分析结果"""
    ticker: str
    analysis_time: str
    current_price: float

    # 三情景
    base_case: Scenario = None
    bull_case: Scenario = None
    bear_case: Scenario = None

    # 概率分布校验
    total_probability: float = 0.0

    # 综合期望收益
    expected_return_5d: float = 0.0
    expected_return_20d: float = 0.0
    expected_max_dd_5d: float = 0.0

    # 关键变量敏感性
    key_variables: list = field(default_factory=list)

    # 情景树提示
    scenario_tree: str = ""

    missing_data: list = field(default_factory=list)


# ============================================================
# 情景构建逻辑
# ============================================================

def _calculate_scenario_probabilities(
    tech_score: float,      # 技术面评分 0-100
    macro_score: float,     # 宏观评分 0-100
    vix: float,
    trend: str,             # "bullish"/"bearish"/"neutral"
    pattern_match_winrate: float = 50.0,  # 模式匹配胜率
) -> tuple[float, float, float]:
    """
    计算三情景概率。

    基础概率（等权）：
      基准 = 50%, 乐观 = 25%, 悲观 = 25%

    调整因子：
      - 技术面偏强 → 乐观+，悲观-
      - 宏观偏强 → 乐观+，悲观-
      - VIX高 → 基准+，乐观-/悲观+
      - 模式匹配胜率高 → 相应方向+
    """
    base_p = 50.0
    bull_p = 25.0
    bear_p = 25.0

    # 技术面调整
    if tech_score > 65:
        bull_p += 10
        bear_p -= 10
    elif tech_score < 35:
        bear_p += 10
        bull_p -= 10

    # 宏观调整
    if macro_score > 65:
        bull_p += 5
        bear_p -= 5
    elif macro_score < 35:
        bear_p += 5
        bull_p -= 5

    # VIX调整（高VIX增加不确定性，基准概率上升）
    if vix > 30:
        base_p += 10
        bull_p -= 5
        bear_p -= 5
    elif vix < 15:
        base_p -= 5
        bull_p += 3
        bear_p += 2

    # 趋势方向
    if trend == "bullish":
        bull_p += 5
        bear_p -= 5
    elif trend == "bearish":
        bear_p += 5
        bull_p -= 5

    # 模式匹配胜率
    if pattern_match_winrate > 60:
        # 高胜率偏向该方向
        diff = (pattern_match_winrate - 50) / 2
        bull_p += diff
        bear_p -= diff
    elif pattern_match_winrate < 40:
        diff = (50 - pattern_match_winrate) / 2
        bear_p += diff
        bull_p -= diff

    # 归一化到100%
    total = base_p + bull_p + bear_p
    if total > 0:
        base_p = base_p / total * 100
        bull_p = bull_p / total * 100
        bear_p = bear_p / total * 100

    return round(base_p, 1), round(bull_p, 1), round(bear_p, 1)


def _build_bull_case(
    current_price: float,
    tech_score: float,
    macro_score: float,
    atr_pct: float,
    sr_resistance: float,
) -> Scenario:
    """构建乐观情景"""
    # 5日目标：当前 + 1.5~2.5 ATR
    move_5d = max(1.5 * atr_pct, 1.0) if atr_pct > 0 else 2.0
    target_5d = current_price * (1 + move_5d / 100)

    # 20日目标：当前 + 4~6 ATR 或接近阻力位
    move_20d = max(4 * atr_pct, 3.0) if atr_pct > 0 else 5.0
    if sr_resistance > current_price:
        # 阻力位作为上限
        target_20d = min(current_price * (1 + move_20d / 100), sr_resistance * 1.01)
    else:
        target_20d = current_price * (1 + move_20d / 100)

    assumptions = [
        "技术面持续走强，多头信号得到确认",
        "宏观环境改善或维持向好",
        "市场情绪偏向乐观，资金持续流入",
    ]

    triggers = [
        "价格突破近期阻力位",
        "成交量配合放大",
        "宏观数据超预期",
    ]

    risks = [
        "阻力位遇阻回落",
        "突发宏观利空",
        "市场情绪快速逆转",
    ]

    return Scenario(
        name="乐观",
        probability=0.0,  # 后续填充
        description=f"技术+宏观双重支撑，价格向上拓展空间",
        target_5d=round(target_5d, 2),
        target_20d=round(target_20d, 2),
        target_5d_pct=round((target_5d - current_price) / current_price * 100, 2),
        target_20d_pct=round((target_20d - current_price) / current_price * 100, 2),
        assumptions=assumptions,
        triggers=triggers,
        risks=risks,
    )


def _build_base_case(
    current_price: float,
    tech_score: float,
    macro_score: float,
    atr_pct: float,
) -> Scenario:
    """构建基准情景"""
    # 5日目标：当前 ± 0.5 ATR
    move_5d = 0.5 * atr_pct if atr_pct > 0 else 0.5
    # 偏向当前趋势方向
    bias = (tech_score - 50) / 100 * move_5d
    target_5d = current_price * (1 + bias / 100)

    # 20日目标：当前 ± 1.5 ATR
    move_20d = 1.5 * atr_pct if atr_pct > 0 else 1.5
    bias_20d = (tech_score - 50) / 100 * move_20d
    target_20d = current_price * (1 + bias_20d / 100)

    assumptions = [
        "当前趋势延续，无重大变化",
        "技术面和宏观面无显著偏离",
        "市场按既有路径运行",
    ]

    triggers = [
        "价格维持当前区间",
        "宏观数据符合预期",
        "成交量正常",
    ]

    risks = [
        "技术面信号失效",
        "宏观预期改变",
        "波动率意外上升",
    ]

    return Scenario(
        name="基准",
        probability=0.0,
        description=f"趋势延续，价格在当前区间波动",
        target_5d=round(target_5d, 2),
        target_20d=round(target_20d, 2),
        target_5d_pct=round((target_5d - current_price) / current_price * 100, 2),
        target_20d_pct=round((target_20d - current_price) / current_price * 100, 2),
        assumptions=assumptions,
        triggers=triggers,
        risks=risks,
    )


def _build_bear_case(
    current_price: float,
    tech_score: float,
    macro_score: float,
    atr_pct: float,
    sr_support: float,
) -> Scenario:
    """构建悲观情景"""
    # 5日目标：当前 - 1.5~2.5 ATR
    move_5d = max(1.5 * atr_pct, 1.0) if atr_pct > 0 else 2.0
    target_5d = current_price * (1 - move_5d / 100)

    # 20日目标：当前 - 4~6 ATR 或接近支撑位
    move_20d = max(4 * atr_pct, 3.0) if atr_pct > 0 else 5.0
    if sr_support > 0 and sr_support < current_price:
        target_20d = max(current_price * (1 - move_20d / 100), sr_support * 0.99)
    else:
        target_20d = current_price * (1 - move_20d / 100)

    assumptions = [
        "技术面转弱，空头信号显现",
        "宏观环境恶化或出现逆风",
        "避险情绪升温，资金流出",
    ]

    triggers = [
        "价格跌破关键支撑位",
        "成交量放大下跌",
        "宏观数据不及预期",
    ]

    risks = [
        "支撑位有效反弹",
        "政策干预或利好",
        "超跌反弹",
    ]

    return Scenario(
        name="悲观",
        probability=0.0,
        description=f"技术+宏观双重压力，价格向下寻求支撑",
        target_5d=round(target_5d, 2),
        target_20d=round(target_20d, 2),
        target_5d_pct=round((target_5d - current_price) / current_price * 100, 2),
        target_20d_pct=round((target_20d - current_price) / current_price * 100, 2),
        assumptions=assumptions,
        triggers=triggers,
        risks=risks,
    )


# ============================================================
# 主分析入口
# ============================================================

def analyze_scenarios(
    ticker: str,
    current_price: float,
    tech_score: float,
    macro_score: float,
    vix: float,
    trend: str,
    atr_pct: float,
    sr_support: float = 0.0,
    sr_resistance: float = 0.0,
    pattern_match_winrate: float = 50.0,
) -> ScenarioResult:
    """
    三情景分析。

    参数:
        tech_score: 技术面综合评分 0-100
        macro_score: 宏观评分 0-100
        vix: 当前VIX水平
        trend: 当前趋势方向
        atr_pct: ATR百分比
        sr_support/resistance: 支撑/阻力位
        pattern_match_winrate: 模式匹配历史胜率
    """
    from datetime import datetime

    result = ScenarioResult(
        ticker=ticker,
        analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        current_price=current_price,
    )

    # 计算概率
    base_p, bull_p, bear_p = _calculate_scenario_probabilities(
        tech_score, macro_score, vix, trend, pattern_match_winrate
    )
    result.total_probability = base_p + bull_p + bear_p

    # 构建三情景
    bull = _build_bull_case(current_price, tech_score, macro_score, atr_pct, sr_resistance)
    bull.probability = bull_p

    base = _build_base_case(current_price, tech_score, macro_score, atr_pct)
    base.probability = base_p

    bear = _build_bear_case(current_price, tech_score, macro_score, atr_pct, sr_support)
    bear.probability = bear_p

    result.bull_case = bull
    result.base_case = base
    result.bear_case = bear

    # 期望收益
    result.expected_return_5d = round(
        (bull.target_5d_pct * bull_p + base.target_5d_pct * base_p + bear.target_5d_pct * bear_p) / 100,
        2
    )
    result.expected_return_20d = round(
        (bull.target_20d_pct * bull_p + base.target_20d_pct * base_p + bear.target_20d_pct * bear_p) / 100,
        2
    )

    # 5日期望最大回撤（悲观情景为主）
    result.expected_max_dd_5d = round(-1.5 * atr_pct if atr_pct > 0 else -2.0, 2)

    # 关键变量
    result.key_variables = [
        f"VIX水平 (当前{vix})",
        f"技术面评分 (当前{tech_score:.0f})",
        f"宏观评分 (当前{macro_score:.0f})",
        f"支撑位 {sr_support:.2f}" if sr_support > 0 else "支撑位未识别",
        f"阻力位 {sr_resistance:.2f}" if sr_resistance > 0 else "阻力位未识别",
    ]

    # 情景树提示
    result.scenario_tree = (
        f"如果突破阻力位→乐观({bull_p}%)→目标{ bull.target_20d:.2f}; "
        f"如果维持区间→基准({base_p}%)→目标{ base.target_20d:.2f}; "
        f"如果跌破支撑位→悲观({bear_p}%)→目标{ bear.target_20d:.2f}"
    )

    return result


# ============================================================
# 格式化输出
# ============================================================

def format_scenario_result(result: ScenarioResult) -> dict:
    """格式化情景分析结果为字典（供WebUI）"""
    def _fmt_scenario(s: Scenario):
        if s is None:
            return {}
        return {
            "情景": s.name,
            "概率": f"{s.probability:.0f}%",
            "描述": s.description,
            "5日目标": f"{s.target_5d:.2f} ({s.target_5d_pct:+.2f}%)",
            "20日目标": f"{s.target_20d:.2f} ({s.target_20d_pct:+.2f}%)",
            "关键假设": s.assumptions,
            "触发条件": s.triggers,
            "风险": s.risks,
        }

    return {
        "标的": result.ticker,
        "当前价格": result.current_price,
        "概率分布": {
            "乐观": f"{result.bull_case.probability:.0f}%" if result.bull_case else "N/A",
            "基准": f"{result.base_case.probability:.0f}%" if result.base_case else "N/A",
            "悲观": f"{result.bear_case.probability:.0f}%" if result.bear_case else "N/A",
            "合计": f"{result.total_probability:.0f}%",
        },
        "期望收益": {
            "5日": f"{result.expected_return_5d:+.2f}%",
            "20日": f"{result.expected_return_20d:+.2f}%",
            "5日期望最大回撤": f"{result.expected_max_dd_5d:.2f}%",
        },
        "情景详情": {
            "乐观": _fmt_scenario(result.bull_case),
            "基准": _fmt_scenario(result.base_case),
            "悲观": _fmt_scenario(result.bear_case),
        },
        "关键变量": result.key_variables,
        "情景树": result.scenario_tree,
        "缺少数据": result.missing_data,
    }
