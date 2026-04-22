# ============================================================
# ai/sector_rotation.py — 板块轮动分析模块
# 检测市场资金在不同GICS板块间的流动，识别轮动信号
# ============================================================
import os
import sys
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

# ============================================================
# 板块 ETF 映射（GICS 11大板块）
# ============================================================

SECTOR_ETFS = {
    "XLK": {"name": "Technology", "name_zh": "科技", "name_ja": "テクノロジー"},
    "XLF": {"name": "Financials", "name_zh": "金融", "name_ja": "金融"},
    "XLE": {"name": "Energy", "name_zh": "能源", "name_ja": "エネルギー"},
    "XLV": {"name": "Healthcare", "name_zh": "医疗保健", "name_ja": "ヘルスケア"},
    "XLI": {"name": "Industrials", "name_zh": "工业", "name_ja": "インダストリアル"},
    "XLP": {"name": "Consumer Staples", "name_zh": "必需消费品", "name_ja": "生活必需品"},
    "XLY": {"name": "Consumer Discretionary", "name_zh": "可选消费", "name_ja": "一般消費財"},
    "XLB": {"name": "Materials", "name_zh": "原材料", "name_ja": "素材"},
    "XLU": {"name": "Utilities", "name_zh": "公用事业", "name_ja": "公益事業"},
    "XLRE": {"name": "Real Estate", "name_zh": "房地产", "name_ja": "不動産"},
    "XLC": {"name": "Communication Services", "name_zh": "通信服务", "name_ja": "通信サービス"},
}

# 防御型板块（经济不确定时表现好）
DEFENSIVE_SECTORS = {"XLU", "XLP", "XLV", "XLRE"}

# 进攻型板块（经济扩张期表现好）
OFFENSIVE_SECTORS = {"XLK", "XLY", "XLE", "XLC", "XLF", "XLI"}


# ============================================================
# 数据类
# ============================================================

class SectorPerformance:
    """
    单个板块表现指标

    Attributes:
        ticker: ETF代码
        name: 板块英文名
        name_zh: 板块中文名
        name_ja: 板块日文名
        returns_5d: 5日收益率（%）
        returns_20d: 20日收益率（%）
        returns_60d: 60日收益率（%）
        momentum_score: 动量综合得分（0-100）
        volume_change: 成交量变化率（%）
        relative_strength: 相对强度指标
    """

    def __init__(
        self,
        ticker: str,
        name: str,
        name_zh: str = "",
        name_ja: str = "",
        returns_5d: float = 0.0,
        returns_20d: float = 0.0,
        returns_60d: float = 0.0,
        momentum_score: float = 50.0,
        volume_change: float = 0.0,
        relative_strength: float = 0.0,
    ):
        self.ticker = ticker
        self.name = name
        self.name_zh = name_zh
        self.name_ja = name_ja
        self.returns_5d = returns_5d
        self.returns_20d = returns_20d
        self.returns_60d = returns_60d
        self.momentum_score = momentum_score
        self.volume_change = volume_change
        self.relative_strength = relative_strength

    def to_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "name_zh": self.name_zh,
            "name_ja": self.name_ja,
            "returns_5d": round(self.returns_5d, 2),
            "returns_20d": round(self.returns_20d, 2),
            "returns_60d": round(self.returns_60d, 2),
            "momentum_score": round(self.momentum_score, 2),
            "volume_change": round(self.volume_change, 2),
            "relative_strength": round(self.relative_strength, 2),
        }


class RotationSignal:
    """
    板块轮动信号

    Attributes:
        type: 信号类型
            - "defensive": 资金转向防御板块
            - "offensive": 资金转向进攻板块
            - "neutral": 无明显轮动
        strength: 信号强度（0.0-1.0）
        description: 信号描述
        sectors_involved: 涉及的板块列表
    """

    def __init__(
        self,
        type: str,
        strength: float,
        description: str,
        sectors_involved: Optional[List[str]] = None,
    ):
        self.type = type  # "defensive" | "offensive" | "neutral"
        self.strength = strength
        self.description = description
        self.sectors_involved = sectors_involved or []

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "strength": round(self.strength, 2),
            "description": self.description,
            "sectors_involved": self.sectors_involved,
        }


class SectorRotationResult:
    """
    板块轮动分析结果

    Attributes:
        sectors: 各板块表现列表
        ranking: 强弱排名（从强到弱，ticker列表）
        rotation_signals: 轮动信号列表
        market_phase: 市场阶段
            - "early_bull": 早期牛市（经济复苏，周期性板块领涨）
            - "late_bull": 晚期牛市（防御板块开始走强预警）
            - "bear": 熊市（防御板块持续领跑）
            - "recovery": 恢复期（金融/能源领涨）
        summary: 分析摘要
        analysis_steps: 详细分析步骤列表
        clusters: 聚类结果（相似走势板块归组）
        timestamp: 分析时间戳
    """

    def __init__(
        self,
        sectors: List[SectorPerformance],
        ranking: List[str],
        rotation_signals: List[RotationSignal],
        market_phase: str,
        summary: str,
        analysis_steps: List[str],
        clusters: Optional[List[List[str]]] = None,
    ):
        self.sectors = sectors
        self.ranking = ranking
        self.rotation_signals = rotation_signals
        self.market_phase = market_phase
        self.summary = summary
        self.analysis_steps = analysis_steps
        self.clusters = clusters or []
        self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        return {
            "sectors": [s.to_dict() for s in self.sectors],
            "ranking": self.ranking,
            "rotation_signals": [s.to_dict() for s in self.rotation_signals],
            "market_phase": self.market_phase,
            "summary": self.summary,
            "analysis_steps": self.analysis_steps,
            "clusters": self.clusters,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================
# 板块轮动分析器
# ============================================================

class SectorRotationAnalyzer:
    """
    板块轮动分析器

    通过分析11大GICS板块ETF的相对强弱，
    检测市场资金在不同板块间的流动，识别轮动信号，
    并判断当前市场阶段（牛市早期/晚期/熊市/恢复期）。

    示例:
        >>> from data.fetcher import YFinanceFetcher
        >>> fetcher = YFinanceFetcher()
        >>> analyzer = SectorRotationAnalyzer(data_fetcher=fetcher)
        >>> result = analyzer.analyze(period="6mo")
        >>> print(result.summary)
    """

    def __init__(self, data_fetcher=None):
        """
        Args:
            data_fetcher: 数据获取器（YFinanceFetcher 实例），若为 None 则自动创建
        """
        self.fetcher = data_fetcher
        self._sector_etfs = SECTOR_ETFS

    # ---- 公开接口 ----

    def analyze(self, period: str = "6mo") -> SectorRotationResult:
        """
        执行完整的板块轮动分析

        Args:
            period: 分析周期，yfinance 支持的格式，默认 "6mo"

        Returns:
            SectorRotationResult 实例
        """
        logger.info(f"[SectorRotation] 开始板块轮动分析，周期={period}")

        # 步骤1: 获取所有板块ETF数据
        analysis_steps = [f"1. 获取 {len(self._sector_etfs)} 个板块ETF数据 (周期={period})"]
        performances = []
        failed_tickers = []

        for ticker in self._sector_etfs:
            try:
                df = self._fetch_sector_data(ticker, period)
                if df.empty:
                    failed_tickers.append(ticker)
                    continue
                perf = self._calculate_performance(df, ticker)
                performances.append(perf)
            except Exception as e:
                logger.warning(f"[SectorRotation] {ticker} 分析失败: {e}")
                failed_tickers.append(ticker)

        if len(performances) < 3:
            logger.error("[SectorRotation] 有效板块数据不足，无法分析")
            return self._default_result()

        # 步骤2: 计算相对强弱排名
        analysis_steps.append(f"2. 计算板块相对强弱 ({len(performances)} 个有效板块)")
        ranking = self._calculate_ranking(performances)
        analysis_steps.append(f"   排名: {' > '.join(ranking)}")

        # 步骤3: 检测轮动信号
        analysis_steps.append("3. 检测板块轮动信号")
        rotation_signals = self._detect_rotation(performances)
        analysis_steps.append(f"   检测到 {len(rotation_signals)} 个信号")

        # 步骤4: 判断市场阶段
        analysis_steps.append("4. 判断市场阶段")
        market_phase = self._determine_market_phase(ranking, performances)
        analysis_steps.append(f"   市场阶段: {market_phase}")

        # 步骤5: 聚类分析（相似走势归组）
        analysis_steps.append("5. 执行板块聚类分析（Hierarchical Clustering）")
        clusters = self._cluster_sectors(performances, period)

        # 步骤6: 生成分析摘要
        summary = self._generate_summary(
            performances, ranking, rotation_signals, market_phase, clusters
        )
        analysis_steps.append(f"6. 生成分析摘要")

        logger.info(f"[SectorRotation] 分析完成，市场阶段={market_phase}")

        return SectorRotationResult(
            sectors=performances,
            ranking=ranking,
            rotation_signals=rotation_signals,
            market_phase=market_phase,
            summary=summary,
            analysis_steps=analysis_steps,
            clusters=clusters,
        )

    def get_sector_table(self, result: SectorRotationResult) -> pd.DataFrame:
        """
        生成板块分析结果表格（方便打印或导出）

        Args:
            result: SectorRotationResult 实例

        Returns:
            DataFrame，各列: 排名/ticker/名称/5日%/%20日%/60日%/动量/量能变化/阶段分类
        """
        rows = []
        for rank, ticker in enumerate(result.ranking, 1):
            perf = next(s for s in result.sectors if s.ticker == ticker)
            phase = self._classify_sector_phase(perf)
            rows.append({
                "排名": rank,
                "代码": ticker,
                "板块": f"{perf.name_zh}({perf.name})",
                "5日%": f"{perf.returns_5d:+.2f}%",
                "20日%": f"{perf.returns_20d:+.2f}%",
                "60日%": f"{perf.returns_60d:+.2f}%",
                "动量": f"{perf.momentum_score:.1f}",
                "量能变化%": f"{perf.volume_change:+.2f}%",
                "分类": phase,
            })
        return pd.DataFrame(rows)

    # ---- 内部方法 ----

    def _fetch_sector_data(self, ticker: str, period: str) -> pd.DataFrame:
        """
        获取单个板块ETF的历史数据

        Args:
            ticker: ETF代码
            period: 时间周期

        Returns:
            DataFrame，列: Open/High/Low/Close/Volume
        """
        if self.fetcher is None:
            from data.fetcher import YFinanceFetcher
            self.fetcher = YFinanceFetcher()

        df = self.fetcher.download_history(ticker, period=period, interval="1d")
        return df

    def _calculate_performance(self, df: pd.DataFrame, ticker: str) -> SectorPerformance:
        """
        计算单个板块的多维度表现指标

        Args:
            df: 历史行情 DataFrame
            ticker: ETF代码

        Returns:
            SectorPerformance 实例
        """
        close = df["Close"]
        volume = df["Volume"]

        # 计算收益率
        returns_5d = self._calc_return(close, 5)
        returns_20d = self._calc_return(close, 20)
        returns_60d = self._calc_return(close, 60)

        # 动量综合得分（加权平均：短期30%，中期40%，长期30%）
        momentum_score = (
            returns_5d * 0.3
            + returns_20d * 0.4
            + returns_60d * 0.3
        )
        # 归一化到 0-100，以各板块中位数为基准
        # 简单处理：+5%以上→强，-5%以下→弱
        momentum_score = 50 + momentum_score  # 中性偏移

        # 成交量变化（最近10日均值 vs 更早期20日均值）
        vol_recent = volume.iloc[-10:].mean()
        vol_older = volume.iloc[-30:-10].mean()
        volume_change = ((vol_recent / vol_older) - 1) * 100 if vol_older > 0 else 0

        # 相对强弱（相对于SPY基准，简单用自身60日动量代替）
        relative_strength = returns_60d

        info = self._sector_etfs.get(ticker, {})
        return SectorPerformance(
            ticker=ticker,
            name=info.get("name", ticker),
            name_zh=info.get("name_zh", ""),
            name_ja=info.get("name_ja", ""),
            returns_5d=round(returns_5d, 4),
            returns_20d=round(returns_20d, 4),
            returns_60d=round(returns_60d, 4),
            momentum_score=round(momentum_score, 2),
            volume_change=round(volume_change, 2),
            relative_strength=round(relative_strength, 2),
        )

    def _calc_return(self, series: pd.Series, periods: int) -> float:
        """计算N日前至今的收益率（%）。数据不足时返回0。"""
        if len(series) <= periods:
            return 0.0
        current = series.iloc[-1]
        past = series.iloc[-periods]
        return ((current / past) - 1) * 100

    def _calculate_ranking(self, performances: List[SectorPerformance]) -> List[str]:
        """
        根据动量综合得分对板块排序（从强到弱）

        Args:
            performances: 各板块表现列表

        Returns:
            排序后的 ticker 列表
        """
        sorted_perfs = sorted(
            performances,
            key=lambda p: p.momentum_score,
            reverse=True
        )
        return [p.ticker for p in sorted_perfs]

    def _detect_rotation(
        self, performances: List[SectorPerformance]
    ) -> List[RotationSignal]:
        """
        检测板块轮动信号

        核心逻辑:
        - 防御板块（XLU/XLP/XLV/XLRE）排名↑ + 进攻板块排名↓ = defensive switch
        - 金融(XLF)/能源(XLE)领涨 = 经济复苏信号
        - 科技(XLK)领涨 + 公用事业(XLU)也涨 = 晚期牛市预警

        Args:
            performances: 各板块表现列表

        Returns:
            RotationSignal 列表
        """
        signals = []
        ranking = self._calculate_ranking(performances)

        # 防御板块和进攻板块的排名
        def get_rank(ticker: str) -> int:
            try:
                return ranking.index(ticker) + 1
            except ValueError:
                return len(ranking) + 1

        # --- 信号1: 防御轮动 ---
        defensive_tickers = [t for t in ranking if t in DEFENSIVE_SECTORS]
        offensive_tickers = [t for t in ranking if t in OFFENSIVE_SECTORS]

        if defensive_tickers and offensive_tickers:
            # 防御板块最好排名 vs 进攻板块最好排名
            best_defensive_rank = get_rank(defensive_tickers[0])
            best_offensive_rank = get_rank(offensive_tickers[0])

            # 防御板块进入前3且进攻板块跌出前3
            if best_defensive_rank <= 3 and best_offensive_rank > 5:
                # 计算强度：防御越靠前越强
                strength = min(1.0, (6 - best_defensive_rank) / 3 * 0.8 + 0.2)
                signals.append(RotationSignal(
                    type="defensive",
                    strength=strength,
                    description=(
                        f"资金轮动至防御板块：{defensive_tickers[0]} "
                        f"排名 #{best_defensive_rank}，"
                        f"最佳进攻板块 {offensive_tickers[0]} 排名 #{best_offensive_rank}"
                    ),
                    sectors_involved=[defensive_tickers[0], offensive_tickers[0]],
                ))

            # 进攻板块领涨（前3中有2个以上进攻板块）
            offensive_in_top3 = sum(1 for t in ranking[:3] if t in OFFENSIVE_SECTORS)
            if offensive_in_top3 >= 2:
                top_offensive = [t for t in ranking[:3] if t in OFFENSIVE_SECTORS]
                strength = offensive_in_top3 / 3
                signals.append(RotationSignal(
                    type="offensive",
                    strength=strength,
                    description=(
                        f"资金轮动至进攻板块：{', '.join(top_offensive)} 进入前3，"
                        f"市场风险偏好上升"
                    ),
                    sectors_involved=top_offensive,
                ))

        # --- 信号2: 经济复苏（金融/能源领涨）---
        recovery_sectors = {"XLF", "XLE", "XLI"}
        recovery_in_top5 = [t for t in ranking[:5] if t in recovery_sectors]
        if recovery_in_top5:
            # 计算这3个板块的平均排名
            avg_rank = sum(get_rank(t) for t in recovery_in_top5) / len(recovery_in_top5)
            if avg_rank <= 4:
                strength = max(0, 1 - (avg_rank - 1) / 3)
                signals.append(RotationSignal(
                    type="offensive",
                    strength=strength,
                    description=(
                        f"经济复苏信号：{', '.join(recovery_in_top5)} 表现强劲，"
                        f"平均排名 #{avg_rank:.1f}，反映经济扩张预期"
                    ),
                    sectors_involved=recovery_in_top5,
                ))

        # --- 信号3: 晚期牛市预警（科技+防御同涨）---
        tech_top = "XLK" in ranking[:3]
        util_top = "XLU" in ranking[:3]
        if tech_top and util_top:
            # 获取两者动量
            tech_perf = next((p for p in performances if p.ticker == "XLK"), None)
            util_perf = next((p for p in performances if p.ticker == "XLU"), None)
            if tech_perf and util_perf:
                if tech_perf.returns_20d > 0 and util_perf.returns_20d > 0:
                    strength = min(1.0, abs(tech_perf.returns_20d - util_perf.returns_20d) / 5)
                    signals.append(RotationSignal(
                        type="defensive",
                        strength=strength,
                        description=(
                            "晚期牛市预警：科技(XLK)和公用事业(XLU)同时走强，"
                            "历史上常见于牛市顶部前兆，需警惕回调"
                        ),
                        sectors_involved=["XLK", "XLU"],
                    ))

        # --- 信号4: 科技独涨（AI行情）---
        tech_perf = next((p for p in performances if p.ticker == "XLK"), None)
        if tech_perf and "XLK" == ranking[0]:
            other_offensive = [
                p for p in performances
                if p.ticker in OFFENSIVE_SECTORS and p.ticker != "XLK"
            ]
            if other_offensive:
                avg_other = sum(p.momentum_score for p in other_offensive) / len(other_offensive)
                gap = tech_perf.momentum_score - avg_other
                if gap > 10:
                    signals.append(RotationSignal(
                        type="neutral",
                        strength=min(1.0, gap / 20),
                        description=(
                            f"科技独涨格局：XLK动量得分{tech_perf.momentum_score:.1f}，"
                            f"领先其他进攻板块 {gap:.1f} 分，可能存在集中抱团"
                        ),
                        sectors_involved=["XLK"],
                    ))

        # 如果没有检测到明显信号，返回中性
        if not signals:
            signals.append(RotationSignal(
                type="neutral",
                strength=0.3,
                description="未检测到明显板块轮动信号，市场暂无明确方向",
                sectors_involved=[],
            ))

        return signals

    def _determine_market_phase(
        self,
        ranking: List[str],
        performances: List[SectorPerformance],
    ) -> str:
        """
        根据板块排名结构判断市场阶段

        Args:
            ranking: 板块排名
            performances: 各板块表现

        Returns:
            市场阶段字符串
        """
        def rank_of(ticker: str) -> int:
            try:
                return ranking.index(ticker) + 1
            except ValueError:
                return 99

        # 计算各类型板块的平均排名
        def avg_rank_of(tickers: set) -> float:
            filtered = [rank_of(t) for t in tickers if t in ranking]
            return sum(filtered) / len(filtered) if filtered else 99

        off_avg = avg_rank_of(OFFENSIVE_SECTORS)
        def_avg = avg_rank_of(DEFENSIVE_SECTORS)

        tech_rank = rank_of("XLK")
        fin_rank = rank_of("XLF")
        energy_rank = rank_of("XLE")
        util_rank = rank_of("XLU")

        # --- 判断规则 ---
        # 1. 早期牛市：进攻板块整体领先，金融/能源/工业表现好
        if (
            off_avg < def_avg
            and fin_rank <= 4
            and tech_rank <= 4
        ):
            return "early_bull"

        # 2. 恢复期：金融和能源领涨（经济重启）
        if (
            fin_rank <= 3
            and energy_rank <= 4
            and def_avg < 7
        ):
            return "recovery"

        # 3. 晚期牛市：科技领涨但公用事业也涨
        tech_perf = next((p for p in performances if p.ticker == "XLK"), None)
        util_perf = next((p for p in performances if p.ticker == "XLU"), None)
        if (
            tech_rank <= 2
            and util_rank <= 4
            and tech_perf
            and util_perf
            and tech_perf.returns_20d > 0
            and util_perf.returns_20d > 0
        ):
            return "late_bull"

        # 4. 熊市：防御板块长期领先
        if def_avg < off_avg - 1:
            return "bear"

        return "early_bull"  # 默认

    def _cluster_sectors(
        self,
        performances: List[SectorPerformance],
        period: str,
    ) -> List[List[str]]:
        """
        使用层次聚类将走势相似的板块归组

        Args:
            performances: 各板块表现
            period: 数据周期（用于命名）

        Returns:
            聚类结果，每组为 ticker 列表
        """
        try:
            # 构建特征矩阵：各板块的日收益率序列（归一化）
            features_dict = {}
            for ticker, info in self._sector_etfs.items():
                df = self._fetch_sector_data(ticker, period)
                if df.empty or len(df) < 20:
                    continue
                close = df["Close"]
                returns = close.pct_change().dropna()
                # 取最近60日（若不足则取全部）
                returns = returns.iloc[-min(60, len(returns)):]
                features_dict[ticker] = returns.values

            if len(features_dict) < 3:
                return []

            # 对齐长度
            min_len = min(len(v) for v in features_dict.values())
            matrix = np.array([v[-min_len:] for v in features_dict.values()])
            tickers = list(features_dict.keys())

            # 层次聚类（Ward方法，3个簇）
            dist_mat = pdist(matrix, metric="correlation")
            linkage_mat = linkage(dist_mat, method="ward")
            labels = fcluster(linkage_mat, t=3, criterion="maxclust")

            # 按簇分组
            clusters_dict = {}
            for ticker, label in zip(tickers, labels):
                clusters_dict.setdefault(label, []).append(ticker)

            # 按簇内板块数量降序
            sorted_clusters = sorted(
                clusters_dict.values(),
                key=len,
                reverse=True
            )

            logger.info(f"[SectorRotation] 聚类结果: {sorted_clusters}")
            return sorted_clusters

        except Exception as e:
            logger.warning(f"[SectorRotation] 聚类分析失败: {e}")
            return []

    def _generate_summary(
        self,
        performances: List[SectorPerformance],
        ranking: List[str],
        rotation_signals: List[RotationSignal],
        market_phase: str,
        clusters: List[List[str]],
    ) -> str:
        """生成分析摘要文本"""
        # 板块名称映射
        name_map = {t: self._sector_etfs.get(t, {}).get("name_zh", t) for t in ranking}

        # 领涨/领跌
        top3 = ranking[:3]
        bottom3 = ranking[-3:]
        top_names = "、".join([f"{t}({name_map[t]})" for t in top3])
        bottom_names = "、".join([f"{t}({name_map[t]})" for t in bottom3])

        # 市场阶段描述
        phase_desc = {
            "early_bull": "早期牛市",
            "late_bull": "晚期牛市",
            "bear": "熊市",
            "recovery": "经济恢复期",
        }
        phase_text = phase_desc.get(market_phase, "未知")

        # 聚类描述
        cluster_desc = ""
        if clusters:
            cluster_lines = []
            for i, grp in enumerate(clusters, 1):
                grp_names = "、".join([f"{t}({name_map.get(t, t)})" for t in grp])
                cluster_lines.append(f"  组{i}: {grp_names}")
            cluster_desc = "\n板块聚类（相似走势归组）:\n" + "\n".join(cluster_lines)

        # 信号描述
        signal_lines = []
        for sig in rotation_signals:
            type_cn = {"defensive": "防御", "offensive": "进攻", "neutral": "中性"}
            signal_lines.append(
                f"  [{type_cn.get(sig.type, sig.type)}] {sig.description} (强度:{sig.strength:.0%})"
            )

        summary = f"""=== 板块轮动分析摘要 ===
分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

【市场阶段】{phase_text}
【领涨板块】{top_names}
【领跌板块】{bottom_names}

【轮动信号】
{chr(10).join(signal_lines)}
{cluster_desc}

注：聚类分析基于最近60日收益率相关性，相同组内板块走势相似，可用于分散配置或因子剥离。
"""
        return summary.strip()

    def _classify_sector_phase(self, perf: SectorPerformance) -> str:
        """将单个板块分类为"强势"/"中性"/"弱势" """
        if perf.momentum_score >= 60:
            return "🟢强势"
        elif perf.momentum_score <= 40:
            return "🔴弱势"
        else:
            return "🟡中性"

    def _default_result(self) -> SectorRotationResult:
        """返回默认结果（数据不足时）"""
        return SectorRotationResult(
            sectors=[],
            ranking=[],
            rotation_signals=[],
            market_phase="early_bull",
            summary="数据不足，无法完成板块轮动分析",
            analysis_steps=["数据获取失败"],
            clusters=[],
        )


# ============================================================
# 主入口（独立测试）
# ============================================================

if __name__ == "__main__":
    import logging as std_logging

    std_logging.basicConfig(
        level=std_logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    print("=" * 60)
    print("板块轮动分析 - 独立测试")
    print("=" * 60)

    # 使用 yfinance fetcher
    from data.fetcher import YFinanceFetcher

    fetcher = YFinanceFetcher(cache_dir="data_cache/")
    analyzer = SectorRotationAnalyzer(data_fetcher=fetcher)

    # 执行分析（6个月数据）
    result = analyzer.analyze(period="6mo")

    # 打印摘要
    print(result.summary)
    print()

    # 打印详细排名表
    df = analyzer.get_sector_table(result)
    print("【板块强弱排名】")
    print(df.to_string(index=False))
    print()

    # 打印聚类
    if result.clusters:
        print("【板块聚类结果】")
        for i, grp in enumerate(result.clusters, 1):
            grp_names = [f"{t}({analyzer._sector_etfs.get(t, {}).get('name_zh', t)})" for t in grp]
            print(f"  簇{i}: {' | '.join(grp_names)}")
