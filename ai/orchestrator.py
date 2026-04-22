# ============================================================
# ai/orchestrator.py — AI 量化分析师主控制器
# 协调市场分析、选股、策略推荐、报告生成
# ============================================================
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 数据类
# ============================================================

@dataclass
class DailyReport:
    """每日报告"""
    date: str
    market_state: Dict
    top_opportunities: List[Dict]
    strategy_recommendations: List[Dict]
    portfolio_summary: Dict
    risk_alerts: List[str]
    ai_summary: str


# ============================================================
# AI 量化分析师
# ============================================================

class AIQuantAnalyst:
    """
    AI 量化分析师
    
    整合市场分析、选股、策略推荐、报告生成。
    """
    
    def __init__(
        self,
        data_fetcher=None,
        notification_manager=None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            data_fetcher: 数据获取器
            notification_manager: 通知管理器
            config: 配置字典
        """
        from .market_analyzer import MarketAnalyzer
        from .stock_screener import StockScreener
        
        self.fetcher = data_fetcher
        self.notifier = notification_manager
        self.config = config or {}
        
        # 初始化子模块
        self.market_analyzer = MarketAnalyzer(self.fetcher)
        self.screener = StockScreener(self.fetcher)
        
        # AI 配置
        self.ai_config = self.config.get("ai", {})
        self.llm_config = self.ai_config.get("llm", {})
        self.enable_llm = self.llm_config.get("enable", False)
    
    def run_daily_analysis(
        self,
        watchlist: Optional[List[str]] = None,
        generate_report: bool = True
    ) -> DailyReport:
        """
        执行每日分析
        
        Args:
            watchlist: 观察列表
            generate_report: 是否生成报告
        
        Returns:
            DailyReport 实例
        """
        logger.info("开始每日分析...")
        
        # 1. 市场分析
        market_state = self.market_analyzer.get_market_summary()
        
        # 2. 选股
        opportunities = []
        if watchlist:
            opportunities = self.screener.screen(
                tickers=watchlist,
                top_n=10
            )
        else:
            opportunities = self.screener.screen(
                pool_name="popular",
                top_n=10
            )
        
        # 3. 策略推荐
        recommendations = self._recommend_strategies(
            market_state,
            opportunities
        )
        
        # 4. 风险预警
        risk_alerts = self._check_risk_alerts(market_state)
        
        # 5. 生成 AI 摘要
        ai_summary = ""
        if self.enable_llm:
            ai_summary = self._generate_ai_summary(
                market_state,
                opportunities,
                recommendations
            )
        
        # 6. 生成报告
        report = DailyReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            market_state=market_state,
            top_opportunities=[
                {
                    "ticker": o.ticker,
                    "score": o.score,
                    "signal": o.signal,
                    "price": o.price,
                    "reason": o.reason
                }
                for o in opportunities[:5]
            ],
            strategy_recommendations=recommendations,
            portfolio_summary={},  # TODO: 从持仓模块获取
            risk_alerts=risk_alerts,
            ai_summary=ai_summary
        )
        
        # 7. 发送通知
        if self.notifier and generate_report:
            self.notifier.notify_daily_report(
                self._format_report(report)
            )
        
        logger.info("每日分析完成")
        return report
    
    def _recommend_strategies(
        self,
        market_state: Dict,
        opportunities: List
    ) -> List[Dict]:
        """推荐策略"""
        recommendations = []
        
        recommended_strategies = market_state.get("recommended_strategies", [])
        
        for opp in opportunities[:5]:
            # 根据个股特征调整策略
            ticker_strategies = recommended_strategies.copy()
            
            # RSI 超卖 → 加 RSI 策略
            if opp.rsi < 35:
                if "RSI" not in ticker_strategies:
                    ticker_strategies.append("RSI")
            
            # 高波动 → 加 Bollinger
            if opp.atr_pct > 3:
                if "Bollinger" not in ticker_strategies:
                    ticker_strategies.append("Bollinger")
            
            recommendations.append({
                "ticker": opp.ticker,
                "strategies": ticker_strategies[:3],
                "reason": f"基于市场状态({market_state['trend']})和个股特征(RSI={opp.rsi:.0f})"
            })
        
        return recommendations
    
    def _check_risk_alerts(self, market_state: Dict) -> List[str]:
        """检查风险预警"""
        alerts = []
        
        risk_level = market_state.get("risk_level", 5)
        
        if risk_level >= 8:
            alerts.append(f"⚠️ 高风险警告: 当前风险等级 {risk_level}/10")
        
        if market_state.get("volatility") == "high":
            alerts.append("⚠️ 波动率偏高，建议降低仓位")
        
        if market_state.get("trend") == "bear":
            alerts.append("⚠️ 市场处于下跌趋势，谨慎操作")
        
        return alerts
    
    def _generate_ai_summary(
        self,
        market_state: Dict,
        opportunities: List,
        recommendations: List
    ) -> str:
        """生成 AI 摘要（使用 LLM）"""
        if not self.enable_llm:
            return ""
        
        try:
            import openai
            
            api_key = self.llm_config.get("api_key", "")
            api_base = self.llm_config.get("api_base", "")
            model = self.llm_config.get("model", "gpt-4o-mini")
            
            if not api_key:
                return ""
            
            client = openai.OpenAI(
                api_key=api_key,
                base_url=api_base if api_base else None
            )
            
            # 构建提示
            prompt = f"""
你是一个专业的量化交易分析师。请根据以下信息生成一份简明的每日分析摘要。

## 市场状态
- 趋势: {market_state.get('trend', 'unknown')}
- 波动率: {market_state.get('volatility', 'unknown')}
- 风险等级: {market_state.get('risk_level', 5)}/10

## 热门机会
{json.dumps([{"ticker": o.ticker, "score": o.score, "signal": o.signal, "reason": o.reason} for o in opportunities[:5]], indent=2)}

## 策略推荐
{json.dumps(recommendations, indent=2)}

请生成：
1. 市场研判（2-3句话）
2. 今日操作建议（3-5条）
3. 风险提示（如有）

用简洁专业的语言回答。
"""
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_config.get("temperature", 0.7),
                max_tokens=self.llm_config.get("max_tokens", 500)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI 摘要生成失败: {e}")
            return ""
    
    def _format_report(self, report: DailyReport) -> str:
        """格式化报告"""
        text = f"""
# 📊 量化日报 - {report.date}

## 市场状态
- 趋势: {report.market_state.get('trend', 'unknown')}
- 波动率: {report.market_state.get('volatility', 'unknown')}
- 风险等级: {report.market_state.get('risk_level', 5)}/10

## 热门机会
"""
        for opp in report.top_opportunities:
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(opp["signal"], "⚪")
            text += f"- {opp['ticker']}: {signal_emoji} {opp['signal']} (得分 {opp['score']:.0f})\n"
        
        if report.ai_summary:
            text += f"\n## AI 分析\n{report.ai_summary}\n"
        
        if report.risk_alerts:
            text += f"\n## 风险预警\n"
            for alert in report.risk_alerts:
                text += f"- {alert}\n"
        
        return text
    
    def save_report(self, report: DailyReport, output_dir: str = "reports"):
        """保存报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"daily_report_{report.date}.json"
        filepath = output_path / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "date": report.date,
                "market_state": report.market_state,
                "top_opportunities": report.top_opportunities,
                "strategy_recommendations": report.strategy_recommendations,
                "risk_alerts": report.risk_alerts,
                "ai_summary": report.ai_summary
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"报告已保存: {filepath}")
    
    def quick_analysis(self, ticker: str) -> Dict:
        """快速分析单只股票（用于 WebUI）"""
        # 市场背景
        market = self.market_analyzer.get_market_summary()
        
        # 个股筛选
        candidates = self.screener.screen(tickers=[ticker])
        
        if not candidates:
            return {
                "ticker": ticker,
                "error": "无法获取数据"
            }
        
        candidate = candidates[0]
        
        return {
            "ticker": ticker,
            "price": candidate.price,
            "change_pct": candidate.change_pct,
            "score": candidate.score,
            "signal": candidate.signal,
            "factors": candidate.factors,
            "reason": candidate.reason,
            "market_context": {
                "trend": market["trend"],
                "volatility": market["volatility"],
                "risk_level": market["risk_level"]
            },
            "recommended_strategies": market["recommended_strategies"],
            "timestamp": datetime.now().isoformat()
        }
