# ============================================================
# notification/notifier.py — 多渠道通知推送模块
# 支持：邮件 / Telegram / 企业微信
# ============================================================
import os
import logging
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# 数据类
# ============================================================

@dataclass
class NotificationMessage:
    """通知消息"""
    title: str
    content: str
    level: str = "info"  # info | warning | error | success
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        level_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }
        emoji = level_emoji.get(self.level, "📢")
        return f"""
{emoji} **{self.title}**

{self.content}

---
📅 {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""

# ============================================================
# 邮件通知
# ============================================================

class EmailNotifier:
    """邮件通知"""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        from_addr: str,
        to_addrs: List[str]
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
    
    def send(self, message: NotificationMessage) -> bool:
        """发送邮件"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[量化交易] {message.title}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            
            # 纯文本
            text_part = MIMEText(message.content, "plain", "utf-8")
            msg.attach(text_part)
            
            # HTML
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #333;">{message.title}</h2>
                <div style="white-space: pre-wrap; line-height: 1.6;">
                    {message.content}
                </div>
                <hr style="border: 1px solid #eee;">
                <p style="color: #999; font-size: 12px;">
                    📅 {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </body>
            </html>
            """
            html_part = MIMEText(html_content, "html", "utf-8")
            msg.attach(html_part)
            
            # 发送
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            
            logger.info(f"邮件发送成功: {message.title}")
            return True
            
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return False


# ============================================================
# Telegram 通知
# ============================================================

class TelegramNotifier:
    """Telegram 通知"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send(self, message: NotificationMessage) -> bool:
        """发送 Telegram 消息"""
        try:
            text = message.to_markdown()
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            resp = requests.post(url, data=data, timeout=10)
            result = resp.json()
            
            if result.get("ok"):
                logger.info(f"Telegram 发送成功: {message.title}")
                return True
            else:
                logger.error(f"Telegram 发送失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram 发送失败: {e}")
            return False


# ============================================================
# 企业微信通知
# ============================================================

class WeChatNotifier:
    """企业微信机器人通知"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send(self, message: NotificationMessage) -> bool:
        """发送企业微信消息"""
        try:
            data = {
                "msgtype": "markdown",
                "markdown": {
                    "content": message.to_markdown()
                }
            }
            
            resp = requests.post(
                self.webhook_url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            result = resp.json()
            
            if result.get("errcode") == 0:
                logger.info(f"企业微信发送成功: {message.title}")
                return True
            else:
                logger.error(f"企业微信发送失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"企业微信发送失败: {e}")
            return False


# ============================================================
# 通知管理器
# ============================================================

class NotificationManager:
    """
    统一通知管理器
    
    支持多渠道并发通知，统一配置管理。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化通知管理器
        
        Args:
            config: 配置字典，从 config.yaml 加载
        """
        self.config = config.get("notification", {})
        self.enabled = self.config.get("enable", False)
        
        self.notifiers = []
        self.triggers = self.config.get("triggers", {})
        
        self._init_notifiers()
    
    def _init_notifiers(self):
        """初始化所有启用的通知渠道"""
        if not self.enabled:
            return
        
        # 邮件
        email_cfg = self.config.get("email", {})
        if email_cfg.get("enable"):
            self.notifiers.append(EmailNotifier(
                smtp_host=email_cfg.get("smtp_host", ""),
                smtp_port=email_cfg.get("smtp_port", 587),
                smtp_user=email_cfg.get("smtp_user", ""),
                smtp_password=email_cfg.get("smtp_password", ""),
                from_addr=email_cfg.get("from_addr", ""),
                to_addrs=email_cfg.get("to_addrs", [])
            ))
            logger.info("邮件通知已启用")
        
        # Telegram
        tg_cfg = self.config.get("telegram", {})
        if tg_cfg.get("enable"):
            self.notifiers.append(TelegramNotifier(
                bot_token=tg_cfg.get("bot_token", ""),
                chat_id=tg_cfg.get("chat_id", "")
            ))
            logger.info("Telegram 通知已启用")
        
        # 企业微信
        wechat_cfg = self.config.get("wechat", {})
        if wechat_cfg.get("enable"):
            self.notifiers.append(WeChatNotifier(
                webhook_url=wechat_cfg.get("webhook_url", "")
            ))
            logger.info("企业微信通知已启用")
    
    def notify(
        self,
        title: str,
        content: str,
        level: str = "info",
        trigger_type: Optional[str] = None
    ) -> bool:
        """
        发送通知
        
        Args:
            title: 标题
            content: 内容
            level: 级别（info/warning/error/success）
            trigger_type: 触发类型（trade/signal/error/daily_report/risk_alert）
        
        Returns:
            是否发送成功（至少一个渠道成功）
        """
        if not self.enabled or not self.notifiers:
            return False
        
        # 检查触发条件
        if trigger_type and not self.triggers.get(f"on_{trigger_type}", True):
            return False
        
        message = NotificationMessage(
            title=title,
            content=content,
            level=level
        )
        
        success = False
        for notifier in self.notifiers:
            try:
                if notifier.send(message):
                    success = True
            except Exception as e:
                logger.error(f"通知发送异常: {e}")
        
        return success
    
    def notify_trade(
        self,
        action: str,
        symbol: str,
        price: float,
        quantity: int,
        pnl: Optional[float] = None
    ):
        """交易通知"""
        content = f"""
📊 **交易执行**

- **操作**: {action}
- **标的**: {symbol}
- **价格**: ${price:.2f}
- **数量**: {quantity}
"""
        if pnl is not None:
            emoji = "📈" if pnl >= 0 else "📉"
            content += f"- **盈亏**: {emoji} ${pnl:+.2f}"
        
        self.notify(
            title=f"交易通知: {action} {symbol}",
            content=content,
            level="success" if action == "BUY" else ("warning" if pnl and pnl < 0 else "info"),
            trigger_type="trade"
        )
    
    def notify_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        confidence: float,
        reasons: List[str]
    ):
        """信号通知"""
        content = f"""
📡 **交易信号**

- **标的**: {symbol}
- **信号**: {signal}
- **价格**: ${price:.2f}
- **置信度**: {confidence:.1%}

**原因**:
{chr(10).join(f'• {r}' for r in reasons)}
"""
        self.notify(
            title=f"信号: {signal} {symbol}",
            content=content,
            level="info",
            trigger_type="signal"
        )
    
    def notify_error(self, error_type: str, error_msg: str, details: str = ""):
        """错误通知"""
        content = f"""
❌ **系统错误**

- **类型**: {error_type}
- **信息**: {error_msg}
"""
        if details:
            content += f"\n**详情**:\n```\n{details}\n```"
        
        self.notify(
            title=f"错误: {error_type}",
            content=content,
            level="error",
            trigger_type="error"
        )
    
    def notify_daily_report(self, report: str):
        """每日报告通知"""
        self.notify(
            title="📊 每日量化报告",
            content=report,
            level="info",
            trigger_type="daily_report"
        )
    
    def notify_risk_alert(self, alert_type: str, message: str):
        """风险预警通知"""
        self.notify(
            title=f"⚠️ 风险预警: {alert_type}",
            content=message,
            level="warning",
            trigger_type="risk_alert"
        )


# ============================================================
# 工具函数
# ============================================================

def create_notifier_from_config(config_path: str = "config.yaml") -> NotificationManager:
    """
    从配置文件创建通知管理器
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        NotificationManager 实例
    """
    import yaml
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return NotificationManager(config)
