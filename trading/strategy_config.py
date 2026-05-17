"""
trading/strategy_config.py — 策略配置管理器

功能：
    - 运行时策略参数调整
    - 多策略实例管理
    - 策略性能追踪
    - A/B 测试支持
    - 参数优化结果应用
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# 数据类定义
# ============================================================

@dataclass
class StrategyParameter:
    """策略参数定义"""
    name: str
    value: Any
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    description: str = ""
    param_type: str = "float"  # float/int/bool/choice
    choices: List[Any] = field(default_factory=list)

@dataclass
class StrategyInstance:
    """策略实例"""
    instance_id: str
    strategy_name: str
    ticker: str
    
    # 参数
    parameters: Dict[str, StrategyParameter] = field(default_factory=dict)
    
    # 状态
    is_active: bool = True
    weight: float = 1.0  # 在多策略组合中的权重
    
    # 绩效
    total_trades: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = None
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class StrategyPerformance:
    """策略绩效快照"""
    instance_id: str
    timestamp: datetime
    
    # 收益
    total_return_pct: float
    win_rate_pct: float
    profit_factor: float
    
    # 风险
    max_drawdown_pct: float
    volatility_pct: float
    sharpe_ratio: float
    
    # 交易
    total_trades: int
    avg_holding_days: float
    avg_trade_pnl: float
    
    # 对比
    vs_benchmark_pct: float  # 相对基准超额收益
    vs_buyhold_pct: float    # 相对买入持有

# ============================================================
# 策略配置管理器
# ============================================================

class StrategyConfigManager:
    """
    策略配置管理器：动态管理策略参数和实例。
    
    功能：
    1. 运行时参数调整（无需重启）
    2. 多策略并行运行
    3. 策略绩效追踪与对比
    4. A/B 测试
    5. 参数优化结果一键应用
    """
    
    def __init__(self, config_path: str = "config/strategies.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 策略实例存储
        self.instances: Dict[str, StrategyInstance] = {}
        
        # 绩效历史
        self.performance_history: Dict[str, List[StrategyPerformance]] = {}
        
        # 参数变更回调
        self.param_callbacks: Dict[str, List[Callable]] = {}
        
        # 加载已有配置
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        if not self.config_path.exists():
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for inst_data in data.get('instances', []):
                instance = StrategyInstance(
                    instance_id=inst_data['instance_id'],
                    strategy_name=inst_data['strategy_name'],
                    ticker=inst_data['ticker'],
                    is_active=inst_data.get('is_active', True),
                    weight=inst_data.get('weight', 1.0),
                    tags=inst_data.get('tags', []),
                    notes=inst_data.get('notes', ''),
                )
                
                # 加载参数
                for name, p_data in inst_data.get('parameters', {}).items():
                    instance.parameters[name] = StrategyParameter(
                        name=name,
                        value=p_data.get('value'),
                        default_value=p_data.get('default_value'),
                        min_value=p_data.get('min_value'),
                        max_value=p_data.get('max_value'),
                        step=p_data.get('step'),
                        description=p_data.get('description', ''),
                        param_type=p_data.get('param_type', 'float'),
                        choices=p_data.get('choices', []),
                    )
                
                self.instances[instance.instance_id] = instance
            
            logger.info(f"[StrategyConfig] 加载了 {len(self.instances)} 个策略实例")
        
        except Exception as e:
            logger.error(f"[StrategyConfig] 加载配置失败: {e}")
    
    def _save_config(self):
        """保存配置"""
        data = {
            'updated_at': datetime.now().isoformat(),
            'instances': [],
        }
        
        for instance in self.instances.values():
            inst_data = {
                'instance_id': instance.instance_id,
                'strategy_name': instance.strategy_name,
                'ticker': instance.ticker,
                'is_active': instance.is_active,
                'weight': instance.weight,
                'tags': instance.tags,
                'notes': instance.notes,
                'parameters': {
                    name: {
                        'value': p.value,
                        'default_value': p.default_value,
                        'min_value': p.min_value,
                        'max_value': p.max_value,
                        'step': p.step,
                        'description': p.description,
                        'param_type': p.param_type,
                        'choices': p.choices,
                    }
                    for name, p in instance.parameters.items()
                },
            }
            data['instances'].append(inst_data)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # ---- 实例管理 ----
    
    def create_instance(
        self,
        strategy_name: str,
        ticker: str,
        parameters: Dict[str, Any] = None,
        weight: float = 1.0,
        tags: List[str] = None,
    ) -> StrategyInstance:
        """
        创建策略实例。
        
        示例：
            instance = scm.create_instance(
                strategy_name="MA_Cross",
                ticker="AAPL",
                parameters={
                    "fast_period": 10,
                    "slow_period": 30,
                },
                weight=1.0,
                tags=["trend", "test"],
            )
        """
        instance_id = f"{strategy_name}_{ticker}_{datetime.now().strftime('%H%M%S')}"
        
        instance = StrategyInstance(
            instance_id=instance_id,
            strategy_name=strategy_name,
            ticker=ticker,
            weight=weight,
            tags=tags or [],
        )
        
        # 设置默认参数
        default_params = self._get_default_params(strategy_name)
        for name, p in default_params.items():
            instance.parameters[name] = StrategyParameter(
                name=name,
                value=parameters.get(name, p['default']) if parameters else p['default'],
                default_value=p['default'],
                min_value=p.get('min'),
                max_value=p.get('max'),
                step=p.get('step'),
                description=p.get('description', ''),
                param_type=p.get('type', 'float'),
                choices=p.get('choices', []),
            )
        
        self.instances[instance_id] = instance
        self._save_config()
        
        logger.info(f"[StrategyConfig] 创建实例 {instance_id}")
        
        return instance
    
    def _get_default_params(self, strategy_name: str) -> Dict:
        """获取策略默认参数"""
        defaults = {
            "MA_Cross": {
                "fast_period": {"default": 10, "min": 5, "max": 50, "type": "int", "description": "快线周期"},
                "slow_period": {"default": 30, "min": 20, "max": 200, "type": "int", "description": "慢线周期"},
            },
            "RSI": {
                "period": {"default": 14, "min": 7, "max": 30, "type": "int", "description": "RSI周期"},
                "overbought": {"default": 70, "min": 60, "max": 90, "type": "int", "description": "超买阈值"},
                "oversold": {"default": 30, "min": 10, "max": 40, "type": "int", "description": "超卖阈值"},
            },
            "MACD": {
                "fast": {"default": 12, "min": 8, "max": 20, "type": "int"},
                "slow": {"default": 26, "min": 20, "max": 50, "type": "int"},
                "signal": {"default": 9, "min": 5, "max": 15, "type": "int"},
            },
            "Bollinger": {
                "period": {"default": 20, "min": 10, "max": 50, "type": "int"},
                "std_dev": {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1, "type": "float"},
            },
            "Composite": {
                "ma_weight": {"default": 0.25, "min": 0, "max": 1, "step": 0.05, "type": "float"},
                "rsi_weight": {"default": 0.25, "min": 0, "max": 1, "step": 0.05, "type": "float"},
                "macd_weight": {"default": 0.25, "min": 0, "max": 1, "step": 0.05, "type": "float"},
                "bb_weight": {"default": 0.25, "min": 0, "max": 1, "step": 0.05, "type": "float"},
            },
        }
        return defaults.get(strategy_name, {})
    
    def delete_instance(self, instance_id: str) -> bool:
        """删除实例"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            self._save_config()
            logger.info(f"[StrategyConfig] 删除实例 {instance_id}")
            return True
        return False
    
    def get_instance(self, instance_id: str) -> Optional[StrategyInstance]:
        """获取实例"""
        return self.instances.get(instance_id)
    
    def get_active_instances(self, ticker: str = None) -> List[StrategyInstance]:
        """获取活跃实例"""
        instances = [i for i in self.instances.values() if i.is_active]
        if ticker:
            instances = [i for i in instances if i.ticker == ticker]
        return instances
    
    # ---- 参数调整 ----
    
    def update_parameter(self, instance_id: str, param_name: str, value: Any) -> bool:
        """
        更新参数（运行时调整）。
        
        示例：
            scm.update_parameter("MA_Cross_AAPL_123456", "fast_period", 15)
        """
        if instance_id not in self.instances:
            logger.warning(f"[StrategyConfig] 实例不存在: {instance_id}")
            return False
        
        instance = self.instances[instance_id]
        
        if param_name not in instance.parameters:
            logger.warning(f"[StrategyConfig] 参数不存在: {param_name}")
            return False
        
        param = instance.parameters[param_name]
        
        # 验证范围
        if param.min_value is not None and value < param.min_value:
            logger.warning(f"[StrategyConfig] 参数 {param_name} 低于最小值 {param.min_value}")
            return False
        if param.max_value is not None and value > param.max_value:
            logger.warning(f"[StrategyConfig] 参数 {param_name} 超过最大值 {param.max_value}")
            return False
        
        old_value = param.value
        param.value = value
        
        self._save_config()
        
        # 触发回调
        if instance_id in self.param_callbacks:
            for callback in self.param_callbacks[instance_id]:
                try:
                    callback(instance_id, param_name, old_value, value)
                except Exception as e:
                    logger.error(f"[StrategyConfig] 回调失败: {e}")
        
        logger.info(f"[StrategyConfig] 更新参数 {instance_id}.{param_name}: {old_value} -> {value}")
        
        return True
    
    def reset_parameters(self, instance_id: str):
        """重置参数为默认值"""
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        for param in instance.parameters.values():
            param.value = param.default_value
        
        self._save_config()
        logger.info(f"[StrategyConfig] 重置参数 {instance_id}")
        return True
    
    def register_param_callback(self, instance_id: str, callback: Callable):
        """注册参数变更回调"""
        if instance_id not in self.param_callbacks:
            self.param_callbacks[instance_id] = []
        self.param_callbacks[instance_id].append(callback)
    
    # ---- 绩效追踪 ----
    
    def record_performance(self, instance_id: str, performance: StrategyPerformance):
        """记录绩效"""
        if instance_id not in self.performance_history:
            self.performance_history[instance_id] = []
        
        self.performance_history[instance_id].append(performance)
        
        # 更新实例统计
        if instance_id in self.instances:
            inst = self.instances[instance_id]
            inst.sharpe_ratio = performance.sharpe_ratio
            inst.max_drawdown = performance.max_drawdown
            inst.total_trades = performance.total_trades
    
    def get_performance_history(self, instance_id: str) -> List[StrategyPerformance]:
        """获取绩效历史"""
        return self.performance_history.get(instance_id, [])
    
    def compare_instances(self, instance_ids: List[str]) -> Dict:
        """对比多个实例的绩效"""
        comparison = {}
        
        for iid in instance_ids:
            if iid not in self.instances:
                continue
            
            inst = self.instances[iid]
            history = self.performance_history.get(iid, [])
            
            if history:
                latest = history[-1]
                comparison[iid] = {
                    'strategy': inst.strategy_name,
                    'ticker': inst.ticker,
                    'is_active': inst.is_active,
                    'weight': inst.weight,
                    'total_return': latest.total_return_pct,
                    'sharpe': latest.sharpe_ratio,
                    'max_dd': latest.max_drawdown_pct,
                    'win_rate': latest.win_rate_pct,
                    'trades': latest.total_trades,
                }
        
        return comparison
    
    # ---- A/B 测试 ----
    
    def create_ab_test(
        self,
        strategy_name: str,
        ticker: str,
        param_variations: List[Dict[str, Any]],
        test_days: int = 30,
    ) -> List[StrategyInstance]:
        """
        创建 A/B 测试实例。
        
        示例：
            instances = scm.create_ab_test(
                "MA_Cross",
                "AAPL",
                [
                    {"fast_period": 10, "slow_period": 30},
                    {"fast_period": 15, "slow_period": 30},
                    {"fast_period": 10, "slow_period": 50},
                ],
            )
        """
        instances = []
        
        for i, params in enumerate(param_variations):
            instance = self.create_instance(
                strategy_name=strategy_name,
                ticker=ticker,
                parameters=params,
                weight=1.0 / len(param_variations),  # 均等权重
                tags=["ab_test", f"variant_{chr(65+i)}"],  # A, B, C...
            )
            instances.append(instance)
        
        logger.info(f"[StrategyConfig] 创建 A/B 测试: {len(instances)} 个变体")
        
        return instances
    
    def get_ab_test_results(self, tag: str = "ab_test") -> Dict:
        """获取 A/B 测试结果"""
        test_instances = [i for i in self.instances.values() if tag in i.tags]
        
        results = {}
        for inst in test_instances:
            history = self.performance_history.get(inst.instance_id, [])
            if history:
                latest = history[-1]
                variant = next((t for t in inst.tags if t.startswith("variant_")), "unknown")
                results[variant] = {
                    'instance_id': inst.instance_id,
                    'params': {name: p.value for name, p in inst.parameters.items()},
                    'total_return': latest.total_return_pct,
                    'sharpe': latest.sharpe_ratio,
                    'win_rate': latest.win_rate_pct,
                    'max_dd': latest.max_drawdown_pct,
                }
        
        return results
    
    def promote_best_variant(self, tag: str = "ab_test") -> Optional[str]:
        """
        提升最佳变体为正式策略。
        返回被提升的 instance_id。
        """
        results = self.get_ab_test_results(tag)
        
        if not results:
            return None
        
        # 按夏普比率排序
        best_variant = max(results.items(), key=lambda x: x[1]['sharpe'])
        variant_name, best_data = best_variant
        
        instance_id = best_data['instance_id']
        
        if instance_id in self.instances:
            inst = self.instances[instance_id]
            inst.tags = [t for t in inst.tags if not t.startswith("variant_")]
            inst.tags.append("promoted")
            inst.weight = 1.0  # 恢复全权重
            
            # 停用其他变体
            for other_inst in self.instances.values():
                if tag in other_inst.tags and other_inst.instance_id != instance_id:
                    other_inst.is_active = False
            
            self._save_config()
            logger.info(f"[StrategyConfig] 提升 {variant_name} 为正式策略")
        
        return instance_id
    
    # ---- 批量操作 ----
    
    def apply_optimization_result(self, instance_id: str, optimized_params: Dict[str, Any]):
        """
        应用参数优化结果。
        
        示例：
            scm.apply_optimization_result(
                "MA_Cross_AAPL_123456",
                {"fast_period": 12, "slow_period": 35}
            )
        """
        for param_name, value in optimized_params.items():
            self.update_parameter(instance_id, param_name, value)
        
        logger.info(f"[StrategyConfig] 应用优化结果到 {instance_id}")
    
    def clone_instance(self, instance_id: str, new_ticker: str = None) -> Optional[StrategyInstance]:
        """克隆实例"""
        if instance_id not in self.instances:
            return None
        
        original = self.instances[instance_id]
        
        new_instance = self.create_instance(
            strategy_name=original.strategy_name,
            ticker=new_ticker or original.ticker,
            parameters={name: p.value for name, p in original.parameters.items()},
            weight=original.weight,
            tags=original.tags + ["cloned"],
        )
        
        new_instance.notes = f"克隆自 {instance_id}"
        
        return new_instance
    
    def export_instance_config(self, instance_id: str) -> Optional[Dict]:
        """导出实例配置"""
        if instance_id not in self.instances:
            return None
        
        inst = self.instances[instance_id]
        return {
            'instance_id': inst.instance_id,
            'strategy_name': inst.strategy_name,
            'ticker': inst.ticker,
            'parameters': {name: p.value for name, p in inst.parameters.items()},
            'weight': inst.weight,
            'tags': inst.tags,
        }
    
    def import_instance_config(self, config: Dict) -> StrategyInstance:
        """导入实例配置"""
        return self.create_instance(
            strategy_name=config['strategy_name'],
            ticker=config['ticker'],
            parameters=config.get('parameters', {}),
            weight=config.get('weight', 1.0),
            tags=config.get('tags', []),
        )
