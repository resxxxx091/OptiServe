"""
亮点：利用 Monitor 监控 Agent 在线表现

核心问题：如何利用 Monitor 监控 Agent 的在线表现？

本模块的答案：
  1. 实时采集 —— 每隔 N 秒从 Orchestrator 和 ToolManager 拉取最新统计
  2. 异常检测 —— Z-score 统计方法，自动发现指标突变
  3. 路由反馈 —— 将 Agent 成功率/延迟写回 Orchestrator，
     Orchestrator 的 _best_agent() 会据此动态调整路由权重
  4. 优化建议 —— 基于规则生成可操作的优化建议（不是空话）
  5. 告警 —— 超阈值时记进告警列表并打日志
"""
import asyncio
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class Severity(Enum):
    INFO     = "info"
    WARNING  = "warning"
    ERROR    = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    severity:    Severity
    metric:      str
    message:     str
    value:       float
    threshold:   float
    ts:          str = field(default_factory=lambda: datetime.now().isoformat())
    resolved:    bool = False


@dataclass
class Suggestion:
    """可操作的优化建议。"""
    title:       str
    detail:      str
    action:      str    # 具体操作步骤
    priority:    int    # 1-10


# ── 异常检测 ──────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    基于滑动窗口 Z-score 的异常检测。

    Z-score = |当前值 - 均值| / 标准差
    超过 sensitivity 倍标准差则判定为异常。
    """

    def __init__(self, window: int = 60, sensitivity: float = 2.5):
        self._window      = window
        self._sensitivity = sensitivity
        self._history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, metric: str, value: float) -> Optional[Dict[str, Any]]:
        """记录一个数据点，如果异常则返回异常信息，否则返回 None。"""
        buf = self._history[metric]
        buf.append(value)

        if len(buf) < self._window // 2:
            return None  # 数据不足，不检测

        mean  = statistics.mean(buf)
        stdev = statistics.stdev(buf) if len(buf) > 1 else 0.0
        if stdev == 0:
            return None

        z = abs(value - mean) / stdev
        if z > self._sensitivity:
            return {
                "metric":   metric,
                "value":    value,
                "mean":     mean,
                "z_score":  round(z, 2),
                "severity": "high" if z > self._sensitivity * 1.5 else "medium",
            }
        return None


# ── 性能监控器 ────────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    Agent 在线表现监控。

    与 Orchestrator 的联动：
      Monitor 采集 → 发现某 Agent 成功率下降 →
      Orchestrator.get_stats() 中该 Agent 的 routing_score 自动降低 →
      _best_agent() 路由时自动绕开该 Agent

    这就是"利用 Monitor 监控在线表现"的闭环。
    """

    # 兜底占比判定：样本太少不判（一次失败一次兜底就是 100%）
    FALLBACK_MIN_SAMPLES = 5
    FALLBACK_RATE_MIN    = 0.20

    # 告警阈值
    THRESHOLDS = {
        "agent_success_rate":  (0.90, Severity.ERROR,   "less_than"),
        "tool_success_rate":   (0.95, Severity.WARNING,  "less_than"),
        "tool_fallback_rate":  (FALLBACK_RATE_MIN, Severity.WARNING, "greater_than"),
        "agent_avg_ms":        (3000, Severity.WARNING,  "greater_than"),
        "tool_avg_ms":         (5000, Severity.ERROR,    "greater_than"),
    }

    def __init__(
        self,
        orchestrator,
        tool_manager,
        interval_s:       float = 10.0,
        alert_max:        int   = 200,
        suggestion_max:   int   = 50,
    ):
        self._orchestrator = orchestrator
        self._tool_manager = tool_manager
        self._interval     = interval_s
        self._detector     = AnomalyDetector()

        # 有界：采集每 interval_s 跑一轮，长驻进程不能用无上限 list 存历史
        self._alerts:      Deque[Alert]      = deque(maxlen=alert_max)
        self._suggestions: Deque[Suggestion] = deque(maxlen=suggestion_max)
        self._active       = False
        self._task:        Optional[asyncio.Task] = None

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._task   = asyncio.create_task(self._loop())
        logger.info(f"Monitor 已启动，采集间隔 {self._interval}s")

    async def stop(self) -> None:
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── 采集循环 ──────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._active:
            try:
                await self._collect()
            except Exception as ex:
                logger.error(f"Monitor 采集异常: {ex}")
            await asyncio.sleep(self._interval)

    async def _collect(self) -> None:
        """
        采集 Agent 和工具的实时统计，检测异常，生成建议。

        关键：这里读取的 stats 就是 Orchestrator/ToolManager 在处理请求时
        实时更新的数据，Monitor 不需要额外埋点。
        """
        agent_stats = self._orchestrator.get_stats()
        tool_stats  = self._tool_manager.get_stats()
        routing_penalties: Dict[str, float] = {}

        # ── Agent 指标 ────────────────────────────────────────────────────────
        for agent_key, s in agent_stats.items():
            sr  = s["success_rate"]
            ms  = s["avg_ms"]

            # 异常检测
            for metric, value in [("agent_success_rate", sr), ("agent_avg_ms", ms)]:
                anomaly = self._detector.record(f"{metric}:{agent_key}", value)
                if anomaly:
                    logger.warning(f"异常检测 [{agent_key}] {metric}={value:.3f} z={anomaly['z_score']}")

            # 阈值告警
            self._check_threshold("agent_success_rate", sr, agent_key)
            self._check_threshold("agent_avg_ms", ms, agent_key)

            routing_penalties[agent_key] = self._routing_penalty(sr, ms)

        # ── 工具指标 ──────────────────────────────────────────────────────────
        for tool_name, s in tool_stats.items():
            sr = s["success_rate"]
            ms = s["avg_latency_ms"]
            cf = s["consecutive_fails"]
            fr = s.get("fallback_rate", 0.0)
            total = s.get("total", 0)

            self._check_threshold("tool_success_rate", sr, tool_name)
            self._check_threshold("tool_avg_ms", ms, tool_name)

            # 连续失败 → 生成具体建议
            if cf >= 3:
                self._add_suggestion(Suggestion(
                    title=f"工具 {tool_name} 连续失败",
                    detail=f"连续失败 {cf} 次，成功率 {sr:.1%}，平均延迟 {ms:.0f}ms，熔断状态: {s['circuit_state']}",
                    action="1. 检查工具依赖服务是否正常\n2. 查看错误日志\n3. 考虑增加超时时间或降级策略",
                    priority=9,
                ))

            # 成功率被兜底撑到 100% 时，只有 fallback_rate 看得见真实依赖故障
            if total >= self.FALLBACK_MIN_SAMPLES:
                self._check_threshold("tool_fallback_rate", fr, tool_name)
                if fr >= self.FALLBACK_RATE_MIN:
                    self._add_suggestion(Suggestion(
                        title=f"工具 {tool_name} 降级占比偏高",
                        detail=f"{total} 次调用里 {s.get('fallback', 0)} 次由 fallback 返回（{fr:.1%}），"
                               f"成功率 {sr:.1%} 是被兜底撑起来的",
                        action="1. 查 ToolResult.error 里的原始失败原因\n"
                               "2. 恢复真实 handler 的依赖（向量层/下游服务）\n"
                               "3. 兜底文案长期占高位会污染答案质量，需要设占比上限或告警升级",
                        priority=8,
                    ))

        # ── 路由优化建议 ──────────────────────────────────────────────────────
        updater = getattr(self._orchestrator, "update_routing_penalties", None)
        if updater:
            updater(routing_penalties)
        self._generate_routing_suggestions(agent_stats)

    @staticmethod
    def _routing_penalty(success_rate: float, avg_ms: float) -> float:
        """把在线表现转成 0-0.9 的路由降权系数。"""
        penalty = 0.0
        if success_rate < 0.90:
            penalty += min(0.5, (0.90 - success_rate) * 2)
        if avg_ms > 3000:
            penalty += min(0.4, (avg_ms - 3000) / 10000)
        return min(penalty, 0.9)

    def _check_threshold(self, metric: str, value: float, label: str) -> None:
        if metric not in self.THRESHOLDS:
            return
        threshold, severity, operator = self.THRESHOLDS[metric]
        triggered = (operator == "less_than" and value < threshold) or \
                    (operator == "greater_than" and value > threshold)
        # Alert.metric 是 "指标:对象" 复合键：同一条越界指标原地更新而不堆新条目，
        # 不同 agent / 工具各自保留一条，恢复时才让位给 resolved。
        key      = f"{metric}:{label}"
        existing = next((a for a in self._alerts if a.metric == key and not a.resolved), None)
        message  = f"{label} 的 {metric} = {value:.3f}，阈值 {threshold}"

        if not triggered:
            if existing is not None:
                existing.resolved = True
            return

        if existing is not None:
            existing.value   = value
            existing.message = message
            existing.ts      = datetime.now().isoformat()
            return

        alert = Alert(
            severity=severity,
            metric=key,
            message=message,
            value=value,
            threshold=threshold,
        )
        self._alerts.append(alert)
        # 只在告警首次出现时打日志：持续越界时每 10s 重复一条没有新信息
        logger.warning(f"[{severity.value.upper()}] {alert.message}")

    def _generate_routing_suggestions(self, agent_stats: Dict[str, Any]) -> None:
        """
        基于 Agent 在线表现生成路由优化建议。
        这是 Monitor → Orchestrator 反馈闭环的体现。
        """
        for agent_key, s in agent_stats.items():
            if s["success_rate"] < 0.85 and s["total"] > 10:
                self._add_suggestion(Suggestion(
                    title=f"Agent {agent_key} 成功率偏低",
                    detail=f"成功率 {s['success_rate']:.1%}，路由评分 {s['routing_score']:.3f}",
                    action=(
                        "Orchestrator 已自动降低该 Agent 的路由权重：同类多实例时 _best_agent() 换实例，"
                        "penalty 越过 ROUTING_DEMOTE_THRESHOLD 时路由直接改选意图合法的备选 Agent。\n"
                        "建议：1. 检查 system_prompt 是否需要优化\n"
                        "      2. 检查该类型问题的复杂度是否超出 Agent 能力\n"
                        "      3. 考虑增加同类型 Agent 实例"
                    ),
                    priority=8,
                ))

    def _add_suggestion(self, s: Suggestion) -> None:
        # 去重：相同 title 不重复添加
        if not any(x.title == s.title for x in self._suggestions):
            self._suggestions.append(s)
            logger.info(f"优化建议 [P{s.priority}]: {s.title}")

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """返回当前监控摘要，供 API 层暴露。"""
        weights = getattr(self._orchestrator, "routing_weights", None)
        return {
            "agent_stats":   self._orchestrator.get_stats(),
            "tool_stats":    self._tool_manager.get_stats(),
            "routing":       weights() if weights else {},
            "active_alerts": [asdict(a) for a in self._alerts if not a.resolved][-10:],
            "suggestions":   [
                {"title": s.title, "action": s.action, "priority": s.priority}
                for s in sorted(self._suggestions, key=lambda x: -x.priority)[:5]
            ],
        }
