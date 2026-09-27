"""
定时采集 Agent 在线表现，折算成路由降权系数回写 Orchestrator。
阈值告警与突变检测不在这里：Langfuse Monitors 配阈值，运行期读数由 /monitor 的 summary() 暴露。
"""
import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Agent 在线表现 → 路由降权的闭环。

      _collect 读 Orchestrator.get_stats() → _routing_penalty 折成 0-0.9 →
      update_routing_penalties() 写回 stats.monitor_penalty →
      penalty 越过 ROUTING_DEMOTE_THRESHOLD 时 Orchestrator._apply_demotion
      在意图合法的候选里改选主 Agent

    读的是进程启动以来的累积口径：样本越多，单个请求对 success_rate / avg_ms 的撬动越小，
    penalty 因此移动缓慢、不会跟着一次抖动改判。要看时间窗内的退化，交给 Langfuse 的窗口聚合。
    """

    def __init__(
        self,
        orchestrator,
        tool_manager,
        interval_s: float = 10.0,
    ):
        self._orchestrator = orchestrator
        self._tool_manager = tool_manager
        self._interval     = interval_s
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
        这里的 stats 就是 Orchestrator 处理请求时实时更新的数，不需要额外埋点。
        """
        agent_stats = self._orchestrator.get_stats()
        self._orchestrator.update_routing_penalties({
            agent_key: self._routing_penalty(s["success_rate"], s["avg_ms"])
            for agent_key, s in agent_stats.items()
        })

    @staticmethod
    def _routing_penalty(success_rate: float, avg_ms: float) -> float:
        """把在线表现转成 0-0.9 的路由降权系数。"""
        penalty = 0.0
        if success_rate < 0.90:
            penalty += min(0.5, (0.90 - success_rate) * 2)
        if avg_ms > 3000:
            penalty += min(0.4, (avg_ms - 3000) / 10000)
        return min(penalty, 0.9)

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """运行期读数：Agent/工具统计，加每类 Agent 当前的降权与是否越线。"""
        return {
            "agent_stats": self._orchestrator.get_stats(),
            "tool_stats":  self._tool_manager.get_stats(),
            "routing":     self._orchestrator.routing_weights(),
        }
