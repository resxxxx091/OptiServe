"""工具调用的可靠性外壳（所有工具共用）。

  1. 参数校验（JSON Schema 的 required 与顶层类型）
  2. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用
  3. 超时控制（asyncio.wait_for）—— 单次 handler 执行按工具配 timeout_s
  4. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩；
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。
"""
import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from core.degradation import Dep, degrade

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    error:          Optional[str] = None
    reranked:       bool = False   # data 是否经过 Reranker 精排
    degraded:       bool = False   # data 来自 fallback，不是真实 handler 的输出
    stages:         Dict[str, int] = field(default_factory=dict)  # 检索链路各级的条数


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0 # 每次进入调用的请求（含熔断拒绝，不含"工具不存在"）
    success:            int = 0 # 调用方拿到可用结果（真实成功 + 降级救回）
    fallback:           int = 0 # 调用方拿到兜底结果（降级救回）
    total_latency_ms:   float = 0.0 # handler 执行的总延迟（ms）
    latency_samples:    int = 0 # 真正跑到 handler 的次数，延迟均值只按它算
    consecutive_fails:  int = 0 # 连续失败次数，熔断器用

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def fallback_rate(self) -> float:
        return self.fallback / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.latency_samples if self.latency_samples else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED
                                         → OPEN
    规则：
    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  float = 0.0
        self.probe_at:   Optional[float] = None   # 在飞的探测请求起跑时刻

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # 还没到恢复时间就一直拒；到了才转 HALF_OPEN 并开始探测窗口
            if time.monotonic() - self.opened_at < self.recovery_s:
                return False
            self.state = CircuitState.HALF_OPEN
            self.probe_at = None
        # HALF_OPEN：窗口内只放一个探测请求。probe_at 同时充当租约起点——
        # 探测被外层总超时取消时不会回调 record_*，靠租约到期自愈，否则会永久卡在半开。
        now = time.monotonic()
        if self.probe_at is not None and now - self.probe_at < self.recovery_s:
            return False
        self.probe_at = now
        return True

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED
        self.probe_at = None

    def record_failure(self) -> None:
        self.probe_at = None
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    fallback:    Optional[Callable] = None    # sync/async (params, error) -> Any

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── 工具注册表 ────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    工具调用外壳：注册表 + 校验 / 缓存 / 超时 / 熔断 / 降级 / 统计。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, Tuple[Any, float]] = {}   # key → (result, expire_at)

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 缓存写入
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"工具不存在: {name}")

        tool.stats.total += 1

        # 缓存命中
        if tool.cache_ttl > 0:
            cached = self._get_cache(name, params)
            if cached is not None:
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached,
                )

        # 熔断检查
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, error)

        t0 = time.monotonic()
        try:
            # 参数校验（根据 JSON Schema 的 required 和 properties.type）
            self._validate_params(tool, params)

            data = await asyncio.wait_for(self._run_handler(tool, params), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.stats.latency_samples += 1
            tool.breaker.record_success()

            # 缓存的是双路召回的原始结果；RRF 粗排、精排与断崖截断在检索图里做，
            # 在缓存之外，因此缓存命中时 reranked 必然为 False。
            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl)

            return ToolResult(success=True, data=data)

        except asyncio.TimeoutError:
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, "执行超时")

        except Exception as ex:
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        error: str,
    ) -> ToolResult:
        """
        工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。

        降级算"调用方拿到了结果"（计入 success），但同时记一次 fallback 并打上
        degraded 标记——否则依赖全挂时成功率仍是 100%，监控看不出问题。
        """
        if tool.fallback is None:
            return ToolResult(success=False, data=None, error=error)
        try:
            data = tool.fallback(params, error)
            if asyncio.iscoroutine(data):
                data = await data
            tool.stats.success += 1
            tool.stats.fallback += 1
            degrade(Dep.TOOL, "fallback", f"{tool.name} 由降级兜底返回：{error}")
            return ToolResult(
                success=True,
                data=data,
                error=error,
                degraded=True,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, error=f"{error}; fallback失败: {ex}")

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
    ) -> Any:
        """
        执行工具 handler。

        优先支持 async handler；如果历史工具仍是同步函数，则放入线程池执行，
        避免阻塞事件循环。
        """
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params)
        result = await asyncio.to_thread(tool.handler, params)
        # 如果 handler 返回的是 awaitable 对象（例如 asyncio.Future），则继续 await
        if inspect.isawaitable(result):
            return await result
        return result

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict) -> str:
        payload = json.dumps(params, sort_keys=True)
        return f"{name}:{hashlib.md5(payload.encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict) -> Optional[Any]:
        key = self._cache_key(name, params)
        if key in self._cache:
            data, expire_at = self._cache[key]
            if time.monotonic() < expire_at:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, name: str, params: Dict, data: Any, ttl: float) -> None:
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params)] = (data, time.monotonic() + ttl)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                        )

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "fallback": t.stats.fallback,
                "fallback_rate": round(t.stats.fallback_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "latency_samples": t.stats.latency_samples,
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
