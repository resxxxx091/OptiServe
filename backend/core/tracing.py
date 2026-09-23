"""
一次请求的链路：节点开始即建 Langfuse observation、结束即 .end() 入队，SDK 后台批量上报。
与 core/degradation.py 同一套 ContextVar 机制；不自己搭 span 树，父子链交给 OTel context。
"""
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

_lf: Any = None    # init_tracing() 装好；None 表示本次运行什么都不上报


# 我们的 span 名前缀 → Langfuse 的 observation 类型，让 UI 按语义分组。
# 取值必须是 langfuse._client.constants 里 ObservationTypeSpanLike / GenerationLike 的成员。
_AS_TYPE = (
    ("tool:", "tool"),
    ("agent:", "agent"),
    ("rag.", "retriever"),
)


def _as_type(name: str) -> str:
    for prefix, kind in _AS_TYPE:
        if name.startswith(prefix):
            return kind
    return "span"


# ── 生命周期 ──────────────────────────────────────────────────────────────────

def init_tracing() -> Any:
    """配了真密钥才建客户端；任何一步不成都返回 None，服务照常跑。"""
    global _lf
    public = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    if not public or not secret:
        logger.info("未配置 Langfuse 密钥，本次运行不导出任何链路记录")
        return None
    try:
        from langfuse import Langfuse   # 密钥齐全才 import，没装 SDK 也不影响启动

        # 不传参：host / 密钥 / 采样率由 SDK 自己按 LANGFUSE_* 环境变量解析
        _lf = Langfuse()
    except Exception as ex:
        logger.warning(f"Langfuse 客户端初始化失败，本次启动不上报: {ex}")
        return None
    return _lf


def shutdown_tracing() -> None:
    """应用退出时把批量队列里剩下的节点冲出去并停掉后台线程。"""
    if _lf is None:
        return
    try:
        _lf.shutdown()               # SDK 的 shutdown 自带 flush
    except Exception as ex:
        logger.warning(f"Langfuse 关闭失败: {ex}")


# ── 一次请求的把手 ────────────────────────────────────────────────────────────

class _TraceHandle:
    """一条进行中 trace：只装 trace 级 meta 和根 observation，不装孩子也不装时间。"""

    def __init__(self, root: Any):
        self.root = root
        self.meta: Dict[str, Any] = {}


class _Node:
    """埋点方看到的节点把手。attrs 累加到节点退出时才推给 SDK ——
    update(metadata=) 是整值覆盖，而 .end() 之后再也写不进去。"""
    __slots__ = ("attrs",)

    def __init__(self, attrs: Dict[str, Any]):
        self.attrs: Dict[str, Any] = dict(attrs)


_trace: ContextVar[Optional[_TraceHandle]] = ContextVar("optiserve_trace", default=None)


# ── 开 trace ──────────────────────────────────────────────────────────────────

@contextmanager
def start_trace(trace_id: str, root_name: str = "request") -> Generator[Any, None, None]:
    """新开一次 trace：根节点活到请求收尾，退出时先写 trace 级字段再结束它。"""
    if _lf is None:
        # 仍然占住 ContextVar：trace_scope 靠它按身份复用同一条 trace
        handle = _TraceHandle(None)
        token = _trace.set(handle)
        try:
            yield handle
        finally:
            _trace.reset(token)
        return

    # trace_id 由 request_id 派生：同一个 request_id 重复开仍落在同一条 trace 上，
    # 响应体里的 ID 因此就是 Langfuse 的 trace ID
    seeded = _lf.create_trace_id(seed=f"optiserve:{trace_id}")
    with _lf.start_as_current_observation(
        trace_context={"trace_id": seeded},
        name=root_name,
        as_type="span",
        end_on_exit=False,          # 收尾留给我们：update_trace 必须赶在 end 之前
    ) as root:
        handle = _TraceHandle(root)
        token = _trace.set(handle)
        try:
            yield handle
        except BaseException as ex:
            _push(root.update, level="ERROR", status_message=str(ex))
            raise
        finally:
            _trace.reset(token)
            _push(root.update_trace, **_trace_fields(root_name, handle.meta))
            _push(root.end)


@contextmanager
def trace_scope(trace_id: str, root_name: str = "request") -> Generator[Any, None, None]:
    """已有 trace 就沿用（API 层已经开过），没有才自己开一条。"""
    existing = _trace.get()
    if existing is not None:
        yield existing
        return
    with start_trace(trace_id, root_name) as handle:
        yield handle


@contextmanager
def trace_span(name: str, **attrs: Any) -> Generator[Any, None, None]:
    """在当前位置开一个节点；没 trace 时整段空转，埋点方不需要写判空。

    父节点由 OTel context 决定：contextvars 会被 asyncio.create_task 复制，
    所以 LangGraph 的并发分支各自挂在自己的链上，不会互相插错父节点。
    """
    if _lf is None or _trace.get() is None:
        # 这里不建 observation，否则后台任务那种空 Context 里会漏出一堆孤儿 trace
        yield _Node({})
        return

    with _lf.start_as_current_observation(
        name=name,
        as_type=_as_type(name),
        metadata=dict(attrs),
        end_on_exit=False,
    ) as obs:
        node = _Node(attrs)
        error: Optional[BaseException] = None
        try:
            yield node
        except BaseException as ex:
            error = ex
            raise
        finally:
            _push(obs.update, metadata=dict(node.attrs),
                  **({"level": "ERROR", "status_message": str(error)} if error else {}))
            _push(obs.end)


def _push(fn: Any, **kwargs: Any) -> None:
    """SDK 侧抛错既不能变成请求失败，也不能顶掉节点自己的业务异常。"""
    try:
        fn(**kwargs)
    except Exception as ex:
        logger.warning(f"Langfuse 写入失败: {type(ex).__name__}: {ex}")


# ── 事件 ──────────────────────────────────────────────────────────────────────

def add_event(name: str, **attrs: Any) -> None:
    """往当前节点挂一条 OTel 事件；链路没活起来就直接返回。

    注意：Langfuse 不摄取 OTel span event，这些事件只在 OTel 侧（如另接 collector）可见。
    """
    if _lf is None or _trace.get() is None:
        return                      # 也是 opentelemetry 的 import 闸门：装了 langfuse 才装了它

    from opentelemetry import trace as otel_trace

    span = otel_trace.get_current_span()
    if not span.is_recording():
        return
    try:
        span.add_event(name, attributes=_otel_attributes(attrs))
    except Exception as ex:
        logger.warning(f"Langfuse 事件写入失败: {type(ex).__name__}: {ex}")


def _otel_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """OTel attributes 只收标量与标量序列；其余值转字符串，脏键值直接丢掉。"""
    out: Dict[str, Any] = {}
    for key, value in attrs.items():
        scalar = isinstance(value, (str, bool, int, float))
        sequence = (isinstance(value, (list, tuple)) and bool(value)
                    and all(isinstance(v, (str, bool, int, float)) for v in value))
        if scalar or sequence:
            out[key] = list(value) if isinstance(value, tuple) else value
        elif value is not None:
            out[key] = str(value)
    return out


# ── trace 级字段 ──────────────────────────────────────────────────────────────

def _trace_fields(root_name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """一次请求在 Langfuse 列表页要能按用户/会话/意图筛。"""
    tags: List[str] = []
    if meta.get("intent"):
        tags.append(f"intent:{meta['intent']}")
    if meta.get("agent_type"):
        tags.append(f"agent:{meta['agent_type']}")
    if meta.get("escalated"):
        tags.append("escalated")
    if meta.get("degradations"):
        tags.append("degraded")
    return {
        "name": f"{root_name}:{meta['intent']}" if meta.get("intent") else root_name,
        "user_id": meta.get("user_id") or None,
        "session_id": meta.get("conv_id") or None,
        "output": {"routing_reason": meta["routing_reason"]} if meta.get("routing_reason") else None,
        "metadata": {k: v for k, v in meta.items() if v is not None},
        "tags": tags,
    }
