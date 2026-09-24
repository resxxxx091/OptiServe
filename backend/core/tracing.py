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
    """一条进行中 trace：装 trace 级 meta、input/output 和根 observation，不装孩子也不装时间。

    input/output 是评测器唯一读得到的两个字段（它不遍历子节点），所以 trace 级和根节点各写一份。
    """

    def __init__(self, root: Any, input: Any = None):
        self.root = root
        self.input = input
        self.output: Any = None
        self.meta: Dict[str, Any] = {}


class _Node:
    """埋点方看到的节点把手。attrs/output 累加到节点退出时才推给 SDK ——
    update(metadata=) 是整值覆盖，而 .end() 之后再也写不进去。"""
    __slots__ = ("attrs", "output")

    def __init__(self, attrs: Dict[str, Any]):
        self.attrs: Dict[str, Any] = dict(attrs)
        self.output: Any = None


_trace: ContextVar[Optional[_TraceHandle]] = ContextVar("optiserve_trace", default=None)


# ── 开 trace ──────────────────────────────────────────────────────────────────

@contextmanager
def start_trace(trace_id: str, root_name: str = "request", input: Any = None) -> Generator[Any, None, None]:
    """新开一次 trace：根节点活到请求收尾，退出时先写 trace 级字段再结束它。

    input 在开 trace 时就定下：请求中途抛错时收尾的 update_trace 仍会把它写出去。
    """
    if _lf is None:
        # 仍然占住 ContextVar：trace_scope 靠它按身份复用同一条 trace
        handle = _TraceHandle(None, input)
        token = _trace.set(handle)
        try:
            yield handle
        finally:
            _trace.reset(token)
        return

    # trace_id 由 request_id 派生：同一个 request_id 重复开仍落在同一条 trace 上，
    seeded = _lf.create_trace_id(seed=f"optiserve:{trace_id}")
    with _lf.start_as_current_observation(
        trace_context={"trace_id": seeded},
        name=root_name,
        as_type="span",
        input=input,                # 根节点也留一份：按 observation 挂的评测器读不到 trace 级字段
        end_on_exit=False,          # 收尾留给我们：update_trace 必须赶在 end 之前
    ) as root:
        handle = _TraceHandle(root, input)
        token = _trace.set(handle)
        try:
            yield handle
        except BaseException as ex:
            _push(root.update, level="ERROR", status_message=str(ex))
            raise
        finally:
            _trace.reset(token)
            _push(root.update, output=handle.output)
            _push(root.update_trace, **_trace_fields(root_name, handle))
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
def trace_span(name: str, input: Any = None, **attrs: Any) -> Generator[Any, None, None]:
    """在当前位置开一个节点；没 trace 时整段空转，埋点方不需要写判空。

    input 走 SDK 的 Input 栏，其余 attrs 进 Metadata —— 评测器按 observation 挂时只认
    Input/Output 两栏，塞进 Metadata 就等于没写。结果要跑出来才有的，退出前写 span.output。
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
        input=input,
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
            _push(obs.update, metadata=dict(node.attrs), output=node.output,
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
    """往当前节点挂一条 OTel 事件；链路没活起来就直接返回。"""
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

def _trace_fields(root_name: str, handle: "_TraceHandle") -> Dict[str, Any]:
    """一次请求在 Langfuse 列表页要能按用户/会话/意图筛。"""
    meta = handle.meta
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
        "input": handle.input,
        "output": handle.output,
        "metadata": {k: v for k, v in meta.items() if v is not None},
        "tags": tags,
    }
