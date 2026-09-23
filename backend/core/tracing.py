"""
一次请求的 span 树：编排 → Agent → 工具调用 → RAG 检索各阶段的父子耗时。

与 core/degradation.py 用同一套 ContextVar 机制：API 层 start_trace() 包住整次
请求，图内节点直接 current_trace()/trace_span() 取用——asyncio 任务创建时复制
上下文，所以 LangGraph 的节点与 Send 扇出分支都能看到同一个 recorder。父子关系
记在 _active 这个上下文变量上，因此并发分支各自挂在自己的链上，不会互相插错父节点。
"""
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """一次有始有终的执行片段。children 只在内存里挂链，导出时由 as_dict 展开。"""
    span_id:    str
    trace_id:   str
    name:       str
    parent_id:  Optional[str]     = None
    start_time: str               = ""
    end_time:   str               = ""
    latency_ms: float             = 0.0
    status:     str               = "ok"      # ok / error
    error:      str               = ""
    attrs:      Dict[str, Any]    = field(default_factory=dict)
    events:     List[Dict[str, Any]] = field(default_factory=list)
    children:   List["Span"]      = field(default_factory=list)
    start_ts:   float             = 0.0       # monotonic 锚点，不进导出结构

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_id":    self.span_id,
            "trace_id":   self.trace_id,
            "name":       self.name,
            "parent_id":  self.parent_id,
            "start_time": self.start_time,
            "end_time":   self.end_time,
            "latency_ms": self.latency_ms,
            "status":     self.status,
            "error":      self.error,
            "attrs":      dict(self.attrs),
            "events":     list(self.events),
            "children":   [child.as_dict() for child in self.children],
        }


class TraceRecorder:
    """一次请求的 span 收集器。span() 开子片段，event() 往当前片段挂事件。"""

    def __init__(self, trace_id: str, root_name: str = "request"):
        self.trace_id = trace_id
        self.meta:    Dict[str, Any] = {}
        self._seq   = 0
        self._spans: List[Span] = []
        self.root   = self._new(root_name, None)

    # ── 构建 ──────────────────────────────────────────────────────────────────

    def _new(self, name: str, parent: Optional[Span], **attrs: Any) -> Span:
        self._seq += 1
        span = Span(
            span_id=f"{self.trace_id}-{self._seq}",
            trace_id=self.trace_id,
            name=name,
            parent_id=parent.span_id if parent else None,
            start_time=datetime.now().isoformat(),
            start_ts=time.monotonic(),
            attrs=dict(attrs),
        )
        self._spans.append(span)
        if parent is not None:
            parent.children.append(span)
        return span

    def _close(self, span: Span) -> None:
        span.latency_ms = round((time.monotonic() - span.start_ts) * 1000, 1)
        span.end_time = datetime.now().isoformat()

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = _active.get() or self.root
        span = self._new(name, parent, **attrs)
        token = _active.set(span)
        try:
            yield span
        except Exception as ex:
            span.status = "error"
            span.error = str(ex)
            raise
        finally:
            _active.reset(token)
            self._close(span)

    def event(self, name: str, **attrs: Any) -> None:
        target = _active.get() or self.root
        target.events.append({
            "name":  name,
            "time":  datetime.now().isoformat(),
            "attrs": dict(attrs),
        })

    # ── 导出 ──────────────────────────────────────────────────────────────────

    def as_tree(self) -> Dict[str, Any]:
        return {
            "trace_id":   self.trace_id,
            "span_count": len(self._spans),
            "latency_ms": self.root.latency_ms,
            "status":     self.root.status,
            "meta":       dict(self.meta),
            "root":       self.root.as_dict(),
        }


# ── 上下文 ────────────────────────────────────────────────────────────────────

_trace:  ContextVar[Optional[TraceRecorder]] = ContextVar("optiserve_trace", default=None)
_active: ContextVar[Optional[Span]]          = ContextVar("optiserve_active_span", default=None)


def current_trace() -> Optional[TraceRecorder]:
    return _trace.get()


@contextmanager
def start_trace(trace_id: str, root_name: str = "request") -> Iterator[TraceRecorder]:
    """新开一次 trace，退出时落进环形缓冲并触发导出钩子。"""
    recorder = TraceRecorder(trace_id, root_name)
    token_trace  = _trace.set(recorder)
    token_active = _active.set(recorder.root)
    try:
        yield recorder
    except Exception as ex:
        recorder.root.status = "error"
        recorder.root.error  = str(ex)
        raise
    finally:
        _active.reset(token_active)
        _trace.reset(token_trace)
        recorder._close(recorder.root)
        publish(recorder)


@contextmanager
def trace_scope(trace_id: str, root_name: str = "request") -> Iterator[TraceRecorder]:
    """已有 trace 就沿用（API 层已经开过），没有才自己开一条并发布。

    编排图与检索链路的入口都用它，这样 /chat、/search、评测三条来路都能拿到树，
    而嵌套调用不会开出第二条互不可见的 trace。
    """
    existing = _trace.get()
    if existing is not None:
        yield existing
        return
    with start_trace(trace_id, root_name) as recorder:
        yield recorder


@contextmanager
def trace_span(name: str, **attrs: Any) -> Iterator[Optional[Span]]:
    """在当前位置开子片段；没有 trace 时整段空转，埋点方不需要写判空。"""
    recorder = _trace.get()
    if recorder is None:
        yield None
        return
    with recorder.span(name, **attrs) as span:
        yield span


def add_event(name: str, **attrs: Any) -> None:
    """往当前片段挂一条事件；没有 trace 时空转。"""
    recorder = _trace.get()
    if recorder is not None:
        recorder.event(name, **attrs)


# ── 导出钩子 ──────────────────────────────────────────────────────────────────

_finish_hook: Optional[Callable[[Dict[str, Any]], None]] = None


def set_finish_hook(hook: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """注册 trace 收尾回调（Langfuse 上报走这里）。传 None 撤销。"""
    global _finish_hook
    _finish_hook = hook


def publish(recorder: TraceRecorder) -> None:
    """把一次已完成的 span 树交给导出钩子。导出异常绝不影响请求链路。

    没有钩子时连 as_tree() 都不做：span 树随 recorder 一起释放。
    """
    if _finish_hook is None:
        return
    try:
        _finish_hook(recorder.as_tree())
    except Exception as ex:
        logger.warning(f"trace 导出失败 trace_id={recorder.trace_id}: {ex}")
