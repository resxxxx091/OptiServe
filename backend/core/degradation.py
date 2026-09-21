"""
外部依赖降级的统一记录。

配置类失败（缺 key、维度不符、服务不通）在启动探测里就抛错终止，不留到运行期；
运行期剩下的只有瞬时故障（网络抖动、单次调用失败），按请求记进 degraded 列表，
随响应一起返回，避免只留一行 warning、响应里完全看不出降级。
"""
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class Dep(str, Enum):
    """会被降级的能力来源。"""
    LLM       = "llm"
    EMBEDDING = "embedding"
    RERANKER  = "reranker"  # 检索精排：不通就不启动，运行期失败整次检索判失败，没有兜底路径
    MILVUS    = "milvus"
    REDIS     = "redis"
    AGENT     = "agent"
    TOOL      = "tool"
    MEMORY    = "memory"    # 记忆层自身降级（检索/存取/画像），原因跨多个依赖，具体看事件 message


class DepState(str, Enum):
    OK          = "ok"
    DEGRADED    = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DegradeEvent:
    source:  str
    code:    str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"source": self.source, "code": self.code, "message": self.message}


_events: ContextVar[Optional[List[DegradeEvent]]] = ContextVar("optiserve_degraded", default=None)


def degrade(source: Dep, code: str, message: str, *, log: bool = True) -> None:
    """
    记一条降级事件。同一请求内 (source, code) 只留一条，避免重复。

    log=False 用于配置类缺席——那种情况启动探测时已经报过一次，
    再每请求刷日志就是噪音。
    """
    if log:
        logger.warning(f"降级 [{source.value}/{code}] {message}")
    events = _events.get()
    if events is None:
        return
    if any(e.source == source.value and e.code == code for e in events):
        return
    events.append(DegradeEvent(source=source.value, code=code, message=message))


@contextmanager
def collect_degraded() -> Iterator[List[DegradeEvent]]:
    """包住一次请求，产出其中收集到的降级事件列表。"""
    events: List[DegradeEvent] = []
    token = _events.set(events)
    try:
        yield events
    finally:
        _events.reset(token)


# ── 启动期依赖状态（供 /health 读）────────────────────────────────────────────

_status: Dict[str, Dict[str, str]] = {}


def set_status(source: Dep, state: DepState, detail: str = "") -> None:
    _status[source.value] = {"state": state.value, "detail": detail}


def statuses() -> Dict[str, Dict[str, str]]:
    return {source: dict(info) for source, info in _status.items()}


def any_degraded() -> bool:
    return any(info["state"] != DepState.OK.value for info in _status.values())
