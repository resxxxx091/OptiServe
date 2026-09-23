"""
外部依赖降级的统一记录。

配置类失败（缺 key、维度不符、服务不通）在启动探测里就抛错终止，不留到运行期；
运行期失败（网络抖动、单次调用失败），按请求记进 degraded 列表，随响应一起返回，避免只留一行 warning、响应里完全看不出降级。
"""
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class Dep(str, Enum):
    """会被降级的能力来源。"""
    LLM       = "llm"
    EMBEDDING = "embedding"
    RERANKER  = "reranker"  
    MILVUS    = "milvus"
    REDIS     = "redis"
    AGENT     = "agent"
    TOOL      = "tool"
    MEMORY    = "memory"    # 记忆层自身降级（检索/存取/画像），原因跨多个依赖，具体看事件 message


# 服务状态枚举（能不能 ping 通），供 /health 读。
class DepState(str, Enum):
    OK          = "ok"
    UNAVAILABLE = "unavailable"


# 每次请求内的降级事件，按 source/code 唯一。message 里的是具体的异常信息。
@dataclass(frozen=True)
class DegradeEvent:
    source:  str
    code:    str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"source": self.source, "code": self.code, "message": self.message}


# 每次请求的降级事件列表：深处调 degrade() 直接 append，免得把降级原因逐层 return 上去
_events: ContextVar[Optional[List[DegradeEvent]]] = ContextVar("optiserve_degraded", default=None)


def degrade(source: Dep, code: str, message: str) -> None:
    """记一条降级事件。同一请求内 (source, code) 只留一条，避免重复。"""
    logger.warning(f"降级 [{source.value}/{code}] {message}")
    events = _events.get()
    if events is None:
        return
    if any(e.source == source.value and e.code == code for e in events):
        return
    events.append(DegradeEvent(source=source.value, code=code, message=message))


@contextmanager
def collect_degraded() -> Generator[List[DegradeEvent], None, None]:
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
