"""
外部依赖降级的统一记录。

配置类失败（缺 key、维度不符、服务不通）在启动探测里就抛错终止，不留到运行期；
运行期剩下的只有瞬时故障（网络抖动、单次调用失败），按请求记进 degraded 列表，
随响应一起返回，避免只留一行 warning、响应里完全看不出降级。
"""
import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# URL 里的凭据段：scheme://user:pass@host → scheme://host
_CREDS_RE = re.compile(r"://[^/@]*@")
# 光洗 URL 形态不够：异常文本常把密码单独复述一遍（"…(real=xxx)"），只能按值来洗
_SECRET_ENV_KEYS = (
    "DEEPSEEK_API_KEY", "EMBEDDING_API_KEY", "RERANK_API_KEY",
    "REDIS_PASSWORD", "MILVUS_TOKEN", "OPTISERVE_API_TOKEN",
)


def redact_creds(text: str) -> str:
    """洗掉字符串里的凭据：`scheme://user:pass@` 形态的 URL，以及已知密钥的字面值。

    URL 用子串替换而非 urlsplit：detail 里常见的是把 URI 嵌进异常文本或拼接串，
    整串解析会失败并让凭据原样漏出去。短于 6 位的值不替换，免得把正常词洗花。
    """
    text = _CREDS_RE.sub("://", text)
    for key in _SECRET_ENV_KEYS:
        value = os.getenv(key) or ""
        if len(value) >= 6:
            text = text.replace(value, "***")
    return text


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

    事件 message 会随 /chat 响应回到终端用户手里，而调用方常把异常原文拼进来
    （里面可能有 URI 里的账号密码段或密钥字面值），入事件前统一洗一遍；
    服务端日志保留原文，运维排障要看得到真实细节。
    """
    if log:
        logger.warning(f"降级 [{source.value}/{code}] {message}")
    events = _events.get()
    if events is None:
        return
    if any(e.source == source.value and e.code == code for e in events):
        return
    events.append(DegradeEvent(source=source.value, code=code, message=redact_creds(message)))


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
