"""
把 core/tracing.py 的 span 树重放进 Langfuse —— 全项目唯一持有 langfuse 引用的文件。

为什么单独一层：Langfuse v3 的重放只需要两个动作 —— 在父 otel span 的上下文里
start_observation()（父子链自然成立），再用记录好的墙钟时间覆写 _start_time 与
end(end_time=…)（各阶段耗时口径不丢）。SDK 版本漂移只会坏这一个文件，埋点方与测试
都不感知 Langfuse 的存在。

三种情况都只让导出变成 no-op，绝不影响请求链路：没装 SDK、没配密钥（或仍是 .env
模板里的 xxx 占位）、导出过程中抛异常。no-op 时本地也不留副本，链路记录只有 Langfuse 一份。
"""
import logging
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# .env 模板里的占位值：等价于没配，不要 attempt 连接
_PLACEHOLDERS = {"", "xxx", "your-key", "your_api_key", "changeme", "none"}

# 我们的 span 名前缀 → Langfuse 的 observation 类型，让 UI 按语义分组
_AS_TYPE = (
    ("tool:", "tool"),
    ("agent:", "agent"),
    ("rag.", "retriever"),
)


def as_type(name: str) -> str:
    for prefix, kind in _AS_TYPE:
        if name.startswith(prefix):
            return kind
    return "span"


def _to_ns(iso: str) -> Optional[int]:
    """ISO 墙钟串 → OTel 纳秒时间戳；空串或脏值返回 None，让 SDK 用当前时间。"""
    if not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso).timestamp() * 1_000_000_000)
    except (TypeError, ValueError):
        return None


class LangfuseExporter:
    """一棵已完成的 span 树 → 一个 Langfuse trace。由 core.tracing 的收尾钩子调用。"""

    def __init__(self, client: Any):
        self._lf = client

    def export(self, tree: Dict[str, Any]) -> None:
        """入口必须吞掉一切异常：上报失败不能让一次客服请求变红。"""
        try:
            self._replay(tree)
        except Exception as ex:
            logger.warning(f"Langfuse 导出失败 trace_id={tree.get('trace_id')}: {ex}")

    def shutdown(self) -> None:
        """应用退出时把批量队列里剩下的 span 冲出去。"""
        try:
            self._lf.shutdown()
        except Exception as ex:
            logger.warning(f"Langfuse 关闭失败: {ex}")

    # ── 重放 ──────────────────────────────────────────────────────────────────

    def _replay(self, tree: Dict[str, Any]) -> None:
        # trace_id 由 request_id 派生：同一个 request_id 重复导出仍落在同一条 trace 上
        trace_id = self._lf.create_trace_id(seed=f"optiserve:{tree['trace_id']}")
        self._node(tree["root"], trace_id, None, tree.get("meta") or {})

    def _node(self, node: Dict[str, Any], trace_id: str,
              parent_otel: Any, meta: Dict[str, Any]) -> None:
        from opentelemetry import trace as otel_trace

        error = node.get("status") == "error"
        kwargs: Dict[str, Any] = {
            "name": node["name"],
            "as_type": as_type(node["name"]),
            "metadata": _metadata(node),
            "level": "ERROR" if error else None,
            "status_message": node.get("error") or None,
        }
        # 只有根需要显式声明 trace_id；子片段靠 OTel 上下文挂到父上
        ctx = otel_trace.use_span(parent_otel, end_on_exit=False) if parent_otel else nullcontext()
        with ctx:
            if parent_otel is None:
                kwargs["trace_context"] = {"trace_id": trace_id}
            span = self._lf.start_observation(**kwargs)

        otel = getattr(span, "_otel_span", None)
        start = _to_ns(node.get("start_time", ""))
        if otel is not None and start is not None and hasattr(otel, "_start_time"):
            otel._start_time = start
        if parent_otel is None:
            span.update_trace(**_trace_fields(node, meta))

        for child in node.get("children") or []:
            self._node(child, trace_id, otel, meta)
        span.end(end_time=_to_ns(node.get("end_time", "")))


def _metadata(node: Dict[str, Any]) -> Dict[str, Any]:
    """埋点方算好的 attrs 原样带过去，事件与本地 span_id 一并留在 metadata 里。"""
    return {
        **node.get("attrs", {}),
        "span_id": node.get("span_id"),
        "parent_id": node.get("parent_id"),
        "latency_ms": node.get("latency_ms"),
        "events": node.get("events") or [],
    }


def _trace_fields(node: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """trace 级字段：一次请求在 Langfuse 列表页要能按用户/会话/意图筛。"""
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
        "name": f"{node['name']}:{meta['intent']}" if meta.get("intent") else node["name"],
        "user_id": meta.get("user_id") or None,
        "session_id": meta.get("conv_id") or None,
        "output": {"routing_reason": meta["routing_reason"]} if meta.get("routing_reason") else None,
        "metadata": {k: v for k, v in meta.items() if v is not None},
        "tags": tags,
    }


def create_exporter() -> Optional[LangfuseExporter]:
    """配了真密钥才建客户端；任何一步不成都返回 None，服务照常跑。"""
    public = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip().lower()
    secret = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip().lower()
    if public in _PLACEHOLDERS or secret in _PLACEHOLDERS:
        logger.info("未配置 Langfuse 密钥，本次运行不导出任何链路记录")
        return None
    try:
        from langfuse import Langfuse   # 密钥齐全才 import，没装 SDK 也不影响启动

        # 不传参：host / 密钥 / 采样率由 SDK 自己按 LANGFUSE_* 环境变量解析
        return LangfuseExporter(Langfuse())
    except Exception as ex:
        logger.warning(f"Langfuse 客户端初始化失败，本次启动不上报: {ex}")
        return None
