"""
亮点：多轮对话记忆管理

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（Milvus）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（Milvus）—— 从对话中提炼的长期偏好和实体

关键设计：
  - 上下文构建时三级记忆并发读取后融合（CONTEXT_GRAPH），每一路各自带超时预算
  - 工作记忆超过阈值时自动压缩（LLM 摘要），压缩在后台任务里跑，不阻塞 /chat 响应
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import redis.asyncio as redis
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pymilvus import AsyncMilvusClient, DataType, FieldSchema

from core.degradation import Dep, degrade
from core.llm import LLMProvider, message_text
from core.vector_store import (
    AsyncEmbeddingClient,
    CollectionSpec,
    MilvusStore,
    VectorStoreConfig,
    milvus_literal,
    sanitize_text,
)

logger = logging.getLogger(__name__)

MEMORY_COLLECTIONS = (
    CollectionSpec(
        name="episodic",
        description="跨会话的历史对话摘要",
        fields=[
            FieldSchema("user_id", DataType.VARCHAR, max_length=256),
            FieldSchema("conv_id", DataType.VARCHAR, max_length=256),
            FieldSchema("ts", DataType.VARCHAR, max_length=64),
            FieldSchema("summary", DataType.VARCHAR, max_length=8192),
            FieldSchema("full_text", DataType.VARCHAR, max_length=65535),
        ],
    ),
    CollectionSpec(
        name="user_profile",
        description="用户画像：长期偏好与关键实体",
        fields=[
            FieldSchema("user_id", DataType.VARCHAR, max_length=256),
            FieldSchema("conv_id", DataType.VARCHAR, max_length=256),
            FieldSchema("updated_at", DataType.VARCHAR, max_length=64),
            FieldSchema("profile_json", DataType.VARCHAR, max_length=65535),
        ],
    ),
)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 用户画像：偏好、常用实体
    summary:          str             # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages[-8:]:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 Milvus（持久化）。
    Milvus 或 embedding 服务不可用时整体降级：检索返回空、写入丢弃，工作记忆不受影响。
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    KEEP_AFTER_COMPRESS = 5
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数
    SUMMARY_MAX_CHARS = 800
    FULL_TEXT_LIMIT = 2000
    PROFILE_DOC_PREFIX = "user_profile:"
    EPISODIC_COLLECTION = "episodic"
    PROFILE_COLLECTION  = "user_profile"
    SHUTDOWN_DRAIN_S = 5.0

    # 读路径预算：三路并发跑，单路超时就只丢那一路，整个 /chat 最多多等 READ_BUDGET_S。
    READ_BUDGET_S = 1.0
    # 向量服务单独收紧：写路径（压缩、画像）不在 READ_BUDGET_S 里，靠这个上限兜住。
    EMBED_BUDGET_S = 3.0
    REDIS_SOCKET_TIMEOUT_S = 2.0

    def __init__(
        self,
        redis_url:      str = "redis://localhost:6379/0",
        vector_config:  Optional[VectorStoreConfig] = None,
        api_key:        str = "",
        base_url:       Optional[str] = None,
        model:          str = "claude-3-5-sonnet-20241022",
    ):
        self._llm    = LLMProvider(api_key, base_url)
        self._model  = model

        # socket 超时 + 周期健康检查：Redis 卡住时让调用方拿到异常并降级，而不是挂着整个请求
        self._redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=self.REDIS_SOCKET_TIMEOUT_S,
            socket_timeout=self.REDIS_SOCKET_TIMEOUT_S,
            health_check_interval=30,
        )

        self._vector_config = vector_config or VectorStoreConfig.from_env()
        self._embedder = AsyncEmbeddingClient(self._vector_config)
        self._store = MilvusStore(self._vector_config, MEMORY_COLLECTIONS)
        self._compress_tasks: Dict[str, asyncio.Task] = {}

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """预热 Milvus（建/校验两个 collection）；返回 False 表示未就绪，由启动闸门决定去留。"""
        return await self._store.ensure_ready()

    async def close(self) -> None:
        """收尾：等后台压缩跑完 → 关 Milvus → 关 embedding → 关 Redis。"""
        await self._drain_compress_tasks()
        await self._store.aclose()
        await self._embedder.aclose()
        await self._redis.aclose()

    async def _drain_compress_tasks(self) -> None:
        """等后台压缩收尾，超时则取消（消息仍在 Redis，下次压缩会重算）。"""
        pending = [task for task in self._compress_tasks.values() if not task.done()]
        if not pending:
            return
        _, still_running = await asyncio.wait(pending, timeout=self.SHUTDOWN_DRAIN_S)
        for task in still_running:
            task.cancel()
        if still_running:
            await asyncio.gather(*still_running, return_exceptions=True)

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时在后台触发压缩（不阻塞调用方）。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        # 追加到 Redis 列表（左推，最新在前）
        await self._redis.lpush(key, json.dumps({
            "role":      msg.role.value,
            "content":   msg.content,
            "ts":        msg.timestamp.isoformat(),
            "metadata":  msg.metadata,
        }))
        await self._redis.expire(key, 86400)  # 24h TTL

        # 超过压缩阈值时把压缩丢到后台，/chat 不等摘要和向量写入
        if await self._redis.llen(key) >= self.COMPRESS_AT:
            self._schedule_compress(user_id, conv_id)

    def _schedule_compress(self, user_id: str, conv_id: str) -> None:
        """同一会话同时只允许一个压缩在跑，避免重复摘要与重复落库。"""
        # 会话身份直接复用 Redis key：同一个 (user, conv) 在两边是同一个字符串
        task_key = self._wm_key(user_id, conv_id)
        running = self._compress_tasks.get(task_key)
        if running is not None and not running.done():
            return

        task = asyncio.create_task(self._compress_guarded(user_id, conv_id))
        self._compress_tasks[task_key] = task
        task.add_done_callback(lambda done: self._forget_compress_task(task_key, done))

    def _forget_compress_task(self, task_key: str, task: asyncio.Task) -> None:
        if self._compress_tasks.get(task_key) is task:
            self._compress_tasks.pop(task_key, None)

    async def _compress_guarded(self, user_id: str, conv_id: str) -> None:
        """后台压缩不能把异常抛给事件循环，否则只是安静地打日志。"""
        try:
            await self._compress(user_id, conv_id)
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            degrade(Dep.MEMORY, "compress_failed", f"后台压缩失败 {user_id}/{conv_id}: {ex}")

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """
        从当前工作记忆中提炼用户偏好，更新用户画像。

        画像用固定主键 upsert，一个用户始终只有一份；向量由外部 embedding 服务生成。
        拓扑见 PROFILE_GRAPH。
        """
        await PROFILE_GRAPH.ainvoke(
            {"user_id": self._safe_text(user_id), "conv_id": self._safe_text(conv_id)},
            {"configurable": {"memory": self}},
        )

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        三级记忆在 CONTEXT_GRAPH 里并发读取，每一路各自带 READ_BUDGET_S 预算。
        """
        state = await CONTEXT_GRAPH.ainvoke(
            {
                "user_id": self._safe_text(user_id),
                "conv_id": self._safe_text(conv_id),
                "query": self._safe_text(query),
            },
            {"configurable": {"memory": self}},
        )
        return state["result"]

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩（拓扑见 COMPRESS_GRAPH）：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存 Redis（覆盖旧摘要）
          3. 摘要存入情景记忆（Milvus）供跨会话检索
          4. 工作记忆只保留最近 5 条
        """
        await COMPRESS_GRAPH.ainvoke(
            {"user_id": user_id, "conv_id": conv_id},
            {"configurable": {"memory": self}},
        )

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(
        self, user_id: str, conv_id: str, *, full: bool = False
    ) -> List[Message]:
        key = self._wm_key(user_id, conv_id)
        # full=True 给压缩路径用：突发写入可以让列表超过读窗口，
        # 超窗的旧消息若不入摘要就会在 ltrim 时静默消失
        end = -1 if full else self.WORKING_MAX - 1
        raws = await self._redis.lrange(key, 0, end)
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _reset_working_memory(self, user_id: str, conv_id: str, keep: Sequence[Message]) -> None:
        """把工作记忆裁剪到保留边界。"""
        if not keep:
            return
        key = self._wm_key(user_id, conv_id)
        boundary = min(m.timestamp for m in keep)
        current = await self._get_working_memory(user_id, conv_id, full=True)
        keep_len = sum(1 for m in current if m.timestamp >= boundary)
        if keep_len == 0:
            return  # 列表已被 TTL 等清掉，没有可裁的
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.ltrim(key, 0, keep_len - 1)
            pipe.expire(key, 86400)
            await pipe.execute()

    async def _search_episodic(self, user_id: str, conv_id: str, query: str) -> List[str]:
        """语义检索情景记忆：先查同会话，不足 HISTORY_TOP_K 条再放宽到该用户的其他会话。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []

        client = await self._store.client()
        if client is None:
            degrade(Dep.MILVUS, "unavailable", "Milvus 不可用，本次不检索情景记忆")
            return []

        try:
            # 两级检索共用同一个查询向量，外部 embedding 只调用一次
            vector = await self._embedder.embed_query(query_text, timeout=self.EMBED_BUDGET_S)
            docs = self._extract_summaries(await self._query_episodic(
                client,
                vector,
                where=f"user_id == {milvus_literal(user_id)} and conv_id == {milvus_literal(conv_id)}",
            ))
            if len(docs) < self.HISTORY_TOP_K:
                docs.extend(self._extract_summaries(await self._query_episodic(
                    client,
                    vector,
                    where=f"user_id == {milvus_literal(user_id)}",
                )))
            return self._dedupe_texts(docs)[: self.HISTORY_TOP_K]
        except Exception as ex:
            degrade(Dep.MEMORY, "episodic_search_failed", f"情景记忆检索失败，本次无历史片段: {ex}")
            return []

    async def _query_episodic(
        self,
        client: AsyncMilvusClient,
        vector: Sequence[float],
        where: str,
    ) -> List[Dict[str, Any]]:
        return await client.search(
            collection_name=self.EPISODIC_COLLECTION,
            data=[list(vector)],
            limit=self.HISTORY_TOP_K,
            filter=where,
            output_fields=["summary"],
        ) # type: ignore

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        """把压缩后的对话摘要写入情景记忆；被检索的是 summary，text 只作为附带存档。"""
        client = await self._store.client()
        if client is None:
            degrade(Dep.MILVUS, "unavailable", "Milvus 不可用，本次不写入情景记忆")
            return

        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            vector = await self._embedder.embed_query(summary, timeout=self.EMBED_BUDGET_S)
            await client.insert(
                collection_name=self.EPISODIC_COLLECTION,
                data=[{
                    "id": doc_id,
                    "vector": vector,
                    "user_id": user_id,
                    "conv_id": conv_id,
                    "ts": datetime.now().isoformat(),
                    "summary": summary,
                    "full_text": text[: self.FULL_TEXT_LIMIT],
                }],
            )
        except Exception as ex:
            degrade(Dep.MEMORY, "episodic_store_failed", f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像（固定主键取；取不到再按 user_id 兜底取最新一条）。"""
        client = await self._store.client()
        if client is None:
            degrade(Dep.MILVUS, "unavailable", "Milvus 不可用，本次无用户画像")
            return {}

        try:
            doc_id = self._profile_doc_id(user_id)
            rows = await client.query(
                collection_name=self.PROFILE_COLLECTION,
                filter=f"id == {milvus_literal(doc_id)}",
                output_fields=["profile_json"],
            )
            for row in rows or []:
                parsed = self._parse_profile(row.get("profile_json"))
                if parsed:
                    return parsed

            rows = await client.query(
                collection_name=self.PROFILE_COLLECTION,
                filter=f"user_id == {milvus_literal(user_id)}",
                output_fields=["profile_json", "updated_at"],
            )
            return self._latest_profile_from_results(rows)
        except Exception as ex:
            degrade(Dep.MEMORY, "profile_read_failed", f"读取用户画像失败，本次无画像: {ex}")
            return {}

    @staticmethod
    def _key_part(value: Any) -> str:
        """把 id 变成 Redis key 里安全的一段。"""
        return sanitize_text(value).replace("%", "%25").replace(":", "%3A")

    @classmethod
    def _wm_key(cls, user_id: str, conv_id: str) -> str:
        return f"wm:{cls._key_part(user_id)}:{cls._key_part(conv_id)}"

    @classmethod
    def _summary_key(cls, user_id: str, conv_id: str) -> str:
        return f"summary:{cls._key_part(user_id)}:{cls._key_part(conv_id)}"

    @classmethod
    def _profile_doc_id(cls, user_id: str) -> str:
        return f"{cls.PROFILE_DOC_PREFIX}{user_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 Redis / Milvus / HTTP 都能接受的普通 UTF-8 字符串。"""
        return sanitize_text(value)

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/Milvus 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value

    @staticmethod
    def _extract_summaries(hits: List[Dict[str, Any]]) -> List[str]:
        """从 Milvus search 结果（[[{id, distance, entity}, ...]]）里取出摘要文本。"""
        if not hits:
            return []
        rows = hits[0] if isinstance(hits[0], list) else hits
        texts: List[str] = []
        for row in rows:
            entity = row.get("entity") if isinstance(row, dict) else None
            text = (entity or {}).get("summary", "")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        return texts

    @staticmethod
    def _dedupe_texts(values: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    async def _merge_summary(self, old_summary: str, new_summary: str) -> str:
        old_summary = self._safe_text(old_summary).strip()
        new_summary = self._safe_text(new_summary).strip()
        if not old_summary:
            return new_summary[: self.SUMMARY_MAX_CHARS]
        if not new_summary:
            return old_summary[: self.SUMMARY_MAX_CHARS]

        prompt = self._safe_text(
            f"""你是对话摘要器。请把下面两段摘要合并为一段不超过 {self.SUMMARY_MAX_CHARS} 个中文字符的摘要。
保留：用户偏好、关键实体、待办事项、约束条件、未解决问题。
只输出摘要正文，不要编号，不要解释。

旧摘要:
{old_summary}

新增摘要:
{new_summary}
"""
        )
        try:
            chat = self._llm.chat_model(model=self._model, temperature=0.0, max_tokens=256)
            resp = await chat.ainvoke([HumanMessage(content=prompt)])
            merged = self._safe_text(message_text(resp)).strip()
            if merged:
                return merged[: self.SUMMARY_MAX_CHARS]
        except Exception as ex:
            degrade(Dep.LLM, "summary_merge_failed", f"合并摘要失败，回退为截断拼接: {ex}")

        merged = self._safe_text(f"{old_summary}\n{new_summary}").strip()
        return merged[-self.SUMMARY_MAX_CHARS :]

    @classmethod
    def _latest_profile_from_results(cls, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """按 updated_at 取最新一条画像。"""
        candidates = [row for row in rows or [] if row.get("profile_json")]
        if not candidates:
            return {}
        candidates.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        return cls._parse_profile(candidates[0].get("profile_json"))

    @staticmethod
    def _parse_profile(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}


# ── G5：三张记忆图 ────────────────────────────────────────────────────────────
#
# CONTEXT_GRAPH   START → {working ∥ episodic ∥ profile} → assemble → END
# COMPRESS_GRAPH  recent →(不足阈值) END
#                          ↘ summarize → merge → store_episodic → reset → END
# PROFILE_GRAPH   load →(空) END / guard →(向量层不可用) END → extract →(失败) END → store → END
#
# 图内不起 asyncio.create_task：api/main.py 在 API 层起，ContextVar 才看得到降级事件。


def _memory(config) -> "MemoryManager":
    return config["configurable"]["memory"]


class ContextState(TypedDict, total=False):
    user_id: str
    conv_id: str
    query: str
    recent: List["Message"]
    summary: str
    history: List[str]
    profile: Dict[str, Any]
    result: "MemoryContext"


async def read_working(state: ContextState, config) -> Dict[str, Any]:
    """工作记忆 + 会话摘要：同一份 Redis，一并取。"""
    mem = _memory(config)
    try:
        recent, summary = await asyncio.wait_for(
            asyncio.gather(
                mem._get_working_memory(state["user_id"], state["conv_id"]),
                mem._redis.get(mem._summary_key(state["user_id"], state["conv_id"])),
            ),
            timeout=MemoryManager.READ_BUDGET_S,
        )
    except Exception as ex:
        degrade(Dep.MEMORY, "working_memory_failed", f"工作记忆读取失败，本次无最近对话: {ex}")
        return {"recent": [], "summary": ""}
    return {"recent": recent, "summary": summary or ""}


async def _episodic_query(mem: "MemoryManager", state: ContextState) -> List[str]:
    query = state["query"]
    if not query:
        # 没给 query 时沿用「最近一条消息」兜底，所以这一路自己补取一次工作记忆
        recent = await mem._get_working_memory(state["user_id"], state["conv_id"])
        query = recent[-1].content if recent else ""
    return await mem._search_episodic(state["user_id"], state["conv_id"], query)


async def read_episodic(state: ContextState, config) -> Dict[str, Any]:
    mem = _memory(config)
    try:
        history = await asyncio.wait_for(
            _episodic_query(mem, state),
            timeout=MemoryManager.READ_BUDGET_S,
        )
    except Exception as ex:
        degrade(Dep.EMBEDDING, "episodic_timeout", f"情景记忆检索超时，本次无历史片段: {ex}")
        history = []
    return {"history": history}


async def read_profile(state: ContextState, config) -> Dict[str, Any]:
    mem = _memory(config)
    try:
        profile = await asyncio.wait_for(
            mem._get_profile(state["user_id"]),
            timeout=MemoryManager.READ_BUDGET_S,
        )
    except Exception as ex:
        degrade(Dep.MEMORY, "profile_timeout", f"读取用户画像超时，本次无画像: {ex}")
        profile = {}
    return {"profile": profile}


async def assemble_context(state: ContextState, config) -> Dict[str, Any]:
    return {
        "result": MemoryContext(
            recent_messages=state["recent"],
            relevant_history=state["history"],
            user_profile=state["profile"],
            summary=state["summary"],
        )
    }


def build_context_graph():
    graph = StateGraph(ContextState)
    graph.add_node("working", read_working)
    graph.add_node("episodic", read_episodic)
    graph.add_node("profile", read_profile)
    graph.add_node("assemble", assemble_context)

    for name in ("working", "episodic", "profile"):
        graph.add_edge(START, name)
        graph.add_edge(name, "assemble")
    graph.add_edge("assemble", END)
    return graph.compile()


class CompressState(TypedDict, total=False):
    user_id: str
    conv_id: str
    keep: List["Message"]
    text: str
    count: int
    summary: str


async def load_for_compress(state: CompressState, config) -> Dict[str, Any]:
    mem = _memory(config)
    # full=True：压缩必须看到全部积压消息，只读最近 WORKING_MAX 条时，
    # 超窗的旧消息既不进摘要也不入情景记忆，却会被 reset 的 ltrim 裁掉
    messages = await mem._get_working_memory(state["user_id"], state["conv_id"], full=True)
    if len(messages) < mem.COMPRESS_AT:
        return {"keep": []}
    to_compress = messages[:-mem.KEEP_AFTER_COMPRESS]
    return {
        "keep": messages[-mem.KEEP_AFTER_COMPRESS:],
        "count": len(to_compress),
        "text": mem._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress)),
    }


def after_compress_load(state: CompressState) -> str:
    return "summarize" if state.get("keep") else "skip"


async def summarize_oldest(state: CompressState, config) -> Dict[str, Any]:
    """LLM 摘要；失败只改用占位摘要，压缩流程照旧走完。"""
    mem = _memory(config)
    prompt = mem._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{state['text']}")
    try:
        chat = mem._llm.chat_model(model=mem._model, temperature=0.0, max_tokens=256)
        resp = await chat.ainvoke([HumanMessage(content=prompt)])
        return {"summary": mem._safe_text(message_text(resp)).strip()}
    except Exception as ex:
        degrade(Dep.LLM, "compress_summary_failed", f"压缩摘要生成失败，改用占位摘要: {ex}")
        return {"summary": f"对话包含 {state['count']} 条消息（摘要生成失败）"}


async def merge_summary_node(state: CompressState, config) -> Dict[str, Any]:
    """存摘要到 Redis（旧摘要与新摘要合并）。"""
    mem = _memory(config)
    skey = mem._summary_key(state["user_id"], state["conv_id"])
    old_summary = await mem._redis.get(skey) or ""
    new_summary = await mem._merge_summary(old_summary, state["summary"])
    await mem._redis.setex(skey, 86400, new_summary)
    return {}


async def store_episodic_node(state: CompressState, config) -> Dict[str, Any]:
    """摘要存入情景记忆。"""
    await _memory(config)._store_episodic(
        state["user_id"], state["conv_id"], state["text"], state["summary"]
    )
    return {}


async def reset_working_node(state: CompressState, config) -> Dict[str, Any]:
    """重置工作记忆为最近 5 条。"""
    mem = _memory(config)
    await mem._reset_working_memory(state["user_id"], state["conv_id"], state["keep"])
    logger.info(f"工作记忆压缩完成: {state['user_id']}/{state['conv_id']}，摘要 {len(state['summary'])} 字")
    return {}


def build_compress_graph():
    graph = StateGraph(CompressState)
    graph.add_node("load", load_for_compress)
    graph.add_node("summarize", summarize_oldest)
    graph.add_node("merge", merge_summary_node)
    graph.add_node("store_episodic", store_episodic_node)
    graph.add_node("reset", reset_working_node)

    graph.add_edge(START, "load")
    graph.add_conditional_edges("load", after_compress_load, {"summarize": "summarize", "skip": END})
    graph.add_edge("summarize", "merge")
    graph.add_edge("merge", "store_episodic")
    graph.add_edge("store_episodic", "reset")
    graph.add_edge("reset", END)
    return graph.compile()


class ProfileState(TypedDict, total=False):
    user_id: str
    conv_id: str
    client: Any
    text: str
    profile_ctx: str
    profile_json: str


def _profile_ready(state: ProfileState) -> str:
    return "extract" if state.get("client") is not None and state.get("text") else "skip"


async def load_profile_source(state: ProfileState, config) -> Dict[str, Any]:
    mem = _memory(config)
    messages = await mem._get_working_memory(state["user_id"], state["conv_id"])
    if not messages:
        return {"text": ""}
    return {
        "text": mem._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
    }


async def guard_profile_store(state: ProfileState, config) -> Dict[str, Any]:
    """先确认向量层可用，再花一次 LLM 调用；不可用时整个画像更新直接跳过。"""
    mem = _memory(config)
    client = await mem._store.client()
    if client is None:
        degrade(Dep.MILVUS, "unavailable", "Milvus 不可用，本次跳过用户画像更新")
        return {"client": None}
    current_profile = await mem._get_profile(state["user_id"])
    return {
        "client": client,
        "profile_ctx": json.dumps(current_profile, ensure_ascii=False) if current_profile else "{}",
    }


async def extract_profile(state: ProfileState, config) -> Dict[str, Any]:
    mem = _memory(config)
    prompt = mem._safe_text(f"""从以下对话和已有用户画像中提炼或更新用户偏好和关键实体，返回 JSON。
对话:
{state["text"]}

已有画像:
{state["profile_ctx"]}

返回格式: {{"preferences": ["..."], "entities": {{"产品": [], "问题类型": []}}}}""")
    try:
        chat = mem._llm.chat_model(model=mem._model, temperature=0.0, max_tokens=512)
        resp = await chat.ainvoke([HumanMessage(content=prompt)])
        raw = message_text(resp)
        s, e = raw.find("{"), raw.rfind("}") + 1
        profile_data = json.loads(raw[s:e])
    except Exception as ex:
        degrade(Dep.MEMORY, "profile_update_failed", f"更新用户画像失败: {ex}")
        return {"profile_json": ""}
    return {"profile_json": mem._safe_text(json.dumps(profile_data, ensure_ascii=False))}


async def store_profile(state: ProfileState, config) -> Dict[str, Any]:
    mem = _memory(config)
    try:
        vector = await mem._embedder.embed_query(state["profile_json"], timeout=MemoryManager.EMBED_BUDGET_S)
        await state["client"].upsert(
            collection_name=mem.PROFILE_COLLECTION,
            data=[{
                "id": mem._profile_doc_id(state["user_id"]),
                "vector": vector,
                "user_id": state["user_id"],
                "conv_id": state["conv_id"],
                "updated_at": datetime.now().isoformat(),
                "profile_json": state["profile_json"],
            }],
        )
        logger.info(f"用户画像已更新: {state['user_id']}")
    except Exception as ex:
        degrade(Dep.MEMORY, "profile_update_failed", f"更新用户画像失败: {ex}")
    return {}


def build_profile_graph():
    graph = StateGraph(ProfileState)
    graph.add_node("load", load_profile_source)
    graph.add_node("guard", guard_profile_store)
    graph.add_node("extract", extract_profile)
    graph.add_node("store", store_profile)

    graph.add_edge(START, "load")
    graph.add_conditional_edges("load", lambda s: "guard" if s.get("text") else "skip",
                                {"guard": "guard", "skip": END})
    graph.add_conditional_edges("guard", _profile_ready, {"extract": "extract", "skip": END})
    graph.add_conditional_edges("extract", lambda s: "store" if s.get("profile_json") else "skip",
                                {"store": "store", "skip": END})
    graph.add_edge("store", END)
    return graph.compile()


CONTEXT_GRAPH = build_context_graph()
COMPRESS_GRAPH = build_compress_graph()
PROFILE_GRAPH = build_profile_graph()
