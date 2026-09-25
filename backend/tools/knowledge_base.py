"""
一、文档导入：将文本切片后写入 Milvus，每条切片同时落稠密向量和 bge-m3 稀疏向量

二、检索优化链路 RETRIEVAL_GRAPH，五级串联：
  问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断

  1. 问题改写：LLM 把用户原始问题扩写成多个角度的子查询，解决"一个问法只召回一个角度"。
  2. 混合索引召回：每个子查询同时打稠密向量（语义）与稀疏向量（关键词命中）两路 ANN。
  3. RRF 粗排：对「子查询 × 索引路」共 2N 份排名做 Reciprocal Rank Fusion（score = Σ 1/(k+rank)，各路等权）。
  4. Reranker 精排：交叉编码器对粗排候选逐条 (query, doc) 打分，它看得到 query 与文档的交互，比向量相似度更接近"对这个问题有用"。
  5. 断崖截断：精排分数相邻两位的落差超阈值就在那里切断，而不是固定取 top-K 。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
)

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pymilvus import DataType, FieldSchema

from core.degradation import Dep, degrade
from core.llm import LLMProvider, message_text
from core.tracing import trace_span
from core.vector_store import (
    VECTOR_FIELD,
    AsyncEmbeddingClient,
    AsyncRerankClient,
    CollectionSpec,
    MilvusStore,
    RerankError,
    VectorStoreConfig,
    VectorStoreError,
    sanitize_text,
)
from tools.agent_tools import AgentToolSpec, make_tool
from tools.tool_manager import Tool, ToolResult

if TYPE_CHECKING:
    from agents.base import Request
    from tools.tool_manager import ToolRegistry

logger = logging.getLogger(__name__)

# 注册进 ToolRegistry 的检索工具名：召回、检索链、Agent 侧适配器三处共用同一个字面量
KNOWLEDGE_SEARCH_TOOL = "knowledge_search"

KNOWLEDGE_COLLECTIONS = (
    CollectionSpec(
        name="knowledge_base",
        description="OptiServe RAG 知识库（稠密 + 稀疏混合索引）",
        fields=[
            FieldSchema("title", DataType.VARCHAR, max_length=512),
            FieldSchema("content", DataType.VARCHAR, max_length=65535),
            FieldSchema("chunk_index", DataType.INT64),
            FieldSchema("total_chunks", DataType.INT64),
        ],
        # 混合索引的第二路：外部 bge-m3 生成的词表稀疏向量，索引度量 IP
        sparse_fields=("sparse_vector",),
    ),
)


class KnowledgeBase:
    """基于 Milvus 混合索引的 RAG 知识库（向量由外部 embedding 服务生成）。"""

    COLLECTION_NAME = "knowledge_base"
    SPARSE_FIELD = "sparse_vector"
    CHUNK_SIZE = 500
    # 单路召回条数上限：RRF 是排名融合，两路各取一份即可，再多只是浪费 ANN 打分
    MAX_RECALL_PER_PATH = 50
    _OUTPUT_FIELDS = ["title", "content", "chunk_index"]

    def __init__(self, vector_config: Optional[VectorStoreConfig] = None):
        self._vector_config = vector_config or VectorStoreConfig.from_env()
        self._embedder = AsyncEmbeddingClient(self._vector_config)
        self._store = MilvusStore(self._vector_config, KNOWLEDGE_COLLECTIONS)

    async def start(self) -> bool:
        """确保 collection 已就绪；知识库为空时播种默认文档。失败返回 False，由启动闸门决定去留。"""
        if not await self._store.ensure_ready():
            return False

        if await self.doc_count_async() == 0:
            await self._load_default_docs()
        return True

    async def close(self) -> None:
        await self._store.aclose()
        await self._embedder.aclose()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字），稠密 + 稀疏两路向量批量生成后按确定性主键 upsert，
        同一份内容重复导入会覆盖而不是堆积。混合索引要求两路向量同批写入，
        所以 embedding 服务必须同时给得出稀疏那一路，否则整批导入直接抛错。
        """
        rows: List[Dict[str, Any]] = []
        chunks: List[str] = []

        for doc in documents:
            title   = sanitize_text(doc.get("title", ""))
            content = sanitize_text(doc.get("content", ""))
            pieces  = self._chunk_text(content, chunk_size=self.CHUNK_SIZE)

            for i, chunk in enumerate(pieces):
                rows.append({
                    "id":           hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest(),
                    "title":        title,
                    "content":      chunk,
                    "chunk_index":  i,
                    "total_chunks": len(pieces),
                })
                chunks.append(chunk)

        if not rows:
            return 0

        client = await self._store.client()
        if client is None:
            raise VectorStoreError("Milvus 不可用，知识库写入失败")

        vectors = await self._embedder.embed_documents_hybrid(chunks)
        await client.upsert(
            collection_name=self.COLLECTION_NAME,
            data=[
                dict(row, vector=dense, **{self.SPARSE_FIELD: sparse})
                for row, (dense, sparse) in zip(rows, vectors)
            ],
        )
        logger.info(f"知识库导入 {len(rows)} 个文档片段（稠密 + 稀疏双向量）")
        return len(rows)

    async def recall_async(self, query: str, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        混合索引召回：一次 query 编码，稠密与稀疏两路 ANN 并行检索。

        返回 {"dense": [...], "sparse": [...]} 两份**各自有序**的列表 —— 这里刻意不融合，
        RRF 要在上层把「多个子查询 × 两路索引」共 2N 份排名一起融合，
        在这一层就合并掉的话排名信息就只剩一份了。

        稠密路是 COSINE，distance 即相似度（越大越相关）；稀疏路是 IP，分数没有距离含义。
        """
        query_text = sanitize_text(query).strip()
        if not query_text or limit <= 0:
            return {"dense": [], "sparse": []}

        client = await self._store.client()
        if client is None:
            raise VectorStoreError("Milvus 不可用，知识库检索失败")

        per_path = min(limit, self.MAX_RECALL_PER_PATH)
        dense_vector, sparse_vector = await self._embedder.embed_query_hybrid(query_text)
        dense_hits, sparse_hits = await asyncio.gather(
            client.search(
                collection_name=self.COLLECTION_NAME,
                data=[dense_vector],
                anns_field=VECTOR_FIELD,
                limit=per_path,
                filter="",
                output_fields=self._OUTPUT_FIELDS,
            ),
            client.search(
                collection_name=self.COLLECTION_NAME,
                data=[sparse_vector],
                anns_field=self.SPARSE_FIELD,
                limit=per_path,
                filter="",
                output_fields=self._OUTPUT_FIELDS,
            ),
        )

        return {
            "dense":  self._format_hits(dense_hits, "dense"),
            "sparse": self._format_hits(sparse_hits, "sparse"),
        }

    @staticmethod
    def _format_hits(hits: Any, path: str) -> List[Dict[str, Any]]:
        """把一路 Milvus 命中转成统一形状；主键 id 是 RRF 融合时的去重依据。"""
        rows = hits[0] if hits and isinstance(hits[0], list) else []
        items: List[Dict[str, Any]] = []
        for row in rows:
            entity = row.get("entity") or {}
            items.append({
                "id":     row.get("id"),
                "title":  entity.get("title", ""),
                "content": entity.get("content", ""),
                "score":  round(float(row.get("distance", 0.0)), 4),
                "chunk":  entity.get("chunk_index", 0),
                "path":   path,
            })
        return items

    async def doc_count_async(self) -> int:
        """文档片段总数；Milvus 不可用时返回 0（启动统计不该因此失败）。"""
        client = await self._store.client()
        if client is None:
            return 0

        try:
            rows = await client.query(
                collection_name=self.COLLECTION_NAME,
                filter="",
                output_fields=["count(*)"],
            )
        except Exception as ex:
            logger.warning(f"统计知识库片段数失败: {ex}")
            return 0

        if not rows:
            return 0
        return int(rows[0].get("count(*)", 0))

    # ── 检索工具 handler 与注册 ───────────────────────────────────────────────

    async def search_handler(
        self, params: Dict[str, Any], context: Any
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        作为检索工具的 handler 注册：一次调用 = 一个查询的混合索引双路召回。

        这里只做到召回为止。RRF 粗排、Reranker 精排、断崖截断在本文件的检索图里做，
        因为融合必须同时看见「改写出的多个子查询 × 两路索引」的全部排名，
        在本层合并就等于把粗排的输入削成一份。
        """
        query = params.get("query", "")
        # 内部检索链路会传 max(top_k, recall_k)，这里只挡直连调用塞进来的天文数字
        limit = max(1, min(500, int(params.get("top_k", 5) or 5)))
        return await self.recall_async(query, limit=limit)

    def search_tool(self) -> Tool:
        """要注册进 ToolRegistry 的那条召回工具。"""
        return Tool(
            name=KNOWLEDGE_SEARCH_TOOL,
            description="搜索 RAG 知识库",
            handler=self.search_handler,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "description": "每路索引各取多少条候选"},
                },
                "required": ["query"],
            },
            cache_ttl=300.0,
            fallback=self._fallback,
        )

    @staticmethod
    def _fallback(
        params: Dict[str, Any], context: Any, error: str
    ) -> List[Dict[str, Any]]:
        """Milvus 不可用时的降级结果：一条兜底文档，_rank_lists 把裸列表当单路召回，下游不用分支。"""
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的混合索引检索。请稍后重试。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    async def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        try:
            await self.add_documents_async(default_docs)
            logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
        except Exception as ex:
            logger.warning(f"默认知识库播种失败: {ex}")


# ── 检索链路参数 ──────────────────────────────────────────────────────────────

# 问题改写的子查询条数：含原始查询在内，一次检索最多扇出这么多次召回
REWRITE_SUB_QUERIES = 3

T = TypeVar("T")


@dataclass(frozen=True)
class RetrievalConfig:
    """RRF 融合与断崖截断的阈值；全部走环境变量，评测调参不用改代码。"""

    rrf_k: int = 60                # RRF 平滑常数：压住名次靠后时的边际差异
    recall_k: int = 10             # 每个子查询在每路索引上取的候选条数（召回宽度）
    coarse_n: int = 20             # 粗排后送进精排的候选数
    gap_abs: float = 0.5           # 相邻分数绝对落差阈值
    gap_ratio: float = 0.25        # 相邻分数相对落差阈值（相对前一名）
    rerank_max_chars: int = 1000   # 送进精排的单篇文档截断长度
    rewrite_timeout_s: float = 15.0   # 问题改写这一次 LLM 调用的时限
    total_timeout_s: float = 45.0     # 整条检索链的总预算，小于「单工具超时 × 级数」的最坏值

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            rrf_k=int(os.getenv("RETRIEVAL_RRF_K", "60")),
            recall_k=int(os.getenv("RETRIEVAL_RECALL_K", "10")),
            coarse_n=int(os.getenv("RETRIEVAL_COARSE_N", "20")),
            gap_abs=float(os.getenv("RERANK_GAP_ABS", "0.5")),
            gap_ratio=float(os.getenv("RERANK_GAP_RATIO", "0.25")),
            rerank_max_chars=int(os.getenv("RERANK_MAX_CHARS", "1000")),
            rewrite_timeout_s=float(os.getenv("RETRIEVAL_REWRITE_TIMEOUT_S", "15.0")),
            total_timeout_s=float(os.getenv("RETRIEVAL_TOTAL_TIMEOUT_S", "45.0")),
        )


# ── G4：检索优化图 ────────────────────────────────────────────────────────────
#
#     START → rewrite → recall → rrf →(有候选) rerank → END
#                                 ↘(候选空) give_up → END
#
# recall 节点内部保留 asyncio.gather：子查询是个位数量级，换成 Send 扇出就要在 state 里
# 额外带下标才能复原"哪一路回来的"，收益为负。

class RetrievalState(TypedDict, total=False):
    tool_name: str
    query: str
    top_k: int
    recall_k: int
    context: Optional[Dict[str, Any]]
    sub_queries: List[str]
    recalls: List[Any]                                  # 每个子查询一个 ToolResult
    coarse: List[Tuple[Any, float]]                     # RRF 融合后的 (文档, 粗排分)
    degraded: bool
    stages: Dict[str, int]
    result: "ToolResult"


def _pipeline(config) -> "RetrievalPipeline":
    return config["configurable"]["pipeline"]


def _doc_key(item: Any) -> str:
    """RRF 融合时识别"同一个文档"的键：优先向量库主键，退到内容哈希。"""
    if isinstance(item, dict):
        doc_id = item.get("id") or item.get("chunk_id")
        if doc_id:
            return f"id:{doc_id}"
        entity = item.get("entity")
        if isinstance(entity, dict) and entity.get("chunk_id"):
            return f"id:{entity['chunk_id']}"
    return "h:" + hashlib.md5(str(item).encode()).hexdigest()


def _rank_lists(data: Any) -> List[Tuple[str, List[Any]]]:
    """把一次召回的返回值拆成 (索引路, 有序列表)。

    知识库 handler 返回 {"dense": [...], "sparse": [...]}；只返回普通列表的工具
    当作单路处理，融合后仍保持原有顺序。
    """
    if isinstance(data, dict) and ("dense" in data or "sparse" in data):
        paths: List[Tuple[str, List[Any]]] = []
        for path in ("dense", "sparse"):
            items = data.get(path)
            paths.append((path, list(items) if isinstance(items, list) else []))
        return paths
    if isinstance(data, list):
        return [("dense", data)]
    return []


def rrf_fuse(
    lists: Sequence[Tuple[str, Sequence[Any]]], cfg: RetrievalConfig
) -> List[Tuple[Any, float]]:
    """Reciprocal Rank Fusion：score(doc) = Σ 1 / (rrf_k + 该路名次)，各路等权。

    只用名次不用分数，所以稠密的 COSINE 与稀疏的 IP 可以同台融合；
    同一篇文档被越多路命中、且名次越靠前，累积分越高。
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Any] = {}
    for _path, items in lists:
        for rank, item in enumerate(items, start=1):
            key = _doc_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (cfg.rrf_k + rank)
            docs.setdefault(key, item)

    # sorted 稳定 + dict 保持插入序 → 同分时按"首次出现的路的原序"，结果可复现
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [(docs[key], scores[key]) for key in ordered[:cfg.coarse_n]]


def cliff_truncate(
    items: Sequence[T],
    score_of: Callable[[T], float],
    top_k: int,
    cfg: RetrievalConfig,
) -> List[T]:
    """断崖截断：在 top_k 之内找第一道落差超阈值的相邻对，在那里切断。

    两个阈值同时存在是因为量纲与分布随模型而异：绝对落差管"整体都低分"，
    相对落差管"前几名挤在一起、后面突然塌"。命中即停，只看第一道断崖。
    窗口内没找到断层就按名次取满 top_k，不设保底篇数。

    满足任一即切断，所以生效边界取决于分数量纲：精排返回的是 [0,1] 概率分，
    head<=1 时 gap_ratio*head 恒不大于 gap_abs，切断点总是先由相对阈值命中，
    gap_abs 只在未归一化的分数量纲下才可能单独起作用。
    """
    keep = min(len(items), top_k)
    for index in range(keep - 1):
        head, tail = score_of(items[index]), score_of(items[index + 1])
        gap = head - tail
        if gap >= cfg.gap_abs or gap >= cfg.gap_ratio * abs(head):
            keep = index + 1
            break
    return list(items[:keep])


async def rewrite_node(state: RetrievalState, config) -> Dict[str, Any]:
    p = _pipeline(config)
    with trace_span("rag.rewrite", input=state["query"]) as span:
        sub_queries = await p._rewrite(state["query"], n=REWRITE_SUB_QUERIES)
        span.output = sub_queries
    logger.info(f"查询改写: {state['query']!r} → {sub_queries}")
    stages = dict(state.get("stages") or {})
    stages["rewrite"] = len(sub_queries)
    return {"sub_queries": sub_queries, "stages": stages}


async def recall_node(state: RetrievalState, config) -> Dict[str, Any]:
    """混合索引召回：所有子查询并行，每个子查询内部再打稠密 + 稀疏两路索引。"""
    p = _pipeline(config)
    with trace_span("rag.recall", input=state["sub_queries"], recall_k=state["recall_k"]):
        recalls = await asyncio.gather(*[
            p.registry.call(state["tool_name"], {"query": q, "top_k": state["recall_k"]}, state["context"])
            for q in state["sub_queries"]
        ], return_exceptions=True)
    return {"recalls": list(recalls)}


async def rrf_node(state: RetrievalState, config) -> Dict[str, Any]:
    """RRF 粗排：把「子查询 × 索引路」共 2N 份排名融合成候选，顺带完成去重。"""
    cfg = _pipeline(config).retrieval
    lists: List[Tuple[str, List[Any]]] = []
    degraded = False
    failed = 0
    for recall in state["recalls"]:
        if isinstance(recall, ToolResult):
            degraded = degraded or recall.degraded
            if recall.success:
                lists.extend(_rank_lists(recall.data))
            else:
                failed += 1
        elif isinstance(recall, Exception):
            failed += 1
            logger.warning(f"子查询召回异常: {recall}")

    with trace_span("rag.rrf", paths=[name for name, _ in lists]):
        fused = rrf_fuse(lists, cfg)
    stages = dict(state.get("stages") or {})
    stages.update(recall_paths=len(lists), coarse=len(fused))
    if failed:
        stages["recall_failed"] = failed
    logger.debug(
        f"RRF 粗排: {len(lists)} 路 → {len(fused)} 条候选"
        + (f"（{failed} 个子查询失败）" if failed else "")
    )
    return {"coarse": fused, "degraded": degraded, "stages": stages}


def after_rrf(state: RetrievalState) -> str:
    return "rerank" if state["coarse"] else "give_up"


async def give_up_node(state: RetrievalState, config) -> Dict[str, Any]:
    return {
        "result": ToolResult(
            success=False, data=[], tool_name=state["tool_name"],
            error="所有子查询均无召回结果", degraded=state["degraded"],
            stages=state.get("stages") or {},
        )
    }


async def rerank_node(state: RetrievalState, config) -> Dict[str, Any]:
    """Reranker 精排 + 断崖截断：精排失败即整次检索失败，不拿粗排顺序冒充精排结果。"""
    p = _pipeline(config)
    cfg = p.retrieval
    candidates = [doc for doc, _ in state["coarse"]]
    texts = [p.doc_text(doc, cfg.rerank_max_chars) for doc in candidates]

    with trace_span(
        "rag.rerank",
        input={"query": state["query"], "documents": len(candidates)},
        rerank_max_chars=cfg.rerank_max_chars,
    ) as span:
        try:
            # 只带回前 top_k 条：断崖截断本来就只在 top_k 窗口内比较相邻对，多回来的是死重
            scored = await p.rerank(state["query"], texts, top_n=min(state["top_k"], len(candidates)))
        except RerankError as ex:
            logger.error(f"精排失败: {ex}")
            return {
                "result": ToolResult(
                    success=False, data=[], tool_name=state["tool_name"],
                    error=f"精排失败: {ex}", degraded=state["degraded"],
                    stages=state.get("stages") or {},
                )
            }
        span.output = [{"candidate": index, "score": score} for index, score in scored]

    rrf_scores = [score for _, score in state["coarse"]]
    triples = [(candidates[index], rrf_scores[index], score) for index, score in scored]
    with trace_span(
        "rag.truncate",
        input={"query": state["query"], "candidates": len(triples)},
        gap_abs=cfg.gap_abs,
        gap_ratio=cfg.gap_ratio,
        top_k=state["top_k"],
    ) as span:
        kept = cliff_truncate(triples, lambda triple: triple[2], state["top_k"], cfg)
        docs = [p._with_scores(doc, rrf, rerank) for doc, rrf, rerank in kept]
        span.output = docs
        span.attrs.update(returned=len(docs))

    stages = dict(state.get("stages") or {})
    stages.update(reranked=len(triples), returned=len(docs))
    return {
        "result": ToolResult(
            success=True,
            data=docs,
            tool_name=state["tool_name"],
            reranked=True,
            degraded=state["degraded"],
            stages=stages,
        )
    }


def build_retrieval_graph():
    graph = StateGraph(RetrievalState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("recall", recall_node)
    graph.add_node("rrf", rrf_node)
    graph.add_node("give_up", give_up_node)
    graph.add_node("rerank", rerank_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "recall")
    graph.add_edge("recall", "rrf")
    graph.add_conditional_edges("rrf", after_rrf, {"rerank": "rerank", "give_up": "give_up"})
    graph.add_edge("give_up", END)
    graph.add_edge("rerank", END)
    return graph.compile()


RETRIEVAL_GRAPH = build_retrieval_graph()


# ── 检索管线 ──────────────────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    检索类工具的优化链路：问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断。

    召回那一跳仍要回到 ToolRegistry 走外壳（缓存、熔断、统计都挂在那儿），
    所以这里持有 registry 引用而不是自己直连知识库 handler。
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        rerank_client: Optional[AsyncRerankClient] = None,
        retrieval: Optional[RetrievalConfig] = None,
    ):
        self.registry  = registry
        self._llm      = LLMProvider(api_key, base_url)
        self._model    = model
        self._rerank_client = rerank_client
        self.retrieval = retrieval or RetrievalConfig.from_env()

    # ── 检索优化链（G4）────────────────────────────────────────────────────────

    async def _rewrite(self, query: str, n: int = REWRITE_SUB_QUERIES) -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。

        目的：单一查询往往只能召回某一角度的文档，
        多角度子查询并行检索后合并，显著提升召回率。

        示例：
          原始: "退款流程"
          改写: ["如何申请退款", "退款需要多少天", "退款政策是什么"]

        返回值含原始查询、放在第一位，且总条数不超过 n：多出来的子查询会原样变成
        recall 节点多出来的 gather 分支，每支都吃一次工具超时和一次 embedding 配额。
        """
        prompt = f"""
将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]
"""
        prompt = sanitize_text(prompt)
        timeout_s = self.retrieval.rewrite_timeout_s
        try:
            chat = self._llm.chat_model(model=self._model, temperature=0.3, max_tokens=256)
            resp = await asyncio.wait_for(
                chat.ainvoke([HumanMessage(content=prompt)]), timeout=timeout_s
            )
            raw = message_text(resp)
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
            return list(dict.fromkeys([query] + queries))[:max(1, n)]
        except asyncio.TimeoutError:
            degrade(Dep.LLM, "rewrite_timeout", f"查询改写超过 {timeout_s:g}s，使用原始查询")
            return [query]
        except Exception as ex:
            degrade(Dep.LLM, "rewrite_failed", f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search(
        self,
        query: str,
        top_k: int = 5,
        tool_name: str = KNOWLEDGE_SEARCH_TOOL,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        完整检索链路：问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断。

        返回的 data 是最终留下的文档（条数可能少于 top_k —— 断崖在哪切就在哪停），
        stages 里带各级条数，评测时可以直接看是哪一级把答案丢了。

        整条链有总预算：单工具超时只管得住一次 handler 执行，五级串起来的最坏值是
        各跳之和。超预算就判这次检索失败并打降级，而不是让上游一直等到 HTTP 层超时。
        """
        cfg = self.retrieval
        try:
            state = await asyncio.wait_for(
                RETRIEVAL_GRAPH.ainvoke(
                    {
                        "tool_name": tool_name,
                        "query": query,
                        "top_k": top_k,
                        "recall_k": max(top_k, cfg.recall_k),
                        "context": context,
                    },
                    {"configurable": {"pipeline": self}},
                ),
                timeout=cfg.total_timeout_s,
            )
        except asyncio.TimeoutError:
            degrade(
                Dep.TOOL,
                "retrieval_timeout",
                f"检索链路超过总预算 {cfg.total_timeout_s:g}s，未走完五级即终止",
            )
            return ToolResult(
                success=False,
                data=[],
                tool_name=tool_name,
                error=f"检索超时：超过总预算 {cfg.total_timeout_s:g}s",
                degraded=True,
            )
        return state["result"]

    # ── 精排（Reranker）───────────────────────────────────────────────────────

    async def rerank(
        self, query: str, texts: Sequence[str], top_n: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """用交叉编码器给候选打相关性分，返回 [(原始下标, 分数)]，降序。

        没有降级路径：精排客户端缺失或调用失败一律抛 RerankError，
        由 rerank_node 判整次检索失败——用粗排顺序冒充精排结果，评测里看不出差别。
        """
        if self._rerank_client is None:
            raise RerankError("未配置精排服务（RERANK_BASE_URL），检索链路无法精排")
        return await self._rerank_client.rerank(query, texts, top_n=top_n)

    @staticmethod
    def doc_text(doc: Any, max_chars: int) -> str:
        """取送进精排的文档正文：优先正文字段，标题作为上下文前缀。"""
        if not isinstance(doc, dict):
            return sanitize_text(doc)[:max_chars]
        body = next(
            (str(doc[key]) for key in ("content", "text", "snippet", "answer") if doc.get(key)),
            json.dumps(doc, ensure_ascii=False),
        )
        title = str(doc.get("title") or "").strip()
        joined = f"{title}：{body}" if title else body
        return sanitize_text(joined)[:max_chars]

    @staticmethod
    def _with_scores(doc: Any, rrf_score: float, rerank_score: float) -> Any:
        """把粗排分与精排分写回文档。

        `score` 一律以精分为准——下游拼 prompt 和评测读的都是这个字段；
        召回阶段的向量相似度另存 `recall_score`，两路名次另存 `rrf_score`，便于分层归因。
        """
        if not isinstance(doc, dict):
            return doc
        out = dict(doc)
        if "score" in out:
            out["recall_score"] = out["score"]
        out["rrf_score"] = round(rrf_score, 6)
        out["score"] = round(rerank_score, 4)
        return out


# ── Agent 侧 RAG 工具 ─────────────────────────────────────────────────────────

def build_shared_rag_tools(retrieval: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        # top_k 来自 LLM 生成的参数，注入类输入能给出任意大数，检索侧钳住
        top_k = max(1, min(20, int(args.get("top_k", 5) or 5)))
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if retrieval is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await retrieval.search(query, top_k=top_k)
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
                "degraded": bool(getattr(result, "degraded", False)),
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
            "degraded": bool(getattr(result, "degraded", False)),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }
