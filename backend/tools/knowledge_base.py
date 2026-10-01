"""
一、文档导入：将文本切片后写入 Milvus，稠密向量由外部服务产出，词法向量由库内 BM25 function 生成

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
    BM25_ANALYZER_PARAMS,
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
        description="OptiServe RAG 知识库（稠密向量 + BM25 混合索引）",
        fields=[
            FieldSchema("title", DataType.VARCHAR, max_length=512),
            FieldSchema(
                "content", DataType.VARCHAR, max_length=65535,
                enable_analyzer=True, analyzer_params=BM25_ANALYZER_PARAMS,
            ),
            FieldSchema("chunk_index", DataType.INT64),
            FieldSchema("total_chunks", DataType.INT64),
        ],
        bm25=("content", "sparse_vector"),
    ),
)


class KnowledgeBase:
    """基于 Milvus 混合索引的 RAG 知识库：稠密向量来自外部 embedding 服务，词法向量在库内由 BM25 function 生成。"""

    COLLECTION_NAME = "knowledge_base"
    SPARSE_FIELD = "sparse_vector"  # BM25 function 的输出字段，写入时不给值、检索时传原文
    CHUNK_SIZE = 500
    # 单路召回条数上限：RRF 是排名融合，两路各取一份即可，再多只是浪费 ANN 打分
    MAX_RECALL_PER_PATH = 50
    _OUTPUT_FIELDS = ["title", "content", "chunk_index"]

    def __init__(self, vector_config: Optional[VectorStoreConfig] = None):
        self._vector_config = vector_config or VectorStoreConfig.from_env()
        self._embedder = AsyncEmbeddingClient(self._vector_config)
        self._store = MilvusStore(self._vector_config, KNOWLEDGE_COLLECTIONS)

    async def start(self) -> bool:
        """确保 collection 已就绪。失败返回 False，由启动闸门决定去留；库空着不算失败。"""
        return await self._store.ensure_ready()

    async def close(self) -> None:
        await self._store.aclose()
        await self._embedder.aclose()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        长文档会自动切片（每片 500 字），同一份内容重复导入会覆盖而不是堆积。
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

        vectors = await self._embedder.embed_documents(chunks)
        await client.upsert(
            collection_name=self.COLLECTION_NAME,
            data=[
                dict(row, vector=dense)
                for row, dense in zip(rows, vectors)
            ],
        )
        # count(*) 只统计已 seal 的数据，不 flush 的话紧随其后的 doc_count 会返回 0
        await client.flush(collection_name=self.COLLECTION_NAME)
        logger.info(f"知识库导入 {len(rows)} 个文档片段（稠密向量 + 库内 BM25 词法向量）")
        return len(rows)

    async def recall_async(self, query: str, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """异步混合索引召回。"""
        query_text = sanitize_text(query).strip()
        if not query_text or limit <= 0:
            return {"dense": [], "sparse": []}

        client = await self._store.client()
        if client is None:
            raise VectorStoreError("Milvus 不可用，知识库检索失败")

        per_path = min(limit, self.MAX_RECALL_PER_PATH)
        dense_vector = await self._embedder.embed_query(query_text)
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
                # BM25 function 的输出字段吃原文：分词与算分都在库内，这里不再产向量
                data=[query_text],
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

    async def chunk_count(self) -> Optional[int]:
        """片段总数；查不到时返回 None，把"确实为 0"和"没打听到"分开留给调用方判断。"""
        client = await self._store.client()
        if client is None:
            return None

        try:
            rows = await client.query(
                collection_name=self.COLLECTION_NAME,
                filter="",
                output_fields=["count(*)"],
            )
        except Exception as ex:
            logger.warning(f"统计知识库片段数失败: {ex}")
            return None

        if not rows:
            return None
        return int(rows[0].get("count(*)", 0))

    async def doc_count_async(self) -> int:
        """启动统计的保守读法：查不到当 0，统计不该因 Milvus 抖动而失败。"""
        return await self.chunk_count() or 0

    # ── 检索工具 handler 与注册 ───────────────────────────────────────────────

    async def search_handler(
        self, params: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        作为检索工具的 handler 注册：一次调用 = 一个查询的混合索引双路召回。"""
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
        params: Dict[str, Any], error: str
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
#     START → rewrite → recall → rrf → rerank → END

class RetrievalState(TypedDict, total=False):
    tool_name: str
    query: str
    top_k: int
    recall_k: int
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
    """把一次召回的返回值拆成 (索引路, 有序列表)。"""
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
    """Reciprocal Rank Fusion：score(doc) = Σ 1 / (rrf_k + 该路名次)，各路等权。"""
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
    """断崖截断：在 top_k 之内找第一道落差超阈值的相邻对，在那里切断。"""
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

    def _summarize(query: str, recall: Any) -> Dict[str, Any]:
        """output 只记每路命中数和前几条标题：全文会撑爆 trace，定位「哪路召回空」够用了。"""
        entry: Dict[str, Any] = {"query": query, "hits": 0, "top": []}
        if isinstance(recall, Exception):
            entry["error"] = str(recall)
            return entry
        if not (isinstance(recall, ToolResult) and recall.success):
            entry["error"] = recall.error if isinstance(recall, ToolResult) else "unknown"
            return entry
        # 召回工具的 data 是 {"dense": [...], "sparse": [...]} 双路结构
        data = recall.data if isinstance(recall.data, dict) else {}
        docs = [d for path in data.values() if isinstance(path, list) for d in path if isinstance(d, dict)]
        entry["hits"] = len(docs)
        entry["top"] = [str(d.get("title", "")) for d in docs[:3]]
        return entry

    with trace_span("rag.recall", input=state["sub_queries"], recall_k=state["recall_k"]) as span:
        recalls = await asyncio.gather(*[
            p.registry.call(state["tool_name"], {"query": q, "top_k": state["recall_k"]})
            for q in state["sub_queries"]
        ], return_exceptions=True)
        span.output = [
            _summarize(query, recall)
            for query, recall in zip(state["sub_queries"], recalls)
        ]
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


async def rerank_node(state: RetrievalState, config) -> Dict[str, Any]:
    """Reranker 精排 + 断崖截断：精排失败即整次检索失败，不拿粗排顺序冒充精排结果。"""
    p = _pipeline(config)
    cfg = p.retrieval
    candidates = [doc for doc, _ in state["coarse"]]
    if not candidates:
        # 前置门挡掉空库与空查询后，只剩工具未注册、collection 被 release 这类配置态。
        # 必须在这儿判失败：精排客户端对空 documents 直接返回 []，会伪装成"精排成功 0 条"。
        return {
            "result": ToolResult(
                success=False, data=[],
                error="知识库有内容，但本次检索无任何候选（各子查询召回为空或全部失败）",
                degraded=state["degraded"],
                stages=state.get("stages") or {},
            )
        }
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
                    success=False, data=[],
                    error=f"精排失败: {ex}", degraded=state["degraded"],
                    stages=state.get("stages") or {},
                )
            }
        # 带上 id/title：纯下标离开 candidates 列表就没有含义，rerank 前后无法对照
        span.output = [
            {"id": candidates[index].get("id", ""), "title": candidates[index].get("title", ""), "score": score}
            for index, score in scored
        ]

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
    graph.add_node("rerank", rerank_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "recall")
    graph.add_edge("recall", "rrf")
    graph.add_edge("rrf", "rerank")
    graph.add_edge("rerank", END)
    return graph.compile()


RETRIEVAL_GRAPH = build_retrieval_graph()


# ── 检索管线 ──────────────────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    检索类工具的优化链路：问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断。"""

    def __init__(
        self,
        registry: "ToolRegistry",
        api_key: str,
        base_url: Optional[str] = None,
        *,
        model: str,
        kb: KnowledgeBase,
        rerank_client: Optional[AsyncRerankClient] = None,
        retrieval: Optional[RetrievalConfig] = None,
    ):
        self.registry  = registry
        self._kb       = kb
        self._llm      = LLMProvider(api_key, base_url)
        self._model    = model
        self._rerank_client = rerank_client
        self.retrieval = retrieval or RetrievalConfig.from_env()

    # ── 检索优化链（G4）────────────────────────────────────────────────────────

    async def _rewrite(self, query: str, n: int = REWRITE_SUB_QUERIES) -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。"""
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
    ) -> ToolResult:
        """
        完整检索链路：问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断。"""
        cfg = self.retrieval
        # 前置门：空查询或空库直接拒绝，避免浪费 LLM 调用。
        refusal = None
        if not query.strip():
            refusal = "检索内容为空，未执行检索。"
        elif await self._kb.chunk_count() == 0:
            refusal = "知识库当前没有任何文档片段，未执行检索。请告知用户需先导入知识库文档。"
        if refusal:
            logger.info(f"检索前置门拦截: {refusal}")
            return ToolResult(
                success=False, data=[], error=refusal,
                stages={"coarse": 0, "returned": 0},
            )
        try:
            state = await asyncio.wait_for(
                RETRIEVAL_GRAPH.ainvoke(
                    {
                        "tool_name": tool_name,
                        "query": query,
                        "top_k": top_k,
                        "recall_k": max(top_k, cfg.recall_k),
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
                error=f"检索超时：超过总预算 {cfg.total_timeout_s:g}s",
                degraded=True,
            )
        return state["result"]

    # ── 精排（Reranker）───────────────────────────────────────────────────────

    async def rerank(
        self, query: str, texts: Sequence[str], top_n: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """用交叉编码器给候选打相关性分，返回 [(原始下标, 分数)]，降序。"""
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
        """把粗排分与精排分写回文档。"""
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
