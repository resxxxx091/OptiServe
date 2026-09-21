"""
亮点：MCP 工具调用框架 + RAG 检索优化链路

本模块管两件事。

一、工具调用的可靠性外壳（所有工具共用）：
  1. 参数校验（JSON Schema 的 required 与顶层类型）
  2. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用
  3. 超时控制（asyncio.wait_for，按工具配 timeout_s）
  4. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果，同时打 `degraded` 标记、
     单独计 `fallback_rate`，不让兜底文案伪装成健康调用

二、检索类工具的优化链路 RETRIEVAL_GRAPH，五级串联：
  问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断

  1. 问题改写：LLM 把用户原始问题扩写成多个角度的子查询，解决"一个问法只召回一个角度"。
  2. 混合索引召回：每个子查询同时打稠密向量（语义）与稀疏向量（关键词命中）两路 ANN。
  3. RRF 粗排：对「子查询 × 索引路」共 2N 份排名做 Reciprocal Rank Fusion
     （score = Σ weight/(k+rank)），把多路共识顶上来，顺带去重。
     融合只用名次、不用分数：稠密路的 COSINE 和稀疏路的 IP 不同量纲，跨路只有排名可比。
  4. Reranker 精排：交叉编码器（外部 /v1/rerank）对粗排候选逐条 (query, doc) 打分，
     它看得到 query 与文档的交互，比向量相似度更接近"对这个问题有用"。
  5. 断崖截断：精排分数相邻两位的落差超阈值就在那里切断，而不是固定取 top-K ——
     相关文档和不相关文档之间通常有个明显的分数断层，硬切会把噪声带进 prompt。

精排不做静默降级：rerank 服务不通在启动闸门就拒绝启动，运行期调用失败则整次检索判失败，
不用召回顺序冒充精排结果。
"""
import asyncio
import hashlib
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, TypeVar

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from core.degradation import Dep, degrade
from core.llm import LLMProvider, message_text
from core.tracing import trace_span
from core.vector_store import AsyncRerankClient, RerankError

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # data 是否经过 Reranker 精排
    degraded:       bool = False   # data 来自 fallback，不是真实 handler 的输出
    stages:         Dict[str, int] = field(default_factory=dict)  # 检索链路各级的条数


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。

    口径：total = 每次进入调用的请求（含熔断拒绝，不含"工具不存在"）；
    success = 调用方拿到了可用结果（真实成功 + 降级救回）；
    failed  = 调用方没拿到结果。所以 success + failed == total。
    fallback 是 success 的子集，单独记才看得出"成功率 100% 但全靠兜底"。
    延迟均值只按 latency_samples 算，避免拒绝和缓存命中把 avg_ms 拉低。
    """
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    fallback:           int = 0
    total_latency_ms:   float = 0.0
    latency_samples:    int = 0     # 真正跑到 handler 的次数，延迟均值只按它算
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def fallback_rate(self) -> float:
        return self.fallback / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.latency_samples if self.latency_samples else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED
                                         → OPEN
    规则：
    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  float = 0.0

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # 如果熔断器处于打开状态，检查是否已经过了恢复时间
            if time.monotonic() - self.opened_at >= self.recovery_s:  
                # 进入 HALF_OPEN 状态，允许一次探测
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── 检索链路参数 ──────────────────────────────────────────────────────────────

T = TypeVar("T")


@dataclass(frozen=True)
class RetrievalConfig:
    """RRF 融合与断崖截断的阈值；全部走环境变量，评测调参不用改代码。"""

    rrf_k: int = 60                # RRF 平滑常数：压住名次靠后时的边际差异
    dense_weight: float = 0.7      # 稠密路权重，语义匹配是主力
    sparse_weight: float = 0.3     # 稀疏路权重，错误码/型号这类精确词靠它兜住
    recall_k: int = 10             # 每个子查询在每路索引上取的候选条数（召回宽度）
    coarse_n: int = 20             # 粗排后送进精排的候选数
    max_topk: int = 10             # 断崖截断的绝对上限，别让一次检索吐出几十篇
    min_topk: int = 1              # 至少留一篇，哪怕第一名就跟第二名断层
    gap_abs: float = 0.5           # 相邻分数绝对落差阈值
    gap_ratio: float = 0.25        # 相邻分数相对落差阈值（相对前一名）
    rerank_max_chars: int = 1000   # 送进精排的单篇文档截断长度

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            rrf_k=int(os.getenv("RETRIEVAL_RRF_K", "60")),
            dense_weight=float(os.getenv("RETRIEVAL_RRF_DENSE_WEIGHT", "0.7")),
            sparse_weight=float(os.getenv("RETRIEVAL_RRF_SPARSE_WEIGHT", "0.3")),
            recall_k=int(os.getenv("RETRIEVAL_RECALL_K", "10")),
            coarse_n=int(os.getenv("RETRIEVAL_COARSE_N", "20")),
            max_topk=int(os.getenv("RERANK_MAX_TOPK", "10")),
            min_topk=int(os.getenv("RERANK_MIN_TOPK", "1")),
            gap_abs=float(os.getenv("RERANK_GAP_ABS", "0.5")),
            gap_ratio=float(os.getenv("RERANK_GAP_RATIO", "0.25")),
            rerank_max_chars=int(os.getenv("RERANK_MAX_CHARS", "1000")),
        )

    @property
    def path_weights(self) -> Dict[str, float]:
        return {"dense": self.dense_weight, "sparse": self.sparse_weight}


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


def _manager(config) -> "MCPToolManager":
    return config["configurable"]["manager"]


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
    """加权 Reciprocal Rank Fusion：score(doc) = Σ 路权重 / (rrf_k + 该路名次)。

    只用名次不用分数，所以稠密的 COSINE 与稀疏的 IP 可以同台融合；
    同一篇文档被越多路命中、且名次越靠前，累积分越高。
    """
    weights = cfg.path_weights
    scores: Dict[str, float] = {}
    docs: Dict[str, Any] = {}
    for path, items in lists:
        weight = weights.get(path, 1.0)
        for rank, item in enumerate(items, start=1):
            key = _doc_key(item)
            scores[key] = scores.get(key, 0.0) + weight / (cfg.rrf_k + rank)
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
    """断崖截断：在 min(top_k, max_topk) 之内，相邻两名分数落差超阈值就从那里切断。

    两个阈值同时存在是因为量纲与分布随模型而异：绝对落差管"整体都低分"，
    相对落差管"前几名挤在一起、后面突然塌"。命中即停，只看第一道断崖。
    """
    if not items:
        return []
    keep = min(len(items), max(cfg.min_topk, top_k), cfg.max_topk)
    for index in range(cfg.min_topk - 1, keep - 1):
        head, tail = score_of(items[index]), score_of(items[index + 1])
        gap = head - tail
        if gap >= cfg.gap_abs or gap >= cfg.gap_ratio * abs(head):
            keep = index + 1
            break
    return list(items[:keep])


async def rewrite_node(state: RetrievalState, config) -> Dict[str, Any]:
    manager = _manager(config)
    with trace_span("rag.rewrite", query=state["query"]):
        sub_queries = await manager.rewrite_query(state["query"], n=3)
    logger.info(f"查询改写: {state['query']!r} → {sub_queries}")
    stages = dict(state.get("stages") or {})
    stages["rewrite"] = len(sub_queries)
    return {"sub_queries": sub_queries, "stages": stages}


async def recall_node(state: RetrievalState, config) -> Dict[str, Any]:
    """混合索引召回：所有子查询并行，每个子查询内部再打稠密 + 稀疏两路索引。"""
    manager = _manager(config)
    with trace_span("rag.recall", sub_queries=len(state["sub_queries"]), recall_k=state["recall_k"]):
        recalls = await asyncio.gather(*[
            manager.call(state["tool_name"], {"query": q, "top_k": state["recall_k"]}, state["context"], use_cache=True)
            for q in state["sub_queries"]
        ], return_exceptions=True)
    return {"recalls": list(recalls)}


async def rrf_node(state: RetrievalState, config) -> Dict[str, Any]:
    """RRF 粗排：把「子查询 × 索引路」共 2N 份排名融合成候选，顺带完成去重。"""
    cfg = _manager(config).retrieval
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

    with trace_span("rag.rrf", paths=[name for name, _ in lists], dense_weight=cfg.dense_weight, sparse_weight=cfg.sparse_weight):
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
    manager = _manager(config)
    cfg = manager.retrieval
    candidates = [doc for doc, _ in state["coarse"]]
    texts = [manager.doc_text(doc, cfg.rerank_max_chars) for doc in candidates]

    with trace_span("rag.rerank", candidates=len(candidates), rerank_max_chars=cfg.rerank_max_chars):
        try:
            scored = await manager.rerank(state["query"], texts, top_n=len(candidates))
        except RerankError as ex:
            logger.error(f"精排失败: {ex}")
            return {
                "result": ToolResult(
                    success=False, data=[], tool_name=state["tool_name"],
                    error=f"精排失败: {ex}", degraded=state["degraded"],
                    stages=state.get("stages") or {},
                )
            }

    rrf_scores = [score for _, score in state["coarse"]]
    triples = [(candidates[index], rrf_scores[index], score) for index, score in scored]
    with trace_span(
        "rag.truncate",
        gap_abs=cfg.gap_abs,
        gap_ratio=cfg.gap_ratio,
        min_topk=cfg.min_topk,
        max_topk=cfg.max_topk,
        top_k=state["top_k"],
    ) as span:
        kept = cliff_truncate(triples, lambda triple: triple[2], state["top_k"], cfg)
    if span is not None:
        span.attrs.update(returned=len(kept))

    stages = dict(state.get("stages") or {})
    stages.update(reranked=len(triples), returned=len(kept))
    return {
        "result": ToolResult(
            success=True,
            data=[manager._with_scores(doc, rrf, rerank) for doc, rrf, rerank in kept],
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


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    MCP 工具调用框架。

    检索类工具的完整链路（RETRIEVAL_GRAPH）五级串联：
      问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        rerank_client: Optional[AsyncRerankClient] = None,
        retrieval: Optional[RetrievalConfig] = None,
    ):
        self._llm    = LLMProvider(api_key, base_url)
        self._model  = model
        self._rerank_client = rerank_client
        self.retrieval = retrieval or RetrievalConfig.from_env()
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, Tuple[Any, float]] = {}   # key → (result, expire_at)

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        logger.info(f"注销工具: {name}")

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 缓存写入
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        tool.stats.total += 1

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params)
            if cached is not None:
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached,
                    tool_name=name,
                    cached=True,
                )

        # 熔断检查
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        try:
            # 参数校验（根据 JSON Schema 的 required 和 properties.type）
            self._validate_params(tool, params)

            data = await asyncio.wait_for(self._run_handler(tool, params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.stats.latency_samples += 1
            tool.breaker.record_success()

            # 缓存的是双路召回的原始结果；RRF 粗排、精排与断崖截断在检索图里做，
            # 在缓存之外，因此缓存命中时 reranked 必然为 False。
            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl)

            return ToolResult(success=True, data=data, tool_name=name, latency_ms=latency)

        except asyncio.TimeoutError:
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """
        工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。

        降级算"调用方拿到了结果"（计入 success），但同时记一次 fallback 并打上
        degraded 标记——否则依赖全挂时成功率仍是 100%，监控看不出问题。
        """
        if tool.fallback is None:
            tool.stats.failed += 1
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            tool.stats.success += 1
            tool.stats.fallback += 1
            degrade(Dep.TOOL, "fallback", f"{tool.name} 由降级兜底返回：{error}")
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
                degraded=True,
            )
        except Exception as ex:
            tool.stats.failed += 1
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """
        执行工具 handler。

        优先支持 async handler；如果历史工具仍是同步函数，则放入线程池执行，
        避免阻塞事件循环。
        """
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params, context)
        result = await asyncio.to_thread(tool.handler, params, context)
        # 如果 handler 返回的是 awaitable 对象（例如 asyncio.Future），则继续 await
        if inspect.isawaitable(result):
            return await result
        return result

    # ── 检索优化链（G4）────────────────────────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。

        目的：单一查询往往只能召回某一角度的文档，
        多角度子查询并行检索后合并，显著提升召回率。

        示例：
          原始: "退款流程"
          改写: ["如何申请退款", "退款需要多少天", "退款政策是什么"]
        """
        prompt = f"""
将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]
"""
        prompt = self._clean_text(prompt)
        try:
            chat = self._llm.chat_model(model=self._model, temperature=0.3, max_tokens=256)
            resp = await chat.ainvoke([HumanMessage(content=prompt)])
            raw = message_text(resp)
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
            # 原始查询也保留，去重
            return list(dict.fromkeys([query] + queries))
        except Exception as ex:
            degrade(Dep.LLM, "rewrite_failed", f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_pipeline(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        完整检索链路：问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断。

        返回的 data 是最终留下的文档（条数可能少于 top_k —— 断崖在哪切就在哪停），
        stages 里带各级条数，评测时可以直接看是哪一级把答案丢了。
        """
        state = await RETRIEVAL_GRAPH.ainvoke(
            {
                "tool_name": tool_name,
                "query": query,
                "top_k": top_k,
                "recall_k": max(top_k, self.retrieval.recall_k),
                "context": context,
            },
            {"configurable": {"manager": self}},
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
            return MCPToolManager._clean_text(doc)[:max_chars]
        body = next(
            (str(doc[key]) for key in ("content", "text", "snippet", "answer") if doc.get(key)),
            json.dumps(doc, ensure_ascii=False),
        )
        title = str(doc.get("title") or "").strip()
        joined = f"{title}：{body}" if title else body
        return MCPToolManager._clean_text(joined)[:max_chars]

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

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict) -> str:
        payload = json.dumps(params, sort_keys=True)
        return f"{name}:{hashlib.md5(payload.encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict) -> Optional[Any]:
        key = self._cache_key(name, params)
        if key in self._cache:
            data, expire_at = self._cache[key]
            if time.monotonic() < expire_at:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, name: str, params: Dict, data: Any, ttl: float) -> None:
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params)] = (data, time.monotonic() + ttl)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "fallback": t.stats.fallback,
                "fallback_rate": round(t.stats.fallback_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "latency_samples": t.stats.latency_samples,
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
