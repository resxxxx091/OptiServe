"""
OptiServe 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
LLM / embedding / reranker / Redis / Milvus 五类外部依赖在启动时逐个真实探测，
任一不通直接抛错终止启动，不带着残缺依赖对外服务。
"""
import asyncio
import hmac
import logging
import os
import pathlib
import re
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field

from core.degradation import Dep, DepState, collect_degraded, set_status, statuses
from core.tracing import get_trace_tree, recent_trace_trees, set_finish_hook, start_trace, trace_span
from core.vector_store import AsyncEmbeddingClient, AsyncRerankClient

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# URL 里的凭据段：scheme://user:pass@host → scheme://host
_CREDS_RE = re.compile(r"://[^/@]*@")
# 光洗 URL 形态不够：异常文本常把密码单独复述一遍（"…(real=xxx)"），只能按值来洗
_SECRET_ENV_KEYS = (
    "DEEPSEEK_API_KEY", "EMBEDDING_API_KEY", "RERANK_API_KEY",
    "REDIS_PASSWORD", "MILVUS_TOKEN", "OPTISERVE_API_TOKEN",
)


def _redact_creds(text: str) -> str:
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


def _api_token() -> str:
    """访问令牌。用函数而非模块常量，避免在 import 期就把值冻住。"""
    return os.getenv("OPTISERVE_API_TOKEN", "")

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   OptiServe  v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_kb           = None
_rerank_client = None
_trace_exporter = None

def _llm_cfg() -> Dict[str, Any]:
    key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    if not key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")
    if not model:
        raise RuntimeError('未设置 DEEPSEEK_MODEL')
    if not base_url:
        raise RuntimeError('未设置 DEEPSEEK_BASE_URL')
    cfg: Dict[str, Any] = {
        "api_key": key,
        "model": model,
        'base_url': base_url
    }
    return cfg


# ── 启动闸门：外部依赖逐个真实探测，任一不通就不启动 ──────────────────────────

async def _gate(dep: Dep, probe: Awaitable[str]) -> None:
    """执行一个探测协程；成功记下 ok，失败记 unavailable 并抛出终止启动。"""
    try:
        detail = await probe
    except Exception as ex:
        # 只取异常首行 + 限长：响应体里可能带回密钥片段，不适合进 /health
        msg = _redact_creds(
            f"{type(ex).__name__}: {(str(ex).splitlines() or [''])[0][:160]}"
        )
        set_status(dep, DepState.UNAVAILABLE, msg)
        raise RuntimeError(f"{dep.value} 依赖不可用，服务拒绝启动（{msg}）") from ex
    set_status(dep, DepState.OK, detail)
    logger.info(f"启动探测通过 [{dep.value}] {detail}")


async def _probe_llm(cfg: Dict[str, Any]) -> str:
    """向 LLM 端点真发一次最小请求，密钥/网络/模型名任一有问题都会在这里暴露。"""
    from langchain_core.messages import HumanMessage

    from core.llm import LLMProvider

    chat = LLMProvider(cfg["api_key"], cfg["base_url"]).chat_model(model=cfg["model"], max_tokens=1)
    try:
        await chat.ainvoke([HumanMessage(content="ping")])
    finally:
        await chat.root_async_client.close()
    return f"{cfg['model']} @ {cfg['base_url']}"


async def _probe_embedding(vector_cfg: Any) -> str:
    """真取一次稠密 + 稀疏两路向量，顺带校验维度与 Milvus collection 的建库维度一致。

    知识库的混合索引要求 embedding 服务同时给得出稀疏那一路，只回稠密向量在这里就判不通过。
    """
    embedder = AsyncEmbeddingClient(vector_cfg)
    try:
        dense, sparse = await embedder.embed_query_hybrid("ping")
    finally:
        await embedder.aclose()
    if len(dense) != vector_cfg.embedding_dim:
        raise RuntimeError(
            f"embedding 维度不匹配：服务返回 {len(dense)}，EMBEDDING_DIM={vector_cfg.embedding_dim}"
        )
    return f"{vector_cfg.embedding_model} dim={len(dense)}，稀疏向量 {len(sparse)} 项"


async def _probe_reranker(vector_cfg: Any) -> str:
    """真发一次精排请求：检索链路的最后一级没有降级路径，端点不通就不该启动。"""
    client = AsyncRerankClient(vector_cfg)
    try:
        scored = await client.rerank("ping", ["相关的一篇", "无关的一篇"])
    finally:
        await client.aclose()
    if not scored:
        raise RuntimeError("rerank 服务返回空结果")
    return f"{vector_cfg.rerank_model} @ {vector_cfg.rerank_base_url}"


async def _probe_redis(redis_url: str) -> str:
    import redis.asyncio as aioredis

    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await client.ping()
    finally:
        await client.aclose()
    return _redact_creds(redis_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _kb, _rerank_client, _trace_exporter

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, build_shared_rag_tools
    from agents.base import _env_int
    from core.intent_recognizer import IntentRecognizer
    from core.vector_store import VectorStoreConfig
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager

    cfg = _llm_cfg()
    logger.info(f"model: {cfg['model']}  base_url: {cfg['base_url']}")

    # 向量层配置（Milvus + 外部 embedding），意图识别、记忆与知识库共用一套
    vector_cfg = VectorStoreConfig.from_env()
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # 依赖不通就让服务直接起不来，而不是运行期逐请求降级
    await _gate(Dep.LLM,       _probe_llm(cfg))
    await _gate(Dep.EMBEDDING, _probe_embedding(vector_cfg))
    await _gate(Dep.RERANKER,  _probe_reranker(vector_cfg))
    await _gate(Dep.REDIS,     _probe_redis(redis_url))

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        vector_config=vector_cfg,
    )

    # Skills：启动时从目录加载，索引（name + description）常驻 system prompt，
    # 正文由 Agent 判断后调用 load_skill 按需取回。
    skills_dir = os.getenv("OPTISERVE_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_body_chars=int(os.getenv("OPTISERVE_SKILL_MAX_BODY_CHARS", "6000")),
        max_index_chars=int(os.getenv("OPTISERVE_SKILL_INDEX_MAX_CHARS", "1500")),
    )
    _skill_manager.load()

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + Milvus 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=redis_url,
        vector_config=vector_cfg,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（Milvus 稠密 + 稀疏混合索引，精排走外部 reranker）
    _rerank_client = AsyncRerankClient(vector_cfg)
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        rerank_client=_rerank_client,
    )
    _kb = KnowledgeBase(vector_config=vector_cfg)

    async def probe_milvus() -> str:
        """建库/校验两侧 collection，任一没就绪说明 Milvus 不可用。"""
        if not await _memory.start():
            raise RuntimeError("记忆的向量层未就绪")
        if not await _kb.start():
            raise RuntimeError("知识库的向量层未就绪")
        return f"{_redact_creds(vector_cfg.milvus_uri)}，知识库 {await _kb.doc_count_async()} 个片段"

    await _gate(Dep.MILVUS, probe_milvus())

    def knowledge_fallback(
        params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str
    ):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的混合索引检索。请稍后重试，或转人工客服确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        # 单次调用 = 一个查询的混合索引双路召回；融合与精排在检索图里，不在这里
        description="搜索 RAG 知识库（Milvus 稠密 + 稀疏混合索引召回）",
        handler=_kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "description": "每路索引各取多少条候选"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        fallback=knowledge_fallback,
    ))
    if _orchestrator is not None:
        _orchestrator.set_shared_tools(build_shared_rag_tools(_tool_manager))

    # 性能监控
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        alert_max=_env_int("OPTISERVE_MONITOR_ALERT_MAX", 200),
        suggestion_max=_env_int("OPTISERVE_MONITOR_SUGGESTION_MAX", 50),
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    # 可观测性：每次请求的 span 树收尾时导出。没装 SDK / 没配密钥 → 钩子为 None，
    # 树仍然只留在本地 /trace，启动与请求链路都不受影响。
    from core.trace_export import create_exporter

    _trace_exporter = create_exporter()
    set_finish_hook(_trace_exporter.export if _trace_exporter else None)
    logger.info(
        "Langfuse 上报已开启" if _trace_exporter else "Langfuse 未配置，span 树仅本地可查"
    )

    logger.info("OptiServe 已就绪")

    # 启动 FastAPI 服务
    yield

    await _monitor.stop()
    if _trace_exporter is not None:
        _trace_exporter.shutdown()
    await recognizer.close()
    if _memory is not None:
        await _memory.close()
    if _kb is not None:
        await _kb.close()
    if _rerank_client is not None:
        await _rerank_client.aclose()
    logger.info("OptiServe 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
# HTTPBearer(auto_error=False) 只为在 openapi 里写出 securitySchemes，让 Swagger 的
# Authorize 输入框可用；强制鉴权只在下述 middleware 一处（auto_error=True 会让缺 header
# 的请求在依赖层先吃 403，与 middleware 的 401 撞成两套错误码）。
app = FastAPI(
    title="OptiServe 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    dependencies=[Depends(HTTPBearer(auto_error=False))],
)

# 文档三件套本身不含凭据，豁免换取"能在 /docs 里填 token 调接口"；
# 代价是接口契约对该端口可达者公开，而 API_HOST 默认回环监听已把范围收到本机。
_PUBLIC_PATHS = {"/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def require_token(request, call_next):
    token = _api_token()
    if not token or request.method == "OPTIONS" or request.url.path in _PUBLIC_PATHS:
        return await call_next(request)
    supplied = request.headers.get("authorization", "").encode()
    if not hmac.compare_digest(supplied, f"Bearer {token}".encode()):
        return JSONResponse(status_code=401, content={"detail": "缺少或无效的访问令牌"})
    return await call_next(request)


# CORS 放在鉴权 middleware 之后添加：后加入的在外层，这样 401 响应也带得上
# Access-Control-Allow-*，跨源前端才读得到错误体而不是报成网络错误。
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("OPTISERVE_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    request_id:  str = ""
    response:    str
    intent:      str
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    degraded: bool = False
    degradations: List[Dict[str, str]] = Field(default_factory=list)


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


class TraceTreeResponse(BaseModel):
    """一次请求的 span 树：编排 → Agent → 工具 → RAG 检索各阶段的父子耗时。"""
    trace_id: str
    found: bool
    tree: Dict[str, Any] = Field(default_factory=dict)


class RecentTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {
        "status": "ok",
        "dependencies": statuses(),
        "agents": _orchestrator.get_stats(),
    }


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    # trace_id 就是响应里的 request_id：同一次请求在 /trace、日志和 Langfuse 里是同一个键
    request_id = str(uuid.uuid4())[:8]

    # 一次请求一条 span 树。记忆读取与意图识别在 orchestrator.run() 外面，
    # 所以树必须从 API 层起，只包 run() 会漏掉这两处。
    with start_trace(request_id, "chat") as tr:
        # 包住整条链路：记忆读取和意图识别都在 orchestrator.run() 外面，
        # 只包 run() 会漏掉这两处的降级。
        with collect_degraded() as degraded:
            conv_id = req.conv_id or str(uuid.uuid4())

            # 1. 读取记忆上下文
            with trace_span("memory_read", user_id=req.user_id, conv_id=conv_id):
                mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

            # 2. 构建编排请求（含对话历史，用于意图识别上下文）
            history = [
                {"role": m.role.value, "content": m.content}
                for m in mem_ctx.recent_messages[-5:]
            ] if mem_ctx.recent_messages else None

            with trace_span("intent_recognition", source="api"):
                intent_result = await _orchestrator.recognize_intent(req.message, history=history)
            full_context = mem_ctx.to_prompt_text()

            orch_req = OrcReq(
                message=req.message,
                user_id=req.user_id,
                conv_id=conv_id,
                context=full_context,
                history=history,
                entities=intent_result.entities,
                intent=intent_result.intent,
                intent_confidence=intent_result.confidence,
                request_id=request_id,
            )

            # 3. 执行
            result = await _orchestrator.run(orch_req)

            # 4. 写入记忆
            with trace_span("memory_write", user_id=req.user_id, conv_id=conv_id):
                await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

            # 5. 异步更新用户画像（不阻塞响应）
            asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

            tr.meta.update(
                degradations=[event.as_dict() for event in degraded],
                agent_type=result.agent_type.value,
                primary_agent=result.primary_agent.value if result.primary_agent else "",
                routing_reason=result.routing_reason,
                tools_used=result.tools_used,
                escalated=result.escalated,
            )

            return ChatResponse(
                conv_id=conv_id,
                request_id=result.request_id,
                response=result.response,
                intent=result.intent.value if result.intent else "other",
                agent_type=result.agent_type.value,
                agent_types=[agent_type.value for agent_type in result.agent_types],
                primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
                supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
                tools_used=result.tools_used,
                routing_reason=result.routing_reason,
                routing_confidence=result.routing_confidence,
                escalated=result.escalated,
                latency_ms=round(result.latency_ms, 1),
                knowledge_used="search_knowledge_base" in result.tools_used,
                entities=intent_result.entities,
                intent_confidence=round(intent_result.confidence, 4),
                intent_source_scores=intent_result.source_scores,
                degraded=bool(degraded),
                degradations=[event.as_dict() for event in degraded],
            )


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    """查看某次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    """查看最近 N 次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_tool_traces(limit=limit))


@app.get("/trace/recent", response_model=RecentTracesResponse)
async def list_recent_traces(limit: int = 20):
    """最近 N 条 trace 的摘要（trace_id、片段数、总耗时、结果），新的在前。"""
    return RecentTracesResponse(items=recent_trace_trees(limit=limit))


@app.get("/trace/{trace_id}", response_model=TraceTreeResponse)
async def get_trace(trace_id: str):
    """一次请求的完整 span 树。/trace/tool/* 看工具明细，这里看谁把时间花在哪一层。"""
    tree = get_trace_tree(trace_id)
    return TraceTreeResponse(trace_id=trace_id, found=tree is not None, tree=tree or {})


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    演示检索链路：问题改写 → 混合索引召回 → RRF 粗排 → Reranker 精排 → 断崖截断。

    `stages` 给出各级条数，用来定位"答案是在哪一级丢掉的"。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    # 检索链路自己成一条 trace：五级漏斗的各级条数落在 rag.* 片段上
    with start_trace(str(uuid.uuid4())[:8], "search"):
        result = await _tool_manager.search_pipeline("knowledge_search", query, top_k=top_k)
    return {
        "query": query,
        "results": result.data,
        "reranked": result.reranked,
        "degraded": result.degraded,
        "stages": result.stages,
        "error": result.error,
    }


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字），向量由外部 embedding 服务（EMBEDDING_MODEL）生成后写入 Milvus。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    if _kb is None:
        raise HTTPException(503, "知识库未初始化")
    from core.vector_store import VectorStoreError

    try:
        count = await _kb.add_documents_async([{"title": d.title, "content": d.content} for d in body.documents])
    except VectorStoreError as ex:
        raise HTTPException(503, f"知识库写入失败: {ex}")
    total = await _kb.doc_count_async()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    if _kb is None:
        raise HTTPException(503, "知识库未初始化")

    from core.vector_store import VectorStoreError

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    try:
        count = await _kb.add_documents_async(docs)
    except VectorStoreError as ex:
        raise HTTPException(503, f"知识库写入失败: {ex}")
    total = await _kb.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    if _kb is None:
        raise HTTPException(503, "知识库未初始化")
    return {"total_chunks": await _kb.doc_count_async()}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("APP_ENV") == "development",
    )
