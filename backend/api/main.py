"""
OptiServe 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
LLM / embedding / reranker / Redis / Milvus 五类外部依赖在启动时逐个真实探测，
任一不通直接抛错终止启动，不带着残缺依赖对外服务。
"""
import asyncio
import contextvars
import hmac
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, TypeAdapter

from core.degradation import (
    Dep, DepState, collect_degraded, redact_creds, set_status, statuses,
)
from core.tracing import set_finish_hook, start_trace, trace_span
from core.vector_store import AsyncEmbeddingClient, AsyncRerankClient, _env_float, _env_int

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        msg = redact_creds(
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
    if len(dense) != embedder.dim:
        raise RuntimeError(
            f"embedding 维度不匹配：服务返回 {len(dense)}，代码里 EMBEDDING_DIM={embedder.dim}"
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
    return redact_creds(redis_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _kb, _rerank_client, _trace_exporter

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, build_shared_rag_tools
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
        max_body_chars=_env_int("OPTISERVE_SKILL_MAX_BODY_CHARS", 6000),
        max_index_chars=_env_int("OPTISERVE_SKILL_INDEX_MAX_CHARS", 1500),
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
        return f"{redact_creds(vector_cfg.milvus_uri)}，知识库 {await _kb.doc_count_async()} 个片段"

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
    _orchestrator.set_shared_tools(build_shared_rag_tools(_tool_manager))

    # 性能监控
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=_env_float("MONITOR_INTERVAL", 10.0),
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
    # 树随请求结束一起释放，启动与请求链路都不受影响。
    from core.trace_export import create_exporter

    _trace_exporter = create_exporter()
    set_finish_hook(_trace_exporter.export if _trace_exporter else None)
    logger.info(
        "Langfuse 上报已开启" if _trace_exporter else "Langfuse 未配置，本次运行不留任何请求链路记录"
    )

    logger.info("OptiServe 已就绪")
    if not _api_token():
        logger.warning(
            "OPTISERVE_API_TOKEN 未设置：/chat、/search 等只读接口对本机全部放行"
            "（API_HOST 默认回环），写接口已被 require_configured_token 禁用。对外暴露前务必配置。"
        )

    # 启动 FastAPI 服务
    yield

    if _profile_tasks:
        # 画像更新不阻塞响应，但关服务前给它一个收尾窗口，否则最后几条请求的画像会丢
        _, pending = await asyncio.wait(_profile_tasks, timeout=PROFILE_DRAIN_TIMEOUT_S)
        for task in pending:
            task.cancel()
        if pending:
            # cancel 只是投递取消，必须等任务真正解栈再关 Redis/Milvus 客户端，
            # 否则任务会在已关闭的客户端上抛错
            await asyncio.gather(*pending, return_exceptions=True)
            logger.info(f"后台画像更新收尾超时，已取消 {len(pending)} 个")

    await _monitor.stop()
    if _trace_exporter is not None:
        _trace_exporter.shutdown()
    await recognizer.close()
    await _memory.close()
    await _kb.close()
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


def require_configured_token():
    """写路由的兜底闸门：middleware 在 token 未配置时是放行的（本地 demo 需要），
    改状态的路由不能跟着一起裸奔——没配 token 就拒绝写。"""
    if not _api_token():
        raise HTTPException(status_code=503, detail="服务未配置 OPTISERVE_API_TOKEN，写操作已禁用")


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


@app.post("/skills/reload", tags=["Skills"], dependencies=[Depends(require_configured_token)])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


# 画像更新是后台任务，但没存引用的 task 可能被 GC 掉、异常也会静默消失，
# 所以统一挂进这个集合，shutdown 时再等未完成的收尾。
_profile_tasks: set = set()
# 同一 user 同时只跑一个画像任务：否则并发请求各自「读旧画像→LLM 提炼→upsert」，
# 后写覆盖先写，LLM / embedding 调用还会随请求数无界放大。
_profile_running_users: set = set()
_profile_slots = asyncio.Semaphore(4)
# 收尾窗口要盖得住画像链路最坏耗时（LLM 提炼 + embedding 预算 + upsert），5s 必超时
PROFILE_DRAIN_TIMEOUT_S = _env_float("OPTISERVE_PROFILE_DRAIN_TIMEOUT_S", 15.0)


async def _run_profile_update(user_id: str, conv_id: str) -> None:
    async with _profile_slots:
        await _memory.update_profile(user_id, conv_id)


def _spawn_profile_update(user_id: str, conv_id: str) -> None:
    if user_id in _profile_running_users:
        return
    _profile_running_users.add(user_id)
    # 全新空 Context：不继承本请求的降级事件列表和 span 树，
    # 否则后台失败会事后追加进已返回请求的 degradations，降级计数与响应体对不上。
    task = asyncio.create_task(
        _run_profile_update(user_id, conv_id),
        context=contextvars.Context(),
    )
    _profile_tasks.add(task)
    task.add_done_callback(lambda t: _forget_profile_task(t, user_id))


def _forget_profile_task(task: asyncio.Task, user_id: str) -> None:
    _profile_tasks.discard(task)
    _profile_running_users.discard(user_id)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.warning(f"后台画像更新失败: {error}")


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

    # trace_id 就是响应里的 request_id：同一次请求在日志和 Langfuse 里是同一个键
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

            # 5. 用户画像异步更新：挪出 with 块后再 spawn，见函数末尾

            tr.meta.update(
                degradations=[event.as_dict() for event in degraded],
                agent_type=result.agent_type.value,
                primary_agent=result.primary_agent.value if result.primary_agent else "",
                routing_reason=result.routing_reason,
                tools_used=result.tools_used,
                escalated=result.escalated,
            )

            response = ChatResponse(
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

    # 出了 collect_degraded / start_trace 的上下文再 spawn：后台失败只进日志，
    # 不会事后回写这条已返回请求的降级列表和 span 树。
    _spawn_profile_update(req.user_id, conv_id)
    return response


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.post("/search")
async def search(
    # 参数级钳制：这是给终端用户/前端直连的口子，top_k 不设上限能被拿来做无界检索
    query: str = Query(..., min_length=1, max_length=2000),
    top_k: int = Query(5, ge=1, le=50),
):
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
    # 每条用例都会打一次真实 LLM 链路，条数不封顶=单请求可放大成任意次调用
    intent_cases: Optional[List[EvalIntentInput]] = Field(default=None, max_length=200)
    dialog_cases: Optional[List[EvalDialogInput]] = Field(default=None, max_length=200)


@app.post("/knowledge/add", tags=["知识库"], dependencies=[Depends(require_configured_token)])
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


@app.post("/knowledge/upload", tags=["知识库"], dependencies=[Depends(require_configured_token)])
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

    content = b""
    # 分块读：先 file.read() 再判大小，等于让 413 校验形同虚设——超大文件已整个进内存
    while chunk := await file.read(1024 * 1024):
        content += chunk
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(413, "文件大小超过 10MB 限制")
    if not content:
        raise HTTPException(400, "文件为空")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
        try:
            # 元素结构也过一遍 DocInput：只判 isinstance(list) 时，非 dict 元素
            # 会一路钻到 knowledge_base 里抛 AttributeError 变 500
            docs = [d.model_dump() for d in TypeAdapter(List[DocInput]).validate_python(docs)]
        except Exception as e:
            raise HTTPException(400, f"JSON 内容应为 [{{title, content}}, ...] 数组: {e}")
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


@app.post("/eval/run", dependencies=[Depends(require_configured_token)])
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
