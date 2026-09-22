"""编排层的两张 LangGraph 图（G1 编排图 + G2 工具子图）。

G2 —— 一次请求在一个 Agent 内部的控制流：

    START → render_prompt → call_model → (tools → call_model)* → END
                                       ↘ exhausted → END

G1 —— 一次请求在编排层的控制流：

    START → recognize_intent → guard →(命中直返) END
                                ↘ route →(多 Agent) parallel → END
                                               ↘ single → END

    parallel 即 PARALLEL_GRAPH：prepare →(Send 扇出) run_agents → join → END

两张图都不持有编排器实例：orchestrator 通过 config.configurable 注入，
所以 set_shared_tools() 与 set_skill_manager() 的运行时热替换不需要重新编译。

轮次上限用显式的 round 计数器而不是 recursion_limit：后者数的是 super-step、
默认 1000、超了抛 GraphRecursionError，文案和异常类型都会变。
"""
from __future__ import annotations

import inspect
import json
import logging
import operator
import os
import time
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from core.degradation import Dep, degrade
from core.llm import message_text
from core.tracing import add_event, trace_span

if TYPE_CHECKING:
    from agents.base import BaseAgent

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；错误配置不应阻塞服务启动。

    base.py 里有同名 helper，但 base 运行时 import 本模块，反向 import 会成环。
    """
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


# 一轮 = 一次 call_model。Agent 自行调 load_skill 取 Skill 正文会占掉一轮，
# 所以默认 4 而不是 3。进程启动时读一次，不是热配置。
MAX_TOOL_ROUNDS = _env_int("OPTISERVE_MAX_TOOL_ROUNDS", 4)


class ToolLoopState(TypedDict, total=False):
    req: Any
    conversation: List[BaseMessage]
    calls: List[Dict[str, Any]]
    round: int
    tools_used: List[str]
    traces: List[Dict[str, Any]]
    text: str
    exhausted: bool


def _inject(config: RunnableConfig) -> "tuple[BaseAgent, Any, Any]":
    """取出本次调用注入的 Agent 实例、已绑定工具的 chat model 与工具表。"""
    cfg = config["configurable"]
    return cfg["agent"], cfg["chat"], cfg["tools"]


async def render_prompt(state: ToolLoopState, config: RunnableConfig) -> Dict[str, Any]:
    """拼出发给模型的前置 user turn。对话历史不进 Agent，只进意图识别。"""
    agent, _, _ = _inject(config)
    conversation = [HumanMessage(content=text) for text in agent._context_turns(state["req"])]
    return {
        "conversation": conversation,
        "calls": [],
        "tools_used": [],
        "traces": [],
        "text": "",
        "exhausted": False,
    }


async def call_model(state: ToolLoopState, config: RunnableConfig) -> Dict[str, Any]:
    agent, chat, _ = _inject(config)
    # system 每轮重建：动态 Skills 只在每轮请求时拼装
    messages = [SystemMessage(content=agent._build_system_prompt(state["req"])), *state["conversation"]]
    with trace_span(
        f"llm_round_{state['round'] + 1}",
        agent_type=agent.agent_type.value,
        model=agent._model,
        prompt_turns=len(messages),
    ):
        resp = await chat.ainvoke(messages)

    # 参数解析失败的工具调用一并带进循环：走 _validate_tool_input 的同一失败分支，
    # 保证 assistant 的每个 tool_call 都有对应的 tool 消息回给端点。
    calls = list(resp.tool_calls) + [
        {"name": call.get("name") or "", "args": call.get("args"), "id": call.get("id") or ""}
        for call in resp.invalid_tool_calls
    ]
    if not calls:
        # 清空上一轮的工具调用，否则 after_model 会读到陈旧值而重复执行
        return {"round": state["round"] + 1, "calls": [], "text": message_text(resp)}

    conversation = state["conversation"] + [AIMessage(
        content=resp.content or "",
        tool_calls=[{**call, "type": "tool_call"} for call in calls],
    )]
    return {"round": state["round"] + 1, "conversation": conversation, "calls": calls}


def after_model(state: ToolLoopState) -> str:
    return "tools" if state["calls"] else "done"


async def run_tools(state: ToolLoopState, config: RunnableConfig) -> Dict[str, Any]:
    """执行模型本轮点名的工具。

    不用 prebuilt.ToolNode：它会把异常改写成英文 status=error，
    中文白名单文案与 tool_result 的 JSON 形状都会变。
    """
    agent, _, tools = _inject(config)
    req = state["req"]
    prefix = agent.agent_type.value

    conversation = list(state["conversation"])
    tools_used = list(state["tools_used"])
    traces = list(state["traces"])

    for call in state["calls"]:
        name = call["name"]
        tool_use_id = call["id"]
        args = call["args"]
        spec = tools.get(name)
        tool_t0 = time.monotonic()
        call_success = True
        result_success: bool | None = None
        error_text = ""
        with trace_span(
            f"tool:{name}",
            agent_type=prefix,
            tool_use_id=tool_use_id,
            input=dict(args) if isinstance(args, dict) else args,
        ) as span:
            if spec is None:
                call_success = False
                result: Any = {"success": False, "error": f"工具不在 {prefix} Agent 白名单中"}
                error_text = result["error"]
            else:
                try:
                    agent._validate_tool_input(spec, args)
                    result = spec.handler(req, args)
                    if inspect.isawaitable(result):
                        result = await result
                    tools_used.append(name)
                    if isinstance(result, dict) and "success" in result:
                        result_success = bool(result.get("success"))
                except Exception as ex:
                    call_success = False
                    degrade(Dep.TOOL, "agent_tool_failed", f"Agent 工具 {name} 执行失败: {ex}")
                    add_event("tool_failed", tool=name, error=str(ex))
                    error_text = str(ex)
                    result = {"success": False, "error": error_text}
            if span is not None:
                span.attrs.update(
                    success=call_success,
                    result_success=result_success,
                    cached=bool(result.get("cached")) if isinstance(result, dict) else False,
                    reranked=bool(result.get("reranked")) if isinstance(result, dict) else False,
                )
        tool_latency_ms = (time.monotonic() - tool_t0) * 1000
        if not error_text and isinstance(result, dict):
            error_text = str(result.get("error", "") or "")
        traces.append(
            {
                "agent_type": prefix,
                "tool_name": name,
                "tool_use_id": tool_use_id,
                "input": dict(args) if isinstance(args, dict) else args,
                "success": call_success,
                "result_success": result_success,
                "latency_ms": round(tool_latency_ms, 1),
                "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                "error": error_text,
            }
        )
        conversation.append(ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=tool_use_id,
        ))

    return {"conversation": conversation, "tools_used": tools_used, "traces": traces}


def after_tools(state: ToolLoopState) -> str:
    return "call_model" if state["round"] < MAX_TOOL_ROUNDS else "exhausted"


async def mark_exhausted(state: ToolLoopState, config: RunnableConfig) -> Dict[str, Any]:
    """跑满轮数：第 N 轮的工具照样执行、照样进 trace，这里只打标记，抛错留给调用方。"""
    return {"exhausted": True}


def build_tool_loop_graph():
    graph = StateGraph(ToolLoopState)
    graph.add_node("render_prompt", render_prompt)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("exhausted", mark_exhausted)

    graph.add_edge(START, "render_prompt")
    graph.add_edge("render_prompt", "call_model")
    graph.add_conditional_edges("call_model", after_model, {"tools": "tools", "done": END})
    graph.add_conditional_edges("tools", after_tools, {"call_model": "call_model", "exhausted": "exhausted"})
    graph.add_edge("exhausted", END)
    return graph.compile()


TOOL_LOOP_GRAPH = build_tool_loop_graph()


# ── G1：编排图 ────────────────────────────────────────────────────────────────


class OrchestratorState(TypedDict, total=False):
    """编排层状态。responses 用 add 归并器承接 Send 扇出的多路结果。

    want / seq 只出现在 run_agents 这一条分支的输入里，节点不写回这两个通道。
    """

    req: Any
    t0: float
    decision: Any
    responses: Annotated[List[Dict[str, Any]], operator.add]
    result: Any
    want: Any
    seq: int


def _orchestrator(config: RunnableConfig):
    return config["configurable"]["orchestrator"]


async def recognize_intent(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """意图补做：/chat 已在 API 层识别过则空转，CLI 与评测在这里补上。"""
    orc = _orchestrator(config)
    req = state["req"]
    with trace_span("intent_recognition", intent=req.intent.value if req.intent else None) as span:
        if req.intent is None:
            intent_result = await orc.recognize_intent(req.message, history=req.history)
            req.intent = intent_result.intent
            req.intent_confidence = intent_result.confidence
            if span is not None:
                span.attrs.update(source="graph", confidence=round(intent_result.confidence, 4))
        elif span is not None:
            span.attrs.update(source="api", confidence=round(req.intent_confidence, 4))
    return {}


async def guard(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """转人工 / 低置信度澄清：两条不经过任何 Agent 的直返路径。"""
    early = _orchestrator(config)._guard(state["req"], state["t0"])
    return {} if early is None else {"result": early}


def after_guard(state: OrchestratorState) -> str:
    return "done" if state.get("result") is not None else "route"


async def route(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """查表定主 Agent，再让 Monitor 的在线表现参与改选（降权见 _apply_demotion）。"""
    orc = _orchestrator(config)
    with trace_span("route") as span:
        decision = orc._apply_demotion(state["req"], orc._route_decision(state["req"]))
    if span is not None:
        span.attrs.update(
            primary=decision.primary_agent.value,
            supporting=[agent.value for agent in decision.supporting_agents],
            reason=decision.reason,
        )
    return {"decision": decision}


def after_route(state: OrchestratorState) -> str:
    return "parallel" if state["decision"].multi_agent else "single"


async def run_single(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    orc = _orchestrator(config)
    req = state["req"]
    decision = state["decision"]
    # 执行主 Agent（含降级）
    with trace_span(f"agent:{decision.primary_agent.value}", mode="single") as span:
        response = await orc._execute(req, decision.primary_agent)
        if span is not None:
            span.attrs.update(
                answered_by=response.agent_type.value,
                success=response.success,
                tools_used=list(response.tools_used),
            )
    return {"result": orc._single_result(req, decision, response, state["t0"])}


async def parallel(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    out = await PARALLEL_GRAPH.ainvoke(
        {"req": state["req"], "decision": state["decision"]}, config
    )
    return {"result": out["result"]}


async def prepare(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """并行段的计时锚点：从扇出前起算，不含意图识别与路由。"""
    return {"t0": time.monotonic(), "responses": []}


def fan_out(state: OrchestratorState) -> List[Send]:
    """按 decision.agent_types 动态扇出。固定条件边要把路由表复制进拓扑，故不用。"""
    return [
        Send("run_agents", {"req": state["req"], "want": agent_type, "seq": index})
        for index, agent_type in enumerate(state["decision"].agent_types)
    ]


async def run_agents(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    orc = _orchestrator(config)
    with trace_span(f"agent:{state['want'].value}", mode="parallel", seq=state["seq"]) as span:
        try:
            response = await orc._execute(state["req"], state["want"])
        except Exception:
            # 对应旧的 asyncio.gather(..., return_exceptions=True)：单条分支炸掉只少一条响应
            logger.exception("并行分支 %s 执行异常", state["want"])
            return {"responses": []}
        if span is not None:
            span.attrs.update(answered_by=response.agent_type.value, success=response.success)
    return {"responses": [{"seq": state["seq"], "response": response}]}


async def join(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """Send 的完成序不确定，按扇出下标还原 [primary] + supporting 后再交给 Composer。"""
    ordered = sorted(state["responses"], key=lambda item: item["seq"])
    result = await _orchestrator(config)._join_parallel(
        state["req"], state["decision"], [item["response"] for item in ordered], state["t0"]
    )
    return {"result": result}


def build_parallel_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("prepare", prepare)
    graph.add_node("run_agents", run_agents)
    graph.add_node("join", join)

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", fan_out, ["run_agents"])
    graph.add_edge("run_agents", "join")
    graph.add_edge("join", END)
    return graph.compile()


def build_orchestration_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("recognize_intent", recognize_intent)
    graph.add_node("guard", guard)
    graph.add_node("route", route)
    graph.add_node("single", run_single)
    graph.add_node("parallel", parallel)

    graph.add_edge(START, "recognize_intent")
    graph.add_edge("recognize_intent", "guard")
    graph.add_conditional_edges("guard", after_guard, {"done": END, "route": "route"})
    graph.add_conditional_edges("route", after_route, {"single": "single", "parallel": "parallel"})
    graph.add_edge("single", END)
    graph.add_edge("parallel", END)
    return graph.compile()


PARALLEL_GRAPH = build_parallel_graph()
ORCHESTRATION_GRAPH = build_orchestration_graph()
