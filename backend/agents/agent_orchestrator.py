"""
路由决策（_route_decision）：
  1. 意图映射表 —— _INTENT_AGENT 按 IntentCategory 唯一确定主处理 Agent，查不到降级 GeneralAgent
  2. 实体触发协作 —— error_code / amount 结构化实体拉入 Technical / Billing 作为 supporting Agent
  3. 转人工 —— human_handoff 意图不经过任何 Agent，由 handoff() 直接产出交接文案

并行协作：
  - 有 supporting Agent 时由编排图的 parallel 节点扇出到 PARALLEL_GRAPH，结果经 ResponseComposer 合并后返回

降级与升级：
  - _best_agent 按 routing_score() 选最优实例；专属 Agent 失败时降级到 GeneralAgent
  - Monitor 回写的 monitor_penalty 越过 ROUTING_DEMOTE_THRESHOLD 时，_apply_demotion
    在意图合法的候选里改选主 Agent（查表结果本身不动）
  - Agent 自检命中升级话术，或意图为转人工 → escalated=True
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from agents.graph import ORCHESTRATION_GRAPH
from agents.base import (
    AgentResponse,
    AgentType,
    BaseAgent,
    Request,
)
from agents.billing import BillingAgent
from agents.general import GeneralAgent
from agents.order import OrderAgent
from agents.technical import TechnicalAgent
from core.degradation import Dep, degrade
from core.intent_recognizer import IntentCategory, IntentRecognizer
from core.llm import LLMProvider, message_text
from core.tracing import add_event, trace_scope, trace_span
from tools.agent_tools import AgentToolSpec

logger = logging.getLogger(__name__)

# Monitor 回写的降权系数越过这条线，路由才在意图合法的候选里改选主 Agent。
# 0.5 的取值来自 _routing_penalty：真要触发改选，该 Agent 早就在告警区里了。
ROUTING_DEMOTE_THRESHOLD = float(os.getenv("OPTISERVE_ROUTING_DEMOTE_THRESHOLD", "0.5"))


# ── 数据结构 ──────────────────────────────────────────────────────────────────


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0


@dataclass
class RoutingDecision:
    """一次请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


# 意图 → 主处理 Agent 的唯一映射表。查不到就走兜底 GeneralAgent。
_INTENT_AGENT: Dict[IntentCategory, AgentType] = {
    IntentCategory.GREETING:        AgentType.GENERAL,
    IntentCategory.FEEDBACK:        AgentType.GENERAL,
    IntentCategory.COMPLAINT:       AgentType.GENERAL,
    IntentCategory.ORDER_STATUS:    AgentType.ORDER,
    IntentCategory.LOGISTICS:       AgentType.ORDER,
    IntentCategory.REFUND:          AgentType.BILLING,
    IntentCategory.INVOICE:         AgentType.BILLING,
    IntentCategory.PAYMENT_ISSUE:   AgentType.BILLING,
    IntentCategory.TECHNICAL_LOGIN: AgentType.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: AgentType.TECHNICAL,
}

# 低置信度时的澄清话术。必须是固定文本——判断"是否已经问过一次"靠比对历史里这一条。
CLARIFY_PROMPT = "我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单，还是技术故障？"


def handoff(req: Request) -> str:
    """转人工：把已知信息整理成标准交接文案。不调 LLM，不做业务操作。"""
    intent = req.intent.value if req.intent else "unknown"
    entities = json.dumps(req.entities or {}, ensure_ascii=False)
    return (
        "我已将这个问题标记为人工升级处理。\n\n"
        f"升级原因：意图={intent}\n"
        f"已记录信息：{entities}\n"
        "请不要发送短信验证码或完整支付凭证；人工客服会根据会话记录继续核验。"
    )


class ResponseComposer:
    """多 Agent 汇总节点，统一主次、去重和输出边界。"""

    def __init__(self, llm: LLMProvider, model: str):
        self._llm = llm
        self._model = model

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [response for response in responses if response.success and response.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        evidence = "\n\n".join(
            f"[{response.agent_type.value} Agent 输出]\n{response.content}"
            for response in successful
        )
        prompt = (
            "你是客服 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造订单、退款、后台查询结果；如果结论冲突，明确说明需要核验；"
            "保留必要的排查步骤、核验字段和升级边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n"
            f"候选结果：\n{evidence}"
        )
        try:
            chat = self._llm.chat_model(
                model=self._model,
                temperature=float(os.getenv("OPTISERVE_COMPOSER_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("OPTISERVE_COMPOSER_MAX_TOKENS", "1000")),
            )
            message = await chat.ainvoke([HumanMessage(content=prompt)])
            content = message_text(message).strip()
            if content:
                return content
        except Exception as ex:
            degrade(Dep.LLM, "compose_failed", f"Response Composer 失败，改用确定性合并: {ex}")

        # 汇总节点不可用时保留主次标签，避免丢失某个专业 Agent 的结论。
        return "\n\n".join(
            f"{response.content}" if index == 0 else f"补充说明：\n{response.content}"
            for index, response in enumerate(successful)
        )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由决策见 _route_decision：意图映射表定主 Agent，结构化实体触发辅助 Agent；
    转人工意图走模块级 handoff()，不经过任何 Agent。
    同类多实例时由 _best_agent 按 routing_score() 选最优，专属 Agent 失败降级到 GeneralAgent。
    """

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "deepseek-flash",
        skill_manager: Optional[Any] = None,
    ):
        llm = LLMProvider(api_key, base_url)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._composer = ResponseComposer(llm, model)
        self._shared_tools: Dict[str, AgentToolSpec] = {}

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [self._make_agent(GeneralAgent, llm, model, skill_manager)],
            AgentType.TECHNICAL: [self._make_agent(TechnicalAgent, llm, model, skill_manager)],
            AgentType.BILLING: [self._make_agent(BillingAgent, llm, model, skill_manager)],
            AgentType.ORDER:     [self._make_agent(OrderAgent, llm, model, skill_manager)],
        }

    @staticmethod
    def _make_agent(
        agent_cls: type[BaseAgent],
        llm: LLMProvider,
        default_model: str,
        skill_manager: Optional[Any],
    ) -> BaseAgent:
        """按角色创建 Agent，并允许用环境变量覆盖该角色的模型。"""
        profile = agent_cls.profile
        env_name = f"OPTISERVE_{agent_cls.agent_type.value.upper()}_MODEL"
        model = os.getenv(env_name, "").strip() or profile.model
        configured_profile = replace(profile, model=model) if model else profile
        return agent_cls(llm, default_model, skill_manager, profile=configured_profile)

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        """更新所有 Agent 共享的工具白名单。"""
        self._shared_tools = dict(tools or {})
        for agents in self._pool.values():
            for agent in agents:
                agent.set_shared_tools(self._shared_tools)

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """对外暴露意图识别，供 API 层先判断是否需要 RAG 等前置能力。"""
        return await self._intent_recognizer.recognize(message, history=history)

    def _warn_unloaded_hint(
        self,
        req: Request,
        primary_agent: Optional[AgentType],
        tools_used: List[str],
    ) -> None:
        """渐进式披露探针：关键词已提示某 Skill，但 Agent 没有去加载正文。

        索引里有【命中】标记意味着"用户措辞确实落在该规范的适用面上"，此时不调
        load_skill 就是漏加载——这条 warning 是"注入是否被判断取代"的观测手段。
        """
        if self._skill_manager is None or "load_skill" in tools_used:
            return
        hinted = self._skill_manager.hinted_for(req.message, primary_agent.value if primary_agent else None)
        if hinted:
            logger.warning(
                "关键词命中 Skill 提示但未加载: agent=%s hinted=%s request_id=%s",
                primary_agent.value if primary_agent else "unknown",
                [skill.name for skill in hinted],
                req.request_id,
            )

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程（拓扑见 agents.graph.ORCHESTRATION_GRAPH）：
          意图补做 → 直返守卫 → 路由 → 单 Agent / Send 并行扇出 → 结果
        """
        # /chat 已经开过 trace 就直接沿用，没有现成 trace 的入口自己开一条
        with trace_scope(req.request_id, "chat") as recorder:
            recorder.meta.update(
                user_id=req.user_id,
                conv_id=req.conv_id,
                intent=req.intent.value if req.intent else None,
            )
            state = await ORCHESTRATION_GRAPH.ainvoke(
                {"req": req, "t0": time.monotonic()},
                {"configurable": {"orchestrator": self}},
            )
        return state["result"]

    def _guard(self, req: Request, t0: float) -> Optional[OrchestratorResult]:
        """两条不经过任何 Agent 的直返路径：转人工交接、低置信度澄清。"""
        # 转人工意图不经过任何 Agent，直接返回交接文案。
        if req.intent is IntentCategory.HUMAN_HANDOFF:
            return self._early_result(
                req, t0, handoff(req), "意图为 human_handoff，直接转人工，不经过 Agent",
                escalated=True,
            )
        if self._needs_clarification(req):
            return self._early_result(req, t0, CLARIFY_PROMPT, "低置信度 OTHER 意图，先澄清用户需求")
        return None

    def _single_result(
        self, req: Request, decision: RoutingDecision, response: AgentResponse, t0: float,
    ) -> OrchestratorResult:
        """单 Agent 路径的结果装配（含升级检查）。"""
        # 升级检查：Agent 自检命中升级话术
        escalated = False
        if response.escalate:
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: intent={req.intent}")
            # 生产环境：此处创建工单、通知人工客服

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(response.tools_used),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
        )
        self._warn_unloaded_hint(req, response.agent_type, response.tools_used)
        return result

    def _early_result(
        self, req: Request, t0: float, content: str, reason: str, escalated: bool = False,
    ) -> OrchestratorResult:
        """不经过 Agent 的直返路径：澄清追问、转人工交接。"""
        result = OrchestratorResult(
            request_id=req.request_id,
            response=content,
            agent_type=AgentType.GENERAL,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[AgentType.GENERAL],
            primary_agent=AgentType.GENERAL,
            routing_reason=reason,
            routing_confidence=req.intent_confidence,
        )
        return result

    async def _join_parallel(
        self, req: Request, decision: RoutingDecision, responses: List[AgentResponse], t0: float,
    ) -> OrchestratorResult:
        """汇聚扇出的响应：compose 取 successful[0] 作「主 Agent」，故顺序必须确定。"""
        agent_types = decision.agent_types
        with trace_span(
            "compose",
            input=req.message,
            agents=[agent_type.value for agent_type in agent_types],
        ) as span:
            combined = await self._composer.compose(req, responses)
            span.output = combined
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            escalated=any(response.escalate for response in responses),
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[
                response.agent_type for response in responses if response.success
            ] or agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            tools_used=list(dict.fromkeys(
                tool_name
                for response in responses
                for tool_name in response.tools_used
            )),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
        )
        for response in responses:
            self._warn_unloaded_hint(req, response.agent_type, response.tools_used)
        return result

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _route_decision(req: Request) -> RoutingDecision:
        """按意图映射表选主 Agent，再由结构化实体触发辅助 Agent。"""
        primary = _INTENT_AGENT.get(req.intent)  # type: ignore[arg-type]

        if primary is None:
            intent_name = req.intent.value if req.intent else "unknown"
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason=f"意图 {intent_name} 未命中映射表，降级到 GeneralAgent",
                confidence=req.intent_confidence,
            )

        entities = req.entities or {}
        supporting: List[AgentType] = []
        if entities.get("error_code"):
            supporting.append(AgentType.TECHNICAL)
        if entities.get("amount"):
            supporting.append(AgentType.BILLING)
        supporting = [agent for agent in supporting if agent != primary]

        return RoutingDecision(
            primary_agent=primary,
            supporting_agents=supporting,
            reason=AgentOrchestrator._routing_reason(req, primary, supporting),
            confidence=req.intent_confidence,
        )

    @staticmethod
    def _routing_reason(
        req: Request,
        primary_agent: AgentType,
        supporting_agents: List[AgentType],
    ) -> str:
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        intent = req.intent.value if req.intent else "unknown"
        return f"intent={intent}, primary={primary_agent.value}, supporting={support_text}"

    def _type_penalty(self, agent_type: AgentType) -> float:
        """该类型实例里最重的一个 Monitor 降权系数。"""
        return max(
            (agent.stats.monitor_penalty for agent in self._pool.get(agent_type, [])),
            default=0.0,
        )

    def _apply_demotion(self, req: Request, decision: RoutingDecision) -> RoutingDecision:
        """Monitor 回写的降权系数越线时，在意图合法的候选里改选主 Agent。"""
        primary = decision.primary_agent
        penalty = self._type_penalty(primary)
        if primary is AgentType.GENERAL or penalty < ROUTING_DEMOTE_THRESHOLD:
            return decision

        healthy = [
            agent for agent in decision.supporting_agents
            if agent is AgentType.GENERAL or self._type_penalty(agent) < ROUTING_DEMOTE_THRESHOLD
        ]
        candidates = [agent for agent in healthy if agent is not primary]
        if AgentType.GENERAL not in candidates:
            candidates.append(AgentType.GENERAL)
        chosen = candidates[0]
        supporting = [agent for agent in healthy if agent is not chosen]
        add_event(
            "routing.demote",
            object=primary.value,
            penalty=round(penalty, 3),
            threshold=ROUTING_DEMOTE_THRESHOLD,
            demoted_to=chosen.value,
        )
        return RoutingDecision(
            primary_agent=chosen,
            supporting_agents=supporting,
            reason=(
                f"{AgentOrchestrator._routing_reason(req, chosen, supporting)}"
                f", demoted_from={primary.value}"
            ),
            confidence=decision.confidence,
        )

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """低置信度且无明确意图时，先追问，避免误路由。同一轮对话只追问一次。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        if AgentOrchestrator._already_clarified(req):
            return False
        return req.intent_confidence < 0.5

    @staticmethod
    def _already_clarified(req: Request) -> bool:
        """上一轮助手消息就是澄清话术 → 这轮不再问，交给兜底 Agent。"""
        history = req.history or []
        if not history:
            return False
        last = history[-1]
        return (
            last.get("role") == "assistant"
            and (last.get("content") or "").strip() == CLARIFY_PROMPT
        )

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """性能路由：从同类 Agent 中选 routing_score() 最高的。"""
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        if not response.success and agent_type != AgentType.GENERAL:
            degrade(Dep.AGENT, "agent_fallback", f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                }
        return result

    def routing_weights(self) -> Dict[str, Any]:
        """当前生效的路由权重：每类 Agent 的降权系数、改判阈值、是否已越线。"""
        return {
            "demote_threshold": ROUTING_DEMOTE_THRESHOLD,
            "agents": {
                agent_type.value: {
                    "penalty": round(self._type_penalty(agent_type), 3),
                    "demoted": self._type_penalty(agent_type) >= ROUTING_DEMOTE_THRESHOLD,
                }
                for agent_type in self._pool
            },
        }

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """接收 Monitor 的在线表现反馈，动态调整路由惩罚项。"""
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
