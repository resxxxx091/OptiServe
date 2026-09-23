"""Agent 契约与基类。

本模块只放「被多个 Agent 共享的东西」：类型枚举、profile、统计、请求/响应结构，
以及所有 Agent 的基类 BaseAgent。编排与路由不在这里，见 agents/agent_orchestrator.py。
"""
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.graph import TOOL_LOOP_GRAPH
from agents.tool_adapter import openai_tool_specs
from agents.tools import AgentToolSpec, build_skill_tools
from core.degradation import Dep, degrade
from core.intent_recognizer import IntentCategory
from core.llm import LLMProvider

logger = logging.getLogger(__name__)


class AgentType(Enum):
    GENERAL   = "general"    # 通用客服
    TECHNICAL = "technical"  # 技术支持
    BILLING   = "billing"    # 账单/退款
    ORDER     = "order"      # 订单与物流


@dataclass(frozen=True)
class AgentProfile:

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


def _env_float(name: str, default: float) -> float:
    """读取可选浮点配置；错误配置不应阻塞服务启动。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
        return default


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；错误配置不应阻塞服务启动。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


# 单个 Agent 跑完一次工具循环的总预算。轮数上限（MAX_TOOL_ROUNDS）管得住"绕太多圈"，
# 管不住"某一圈里模型迟迟不返包"——那一跳没有自己的时限，只能靠这层封顶。
AGENT_LOOP_TIMEOUT_S = _env_float("OPTISERVE_AGENT_LOOP_TIMEOUT_S", 90.0)


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


class ToolRoundsExhausted(RuntimeError):
    """G2 跑满轮数。消息与旧版裸 RuntimeError 一致。"""


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tools_used:  List[str] = field(default_factory=list)


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_confidence: float = 1.0
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用、角色契约和统计。"""

    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile

    def __init__(
        self,
        llm: LLMProvider,
        model: str,
        skill_manager: Optional[Any] = None,
        profile: Optional[AgentProfile] = None,
    ):
        self.profile = profile or self.profile
        self._model  = self.profile.model or model
        self._llm    = llm
        self._chat   = llm.chat_model(
            model=self._model,
            temperature=self.profile.temperature,
            max_tokens=self.profile.max_tokens,
        )
        self._skill_manager = skill_manager
        self.stats   = AgentStats()
        self._shared_tools: Dict[str, AgentToolSpec] = {}

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        """返回该角色真实可调用的工具白名单。"""
        tools = dict(self._shared_tools)
        # 晚绑定 self._skill_manager：共享表会被 api 启动后期的 set_shared_tools
        # 整体替换抹掉，且热加载换引用后这里每次都现取。
        tools.update(build_skill_tools(lambda: self._skill_manager, self.agent_type.value))
        return tools

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            content, tools_used = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=list(tools_used),
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            degrade(Dep.AGENT, "agent_failed", f"{self.agent_type.value} 处理失败，返回兜底文案: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request) -> Tuple[str, List[str]]:
        """跑 G2 工具子图；agent/chat/工具表按次注入，图只在导入时编译一次。

        返回 (正文, tools_used)。不在实例属性上暂存：Agent 是池化共享
        实例，并发请求交错时 A 的响应会带上 B 的工具名。
        """
        tools = self.get_tools()
        chat = self._chat if not tools else self._chat.bind_tools(openai_tool_specs(tools.values()))
        try:
            state = await asyncio.wait_for(
                TOOL_LOOP_GRAPH.ainvoke(
                    {"req": req, "round": 0},
                    {"configurable": {"agent": self, "chat": chat, "tools": tools}},
                ),
                timeout=AGENT_LOOP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            degrade(
                Dep.LLM,
                "agent_loop_timeout",
                f"{self.agent_type.value} 工具循环超过总预算 {AGENT_LOOP_TIMEOUT_S:g}s",
            )
            raise
        if state["exhausted"]:
            raise ToolRoundsExhausted(f"{self.agent_type.value} 工具调用超过最大轮数")
        return state["text"], list(state["tools_used"])

    def _context_turns(self, req: Request) -> List[str]:
        """请求前置的合成 user turn：背景 / 结构化实体 / 角色契约。对话历史不进 Agent。"""
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        turns: List[str] = []
        if req.context:
            turns.append(f"[背景信息]\n{_clean(req.context)}")
        if req.entities:
            turns.append(f"[结构化实体]\n{_clean(json.dumps(req.entities, ensure_ascii=False))}")
        role_packet = self._build_role_packet(req)
        if role_packet:
            turns.append(f"[角色输入契约]\n{_clean(role_packet)}")
        turns.append(_clean(req.message))
        return turns

    @staticmethod
    def _validate_tool_input(spec: AgentToolSpec, args: Any) -> None:
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        schema = spec.input_schema
        for field_name in schema.get("required", []):
            if field_name not in args:
                raise ValueError(f"缺少必需参数: {field_name}")
        properties = schema.get("properties", {})
        unknown = set(args) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected in type_map and not isinstance(value, type_map[expected]):
                raise ValueError(f"参数 {key} 类型错误，期望 {expected}")

    def _build_system_prompt(self, req: Request) -> str:
        """把角色契约和 Skills 索引拼入 system prompt。

        插槽里只有 name + description：正文要不要进上下文由本 Agent 自己判断，
        判断的结果是调用 load_skill 工具。
        """
        profile_prompt = (
            f"\n\n[角色契约]\n"
            f"角色：{self.profile.role}\n"
            f"职责：{self.profile.mission}\n"
            f"处理流程：{' -> '.join(self.profile.workflow)}\n"
            f"可用输入：{'；'.join(self.profile.input_contract)}\n"
            f"输出要求：{'；'.join(self.profile.output_contract)}\n"
            f"升级条件：{'；'.join(self.profile.handoff_conditions) or '无，按通用客服规则处理'}\n"
            f"允许的数据/工具范围：{'、'.join(self.profile.tool_scope) or '仅使用当前请求上下文'}\n"
            "不要声称执行了未提供的查询、修改或退款操作；缺少证据时明确说明需要核验。"
        )
        base_prompt = f"{self.system_prompt}{profile_prompt}"
        if self._skill_manager is None:
            return base_prompt
        skill_index = self._skill_manager.index_for(req.message, self.agent_type.value)
        if not skill_index:
            return base_prompt
        return f"{base_prompt}\n\n[可用 Skills]\n{skill_index}"

    def _build_role_packet(self, req: Request) -> str:
        """给子 Agent 的确定性输入包；子类可补充领域字段。"""
        packet = {
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {},
        }
        return json.dumps(packet, ensure_ascii=False)

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "人工客服", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)
