"""
Agent 侧工具契约与不查外部系统的确定性工具，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具类型：
  - 当前请求分析
  - 技术排障建议
  - 账单字段核验
  - 业务 Skill 规范正文、附表与演示操作的按需取回
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.base import Request

logger = logging.getLogger(__name__)


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def openai_tool_specs(specs: Iterable[AgentToolSpec]) -> List[Dict[str, Any]]:
    """把 Agent 工具白名单转成 OpenAI function 格式（DeepSeek 走这套）。"""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }
        for spec in specs
    ]


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"order_status", "logistics"}:
        fields = ["订单号或下单时间"]
    elif intent in {"refund", "invoice", "payment_issue"}:
        fields = ["订单号或交易号", "金额与发生时间"]
    elif intent in {"technical_login", "technical_crash"}:
        fields = ["错误码或错误提示", "问题发生时间"]
    elif intent in {"complaint", "human_handoff"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：解释常见错误码的排查方向，不声称读取了服务端日志。

    TODO: 这张表与 skills/technical-support/references/error-codes.md 是同一份
    知识的两处副本，将来应归一到热加载那一侧。
    """
    code = str(args.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
    }
    meaning, steps = mapping.get(
        code,
        ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]),
    )
    return {
        "error_code": code,
        "meaning": meaning,
        "next_steps": steps,
        "server_log_checked": False,
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：生成低风险排障顺序。"""
    environment = str(args.get("environment", "unknown"))[:80]
    reproduced = bool(args.get("reproduced", False))
    steps = [
        "复现并记录完整错误信息",
        "确认网络、DNS、代理和证书",
        "确认版本、配置和权限",
    ]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": environment,
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def check_billing_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：检查必要核验字段是否齐全。"""
    fields = {
        "order_id": bool(req.entities.get("order_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "当前工具只做字段检查，不连接订单或支付系统",
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：只做用户明确提供金额之间的算术。"""
    try:
        first = float(args["amount_a"])
        second = float(args["amount_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "amount_a 和 amount_b 必须是数字"}
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def build_skill_tools(
    get_manager: Callable[[], Any],
    agent_type: Optional[str] = None,
) -> Dict[str, AgentToolSpec]:
    """构建三层渐进式披露的取回工具：正文、附表、演示操作。"""

    NAME_SCHEMA = {
        "name": {
            "type": "string",
            "description": "Skill 名称，须与 [可用 Skills] 索引中的名称逐字一致",
        },
    }

    def resolve(manager: Any, name: str) -> Dict[str, Any]:
        """三条工具共用同一种失败形状：带上可见名称，让模型自己纠正。"""
        if manager is None:
            return {"error": "Skills 未初始化", "available": []}
        if not name:
            return {"error": "name 不能为空", "available": []}
        skill = manager.body_for(name)
        if skill is None:
            return {
                "error": f"未找到名为「{name}」的 Skill，请从 [可用 Skills] 索引中逐字复制名称",
                "available": [item.name for item in manager.catalog_for(agent_type)],
            }
        return {"skill": skill}

    def load_skill(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        manager = get_manager()
        resolved = resolve(manager, str(args.get("name") or "").strip())
        if "error" in resolved:
            return {"success": False, "content": "", **resolved}
        skill = resolved["skill"]
        return {
            "success": True,
            "name": skill.name,
            "description": skill.description,
            "content": skill.content.strip(),
            # 清单随正文一起给出，模型才知道下一跳该传哪个相对路径。
            "resources": skill.resources,
            "scripts": skill.scripts,
        }

    def load_skill_resource(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        manager = get_manager()
        resolved = resolve(manager, str(args.get("name") or "").strip())
        if "error" in resolved:
            return {"success": False, "content": "", **resolved}
        skill = resolved["skill"]
        rel = str(args.get("path") or "").strip()
        content = manager.resource_for(skill, rel)
        if content is None:
            return {
                "success": False,
                "content": "",
                "error": f"未能读取「{skill.name}」下的「{rel}」，路径须逐字取自下面的清单",
                "available": skill.resources,
            }
        return {
            "success": True,
            "name": skill.name,
            "path": rel.replace("\\", "/"),
            "content": content.strip(),
        }

    def run_skill_script(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        """演示操作：只登记一次调用并回执，脚本文件的内容从不被执行。

        真实动作发生在线下，所以回执只说"已受理"——billing 这类规范正文规定
        缺少核验信息时不得承诺结果，回执不能替模型破它自己的规矩。
        """
        manager = get_manager()
        resolved = resolve(manager, str(args.get("name") or "").strip())
        if "error" in resolved:
            return {"success": False, **resolved}
        skill = resolved["skill"]
        script = manager.script_key(skill, str(args.get("script") or ""))
        if script is None:
            return {
                "success": False,
                "error": f"「{skill.name}」下没有登记名为「{args.get('script')}」的操作，script 须逐字取自 load_skill 返回的 scripts 清单",
                "available": skill.scripts,
            }
        params = args.get("args") or {}
        logger.info("[skill-sim] request=%s skill=%s script=%s args=%s",
                    req.request_id, skill.name, script, params)
        return {
            "success": True,
            "name": skill.name,
            "operation": script,
            "status": "accepted",
            "detail": "已受理，等待人工核验",
            "request_id": req.request_id,
        }

    return {
        "load_skill": make_tool(
            "load_skill",
            "按名称加载业务 Skill 的完整规范正文。当问题涉及对外业务口径——退款与到账时效、"
            "发票与扣款、订单状态与到货时间、故障排查步骤、升级条件与禁止事项——时，"
            "先加载对应 Skill 再据此回答。name 必须与 system prompt 的 [可用 Skills] 索引逐字一致。"
            "返回体里的 resources/scripts 是该 Skill 的第三层清单，正文指向附表或操作时会一并给出。"
            "需要多个 Skill 或其它工具时，请在同一轮内并行发起多个调用。",
            NAME_SCHEMA,
            load_skill,
            required=["name"],
        ),
        "load_skill_resource": make_tool(
            "load_skill_resource",
            "读取某个 Skill 目录下的附表或参考文档（例如错误码对照表、费率表、话术模板）。"
            "只有当已加载的正文指向它、或你需要正文未展开的细节时才调用。"
            "path 是相对 SKILL.md 的路径，必须逐字取自 load_skill 返回的 resources 列表，不能自行拼造。",
            {
                **NAME_SCHEMA,
                "path": {
                    "type": "string",
                    "description": "相对该 Skill 根目录的文件路径，须与 resources 清单逐字一致",
                },
            },
            load_skill_resource,
            required=["name", "path"],
        ),
        "run_skill_script": make_tool(
            "run_skill_script",
            "发起一次需要人工或二线接手的业务操作（例如转二线技术、提交退款核验工单）。"
            "当规范正文要求走升级或人工流程时使用，返回的是一次受理回执，只代表操作已登记，不代表结果已生效。"
            "script 必须逐字取自 load_skill 返回的 scripts 清单，形如 scripts/xxx.py。",
            {
                **NAME_SCHEMA,
                "script": {
                    "type": "string",
                    "description": "相对该 Skill 根目录的脚本路径，须与 scripts 清单中的条目逐字一致",
                },
                "args": {
                    "type": "object",
                    "description": "该操作需要的关键字段（订单号、错误码、request_id 等），可为空",
                },
            },
            run_skill_script,
            required=["name", "script"],
        ),
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "解释常见 HTTP 错误码的可能含义和低风险排查方向；不会读取服务端日志。",
            {"error_code": {"type": "string", "description": "例如 401、403、500"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "根据运行环境和是否可复现生成排障顺序，不执行修改配置等操作。",
            {
                "environment": {"type": "string", "description": "App、浏览器、服务端或 Docker 等"},
                "reproduced": {"type": "boolean", "description": "问题是否可以稳定复现"},
            },
            build_diagnostic_plan,
            required=["environment", "reproduced"],
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_billing_fields": make_tool(
            "check_billing_fields",
            "检查账单核验字段是否齐全；不连接订单、支付或退款系统。",
            {"payment_channel": {"type": "string", "description": "支付渠道，例如微信、支付宝、银行卡"}},
            check_billing_fields,
        ),
        "compare_amounts": make_tool(
            "compare_amounts",
            "计算用户明确提供的两笔金额差值；不判断是否重复扣款，也不执行退款。",
            {
                "amount_a": {"type": "number", "description": "第一笔金额"},
                "amount_b": {"type": "number", "description": "第二笔金额"},
            },
            compare_amounts,
            required=["amount_a", "amount_b"],
        ),
    }
