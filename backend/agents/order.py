"""订单与物流 Agent。"""
import json
from typing import Dict

from agents.base import AgentProfile, AgentType, BaseAgent, Request
from tools.agent_tools import AgentToolSpec, general_tools


class OrderAgent(BaseAgent):
    agent_type = AgentType.ORDER
    profile = AgentProfile(
        role="订单与物流处理",
        mission="区分订单状态、发货、配送、地址变更等场景，说明当前可判断的事实，并明确需要核验的字段。",
        workflow=("确认订单场景", "收集订单号", "说明当前状态边界", "给出下一步路径", "判断是否升级"),
        input_contract=("订单号", "下单时间", "物流单号", "收货信息", "用户期望", "知识库上下文"),
        output_contract=("需要核验的信息", "当前可判断内容", "下一步处理路径", "时效边界"),
        handoff_conditions=("需要修改已发货订单的地址", "物流长时间无更新", "订单金额异常", "需要后台系统操作"),
        tool_scope=("search_knowledge_base", "load_skill", "inspect_request_context", "suggest_required_fields"),
        temperature=0.1,
        max_tokens=1000,
    )
    system_prompt = (
        "你是订单与物流服务专家。专注于：订单状态查询、发货进度、配送时效、地址变更。"
        "不得声称已查询后台订单系统或物流轨迹；缺少订单号时先索要，再说明处理路径。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["order_fields"] = {
            "order_id": req.entities.get("order_id", []),
            "date": req.entities.get("date", []),
            "missing_fields": [
                name for name, values in (("订单号", req.entities.get("order_id", [])),) if not values
            ],
            "risk_boundary": "不得承诺具体到货时间，不得声称已修改收货信息",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools
