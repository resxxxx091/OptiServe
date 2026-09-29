import json
from typing import Dict

from agents.base import AgentProfile, AgentType, BaseAgent, Request
from tools.agent_tools import AgentToolSpec, service_tools


class ServiceAgent(BaseAgent):
    agent_type    = AgentType.SERVICE
    profile = AgentProfile(
        role="订单履约与客诉处理",
        mission="处理异常订单、退款审核线索和客诉升级：梳理事实、给出处理路径与时效边界，实际赔付和操作走人工。",
        workflow=("确认订单与问题", "区分可判断事实与需核验事实", "给出处理路径", "明确升级条件"),
        input_contract=("订单号", "问题类型", "发生时间", "用户期望", "知识库上下文"),
        output_contract=("事实梳理", "处理路径", "时效边界", "升级/人工受理条件"),
        handoff_conditions=("实际退款或赔付", "需要后台改单操作", "涉及法律纠纷或监管投诉"),
        tool_scope=("search_knowledge_base", "load_skill", "query_anomalous_orders", "build_diagnostic_plan"),
        temperature=0.1,
        max_tokens=1200,
    )
    system_prompt = (
        "你是电商运营服务系统的订单与客诉处理 Agent。专注于：异常订单处理、退款审核线索梳理、"
        "客诉工单升级、系统故障初步排查（后台报错/登录异常）。"
        "给出处理路径和时效边界时不得编造具体时限，实际赔付与改单操作必须说明走人工。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["service_fields"] = {
            "order_ids": req.entities.get("order_id", []),
            "error_codes": req.entities.get("error_code", []),
            "risk_boundary": "不得承诺退款成功或赔付金额；不得要求买家密码与验证码",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(service_tools())
        return tools
