import json
from typing import Dict

from agents.base import AgentProfile, AgentType, BaseAgent, Request
from tools.agent_tools import AgentToolSpec, data_tools


class DataAgent(BaseAgent):
    agent_type    = AgentType.DATA
    profile = AgentProfile(
        role="店铺数据查询与分析",
        mission="查询经营指标、库存与异常订单，把数字翻译成运营能直接用的结论，并明确数据口径和边界。",
        workflow=("确认指标口径与时间范围", "调用数据工具取数", "区分事实与推断", "给出可执行的下一步", "判断是否需要人工核数"),
        input_contract=("指标名称", "时间范围", "商品或SKU", "知识库上下文"),
        output_contract=("数据口径说明", "查询结果", "与对比期的变化", "结论与建议"),
        handoff_conditions=("数据与财务对不上", "需要导出真实业务库明细", "涉及退款赔付金额审批"),
        tool_scope=("search_knowledge_base", "load_skill", "query_metrics", "query_inventory", "query_anomalous_orders"),
        temperature=0.0,
        max_tokens=1100,
    )
    system_prompt = (
        "你是电商运营服务系统的数据分析 Agent。专注于：经营指标查询（销售额/订单/流量/转化/退款）、"
        "商品库存查询、异常订单梳理。回答数据问题时先说明口径与时间范围；"
        "数据来自演示环境模拟数据源时必须明确告知，不得把模拟数据说成真实经营结果。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["data_fields"] = {
            "dates": req.entities.get("date", []),
            "amounts": req.entities.get("amount", []),
            "risk_boundary": "不得编造查询结果之外的数字；对变化归因时区分事实与推测",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(data_tools())
        return tools
