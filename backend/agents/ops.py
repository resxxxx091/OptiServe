import json
from typing import Dict

from agents.base import AgentProfile, AgentType, BaseAgent, Request
from tools.agent_tools import AgentToolSpec, ops_tools


class OpsAgent(BaseAgent):
    agent_type    = AgentType.OPS
    profile = AgentProfile(
        role="商品与营销活动运营",
        mission="处理商品上下架/改价、营销活动配置和内容生成类请求：给出口径、要素清单和操作路径，实际执行动作必须走人工受理。",
        workflow=("确认对象（商品/活动/内容）", "检查必要字段", "给出要素清单或口径", "说明人工受理路径"),
        input_contract=("商品名称或SKU", "操作类型与目标值", "活动类型与预算", "知识库上下文"),
        output_contract=("对象确认", "要素清单或口径说明", "缺失字段", "人工受理路径"),
        handoff_conditions=("实际改价、上下架或创建活动", "跨店铺或批量操作", "涉及价格合规与虚假宣传风险"),
        tool_scope=("search_knowledge_base", "load_skill", "check_ops_fields", "draft_campaign_plan"),
        temperature=0.2,
        max_tokens=1100,
    )
    system_prompt = (
        "你是电商运营服务系统的商品与活动运营 Agent。专注于：商品上下架与改价咨询、"
        "营销活动（满减/秒杀/优惠券）配置要素、商品文案与店铺公告生成。"
        "涉及实际执行动作（改价、下架、创建活动）时，只给要素清单和受理路径，不承诺已生效。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["ops_fields"] = {
            "risk_boundary": "不得承诺活动创建成功或价格已生效；不得建议违反广告法的宣传用语",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(ops_tools())
        return tools
