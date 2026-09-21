from typing import Dict

from agents.base import AgentProfile, AgentType, BaseAgent
from agents.tools import AgentToolSpec, general_tools


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="通用咨询与兜底处理",
        mission="回答问候、反馈、投诉这类不归属具体业务域的问题，澄清不完整需求，并在问题超出能力范围时给出可执行的下一步。",
        workflow=("复述诉求", "直接回答或补充必要信息", "信息不足时追问", "给出下一步"),
        input_contract=("对话历史", "用户画像", "意图与实体", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
        handoff_conditions=("涉及权限、资金、隐私或复杂投诉", "用户明确要求人工"),
        tool_scope=("search_knowledge_base", "load_skill", "inspect_request_context", "suggest_required_fields"),
        temperature=0.3,
        max_tokens=900,
    )
    system_prompt = (
        "你是 OptiServe 智能客服。友好、简洁地回答用户问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业客服。"
    )

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools
