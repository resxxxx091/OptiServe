"""AgentToolSpec → ChatDeepSeek 认识的工具描述。

工具的白名单、参数校验和执行仍在 BaseAgent 里，这里只做协议格式转换。
"""
from typing import Any, Dict, Iterable, List

from agents.tools import AgentToolSpec


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
