"""唯一的 LLM 入口。"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class LLMProvider:
    """唯一的 LLM 提供者。"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "") or DEFAULT_BASE_URL)
        self._cache: Dict[Tuple[Any, ...], ChatDeepSeek] = {}

    def chat_model(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> ChatDeepSeek:
        key = (model, temperature, max_tokens, timeout)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        kwargs: Dict[str, Any] = {"model": model, "api_key": self.api_key, "base_url": self.base_url}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["timeout"] = timeout

        created = ChatDeepSeek(**kwargs)
        self._cache[key] = created
        return created


def message_text(message: BaseMessage) -> str:
    """取消息正文。DeepSeek 返回纯字符串， reasoning 内容在 additional_kwargs 里，不计入。"""
    content = message.content
    if isinstance(content, str):
        return content

    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") in (None, "text"):
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)
