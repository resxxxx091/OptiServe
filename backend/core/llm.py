"""全仓库唯一的 LLM 入口。

对外只暴露两件事：拿一个 ChatDeepSeek（LLMProvider.chat_model）和从返回消息里取正文
（message_text）。组件各自持有一个 LLMProvider，参数相同的调用复用同一个模型实例。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class LLMProvider:
    """按 (model, temperature, max_tokens, timeout) 缓存 ChatDeepSeek 实例。

    ChatDeepSeek 本身是无状态的 Runnable，但构造时会建 openai 异步客户端，
    所以同一组参数只建一次；temperature / max_tokens 属于调用参数而不是会话状态，
    放进缓存键才能保证「换个 max_tokens 就是换个客户端」不会被静默复用。
    """

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
