# ======================================
# LLMClient 的单元测试
# 原则：不联网、不花钱、不依赖真实 Key
# 手段：用httpx.MockTransport 拦截请求，返回构造好的响应
# ======================================

import httpx
import pytest

from ai_agent_lab.errors import LLMAuthError, LLMBadResponseError, LLMRateLimitError, LLMTimeoutError
from ai_agent_lab.client import LLMClient
from ai_agent_lab.config import Settings
from ai_agent_lab.models import Message


def make_settings() -> Settings:
    return Settings(
        llm_api_key="sk-test",
        llm_base_url="https://api.test",
        llm_model="test-model",
        request_timeout=5.0,
    )
    
def make_client(handler) -> LLMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return LLMClient(settings=make_settings(), http=http)

def test_chat_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message":{"content":"你好"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            }
        )
    client = make_client(handler)
    result = client.chat([Message(role="user", content="hi")])

    assert result.content == "你好"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 8

def test_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": "invalid key"}
        )
        
    client = make_client(handler)
    
    with pytest.raises(LLMAuthError, match="401"):
        client.chat([Message(role="user", content="hi")])
        
def test_rate_limit_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "too many request"})
    
    client = make_client(handler)
    
    with pytest.raises(LLMRateLimitError, match="429"):
        client.chat([Message(role="user", content="hi")])

def test_time_out_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = make_client(handler)
    
    with pytest.raises(LLMTimeoutError):
        client.chat([Message(role="user", content="hi")])
        
def test_bad_json_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")
    
    client = make_client(handler)
    
    with pytest.raises(LLMBadResponseError):
        client.chat([Message(role="user", content="hi")])
