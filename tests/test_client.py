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
        max_attempts=3,
        retry_initial_wait=0.001,
        retry_max_wait=0.001,
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
    calls = 0
    
# 模拟服务器返回401的情况    
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401,
            json={"error": "invalid key"}
        )
        
    client = make_client(handler)

    # with 是Python的上下文管理器语法,用于管理某一段代码执行前后的资源或状态
    # with:进入一个受管理的代码块,执行完毕后自动完成相应的清理或处理工作
    # 这里with 用于管理一次预期异常的捕获工作
    # 我预期抛出这个类型的异常,如果抛出LLMAuthError,则测试通过
    with pytest.raises(LLMAuthError, match="401"):
        client.chat([Message(role="user", content="hi")])
        # 用户提交消息,启用一次模型调用
    
    assert calls == 1
        
def test_rate_limit_error() -> None:
    calls = 0
    
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": "too many request"})
    
    client = make_client(handler)
    
    with pytest.raises(LLMRateLimitError, match="429"):
        client.chat([Message(role="user", content="hi")])
        
    assert calls == 3

def test_time_out_error() -> None:
    calls = 0
    
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.TimeoutException("timed out", request=request)

    client = make_client(handler)
    
    with pytest.raises(LLMTimeoutError):
        client.chat([Message(role="user", content="hi")])
    
    assert calls == 3
        
def test_bad_json_error() -> None:
    calls = 0
    
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"not json")
    
    client = make_client(handler)
    
    with pytest.raises(LLMBadResponseError):
        client.chat([Message(role="user", content="hi")])
    
    assert calls == 1

def test_chat_retries_rate_limit_twice_then_succeed() -> None:
    # 记录客户端实际发出的 HTTP 请求次数。
    attempts = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        # 每次收到请求都更新外层计数，第一次请求时 attempts 变成 1。
        nonlocal attempts
        attempts += 1

        # 模拟服务端前两次限流，第三次恢复并返回正文。
        if attempts <= 2:
            return httpx.Response(429)

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "重试后成功"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    # 退回到 4 个空格，和上面的 def handle_request 对齐。
    # 这两行属于测试函数，调用 chat() 后才会触发模拟响应。
    client = make_client(handle_request)
    response = client.chat([Message(role="user", content="你好")])

    # 验证前两次限流后继续尝试，且第三次成功后结束。
    assert attempts == 3
    # 验证成功响应经过解析，返回了第三次请求的正文。
    assert response.content == "重试后成功"