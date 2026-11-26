# ============================================================
# LLMClient：把"发一次请求"封装成一个类
# 第 2 步：输入 list[Message]，返回 str
# 第 3 步（你的任务）：改成返回 ChatResponse
# ============================================================

import time
import uuid
import httpx
import logging

from .config import Settings, get_settings
from .logging_config import RequestIdAdapter
from .models import Message, Usage, ChatResponse
from .errors import LLMAuthError, LLMBadResponseError, LLMError, LLMRateLimitError, LLMServerError, LLMTimeoutError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

RETRYABLE_ERRORS = (LLMRateLimitError, LLMServerError, LLMTimeoutError)

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        # 注意：get_settings 后面有括号，表示调用函数
        self.settings = settings or get_settings()

        # 没传 http 就自己建一个；传了就用传进来的（测试时用）
        self._http = http or httpx.Client(timeout=self.settings.request_timeout)

    def close(self) -> None:
        """释放连接。"""
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
    
    # 前面有_,约定为内部使用的方法
    def _post_once(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
    ) -> httpx.Response:
        """只发一次请求；失败时抛自定义异常，交给重试层判断"""
        # try 尝试执行 尝试执行可能出错的代码
        try:
            # self._http保存了一个HTTPX客户端对象，可以通过它发送请求
            # post通常用于向服务器提交数据，如提交用户的问题，请求大模型生成回答
            # 创建研究任务，提交数据库记录
            # 把body作为json请求数据，HTTPX把字典转化为json格式再发送给服务器
            resp = self._http.post(url, headers=headers, json=body)
            # resp是一个变量,用来保存HTTP请求返回的响应对象
        except httpx.TimeoutException as e:
            # except 捕获指定类型的异常并处理，如果发生超时异常，就执行这里
            # 网络层超时->自己的异常；它在 RETRYABLE_ERRORS里，所以会被重试
            raise LLMTimeoutError(f"请求超时：{url}") from e
        # 把捕获到的异常对象保存到变量 e 中
        #  raise 主动抛出异常
        # from 建立异常之间的因果关系，原始异常：HTTPX的超时异常，新异常：项目自己的LLMTimeoutError
        # 新异常是由原始异常引起的，可以通过异常回溯查看原始异常，对调试很有帮助
    
        
        if resp.status_code in (401, 403):
            raise LLMAuthError(f"认证失败（{resp.status_code}）, 检查API KEY")
        
        if resp.status_code == 429:
            raise LLMRateLimitError(f"请求太频繁（429）：{resp.text[:200]}")
        
        if resp.status_code >= 500:
            raise LLMServerError(f"服务端错误（{resp.status_code}）：{resp.text[:200]}")
        
        if resp.status_code != 200:
            raise LLMError(f"请求失败（{resp.status_code}）：{resp.text[:200]}")
        
        return resp
    
    def _post_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        log: RequestIdAdapter,
    ) -> httpx.Response:
        # 重试控制层,遇到错误怎么办,负责决定是否重试,重试几次,等待多久
        # 关心遇到错误后怎么办
        # tenacity 每次"失败后,等待前"会调用这个函数
        # retry_state 里有:第几次尝试,下次等多久,这次抛了什么异常
        def _on_retry(retry_state) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            log.warning(
                "llm_retry",
                extra={
                    "attempt": retry_state.attempt_number,
                    "wait_s": round(getattr(retry_state.next_action, "sleep", 0) or 0, 3),
                    "error": type(exc).__name__ if exc else None,
                },
            )
            
        retryer = Retrying(
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
            stop=stop_after_attempt(self.settings.max_attempts),
            wait=wait_exponential_jitter(
                multiplier=self.settings.retry_initial_wait,
                max=self.settings.retry_max_wait,
            ),
            before_sleep=_on_retry,
            reraise=True,
        )
        
        return retryer(self._post_once, url, headers, body)
    
    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> ChatResponse:
        request_id = uuid.uuid4().hex[:8]
        log = RequestIdAdapter(logger, {"request_id": request_id})
        
        started = time.perf_counter()
        # 1. Message 对象 -> 字典列表
        payload = [m.model_dump() for m in messages]

        # 2. 地址和请求头
        url = f"{self.settings.llm_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        # 3. 请求体
        body = {
            "model": self.settings.llm_model,
            "messages": payload,
            "temperature": temperature,
        }

        try:
            resp = self._post_with_retry(url, headers, body, log)
        except LLMError as e:
            log.error(
                "llm_request_failed",
                extra={
                    "error": type(e).__name__,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            raise
        
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMBadResponseError(f"返回格式异常：{resp.text[:200]}") from e
        
        usage_data = data.get("usage",{})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )
        
        latency_ms = int((time.perf_counter() - started) * 1000)
        
        logger.info(
            "llm_request_success",
            extra={
                "request_id": request_id,
                "model": self.settings.llm_model,
                "latency_ms": latency_ms,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        )
        
        return ChatResponse(
            content=content,
            model=self.settings.llm_model,
            usage=usage,
            latency_ms=latency_ms,
            request_id=request_id,
        )
