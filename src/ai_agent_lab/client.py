# ============================================================
# LLMClient：把"发一次请求"封装成一个类
# 第 2 步：输入 list[Message]，返回 str
# 第 3 步（你的任务）：改成返回 ChatResponse
# ============================================================

import httpx

from .config import Settings, get_settings
from .models import Message


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

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> str:
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

        # 4. 用 self._http.post 发请求（不是 httpx.Client）
        resp = self._http.post(url, headers=headers, json=body)

        # 5. 非 2xx 抛异常
        resp.raise_for_status()

        # 6. 解析并返回文字
        data = resp.json()
        return data["choices"][0]["message"]["content"]
