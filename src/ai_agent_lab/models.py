# ====================================
# 作用：定义“对话”相关的数据结构
# 好处：函数签名从 list[dict] 变成 list[Message]
#       传错字段时 IDE 和 pydantic 会立刻报错
# ====================================

from typing import Literal

from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    latency_ms: int
    request_id: str
    