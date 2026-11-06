# ==========================
# 作用：定义LLM相关异常
# 原则：能重试的、不能重试的，要分开
# ==========================

class LLMError(Exception):
    """所有LLM错误的基类。except LLMError可以一网打尽。"""
    
class LLMConfigError(LLMError):
    """配置缺失或非法，不重试。"""

class LLMAuthError(LLMError):
    """401/403：key无效或无权限。不重试。"""

class LLMRateLimitError(LLMError):
    """429：请求太频繁。可重试。"""

class LLMServerError(LLMError):
    """5xx：服务器端错误。可重试。"""

class LLMTimeoutError(LLMError):
    """连接或读取超时。可重试。"""

class LLMBadResponseError(LLMError):
    """返回不是合法JSON，或缺字段。不重试。"""