# ==========================================
# 作用:把日志输出成JSON格式,一行一条
# 好处:机器能解析,方便以后接 Langfuse/ELK
# ==========================================

import json
import logging
import sys
from datetime import datetime, timezone


class RequestIdAdapter(logging.LoggerAdapter):
    """在每条日志上附加 request_id，同时保留调用时传入的其他 extra。

    为什么需要子类：标准库的 LoggerAdapter.process() 会执行
    `kwargs["extra"] = self.extra`，把本次调用传的 extra 整个覆盖掉。
    结果就是 attempt / wait_s / error 这些字段丢失，只剩 request_id。
    """

    def process(self, msg, kwargs):
        # self.extra             = 构造时传的 {"request_id": ...}
        # kwargs.get("extra", {}) = 本次调用传的 {"attempt": ...}
        # 用字典解包合并，后者覆盖前者同名字段
        merged = {**self.extra, **kwargs.get("extra", {})}
        kwargs["extra"] = merged
        return msg, kwargs


class JsonFormatter(logging.Formatter):
    EXTRA_FIELDS = (
        "request_id",
        "model",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "attempt",
        "wait_s",
        "error",
    )
    
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        
        for key in self.EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        
        return json.dumps(payload, ensure_ascii=False)
    
def configure_logging(level: str = "INFO") -> None:
        # 配置根logger,输出到stdout, JSON格式
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
        
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
