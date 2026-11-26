# ==========================================
# 作用:把日志输出成JSON格式,一行一条
# 好处:机器能解析,方便以后接 Langfuse/ELK
# ==========================================

import json
import logging
import sys
from datetime import datetime, timezone

class JsonFormatter(logging.Formatter):
    EXTRA_FIELDS = (
        "request_id",
        "model",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "attempt",
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
