# ai-agent-lab

可测试、可观测的 LLM 客户端，作为Content Agent的工程底座。

[![CI](https://giuhub.com/lyxcnnnn-coder/ai-agent-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/lyxcnnnn-coder/ai-agent-lab/actions)

## 现在能做什么

- 命令行多轮对话（`chat.py`）
- 结构化响应：`content / model / usage / latency_ms / request_id`
- 错误分类：401、 429、 5xx、 超时、 坏 JSON 分别抛不同异常
- 有限重试：只重试 429 / 5xx / 超时 ，最多 3 次， 指数退避
- JSON 结构化日志：一次请求的所有重试共用同一个 `request_id`
- 配置集中校验：缺 `LLM_API_KEY` 启动即失败

## 快速开始

要求：Python 3.10+ 、[uv](https://docs.astral.sh/uv/)

```powershell
git clone https://github.com/lyxcnnnn-coder/ai-agent-lab.git
cd ai-agent-lab
uv sync
Copy-Item .env.example .env
# 编辑 .env, 填入你的 LLM_API_KEY
uv run chat.py