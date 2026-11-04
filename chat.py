import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")

if not API_KEY:
    raise SystemExit("没有读到 LLM_API_KEY.检查 1) .env是否在项目根目录 2) 变量名是否正确")

def chat(messages: list[dict], temperature: float = 0.7) -> str:
    url = f"{BASE_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    
    resp = httpx.post(url, headers=headers, json=body, timeout=30)

    if resp.status_code != 200:
        print("请求失败，HTTP状态码：", resp.status_code)
        print("服务器返回：", resp.text)
        resp.raise_for_status()
        
    data = resp.json()  
    # data 为字典类型
    
    usage = data.get("usage", {})
    print(f"[token] 输入 {usage.get('prompt_tokens')}, 输出 {usage.get('completion_tokens')}")

    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    history = []
    
    print("输入exit退出")
    while True:
        user_input = input("你： ")
        
        if user_input.strip().lower() in ("exit", "quit"):
            print("再见")
            break
        
        history.append({"role": "user", "content": user_input})
        
        reply = chat(history)
        print(f"AI: {reply}")
        
        history.append({"role": "assistant", "content": reply})
    
    