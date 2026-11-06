from ai_agent_lab.client import LLMClient
from ai_agent_lab.models import Message


def main() -> None:
    client = LLMClient()
    history: list[Message] = []

    print("输入 exit 退出")
    while True:
        user_input = input("你： ")
        if user_input.strip().lower() in ("exit", "quit"):
            print("再见")
            break

        history.append(Message(role="user", content=user_input))
        reply = client.chat(history)
        print(f"AI：{reply}")
        history.append(Message(role="assistant", content=reply))


if __name__ == "__main__":
    main()
