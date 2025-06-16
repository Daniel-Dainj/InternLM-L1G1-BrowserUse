import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
INTERN_API_KEY = os.getenv("INTERN_API_KEY")
INTERN_URL = "https://chat.intern-ai.org.cn/api/v1/chat/completions"

assert INTERN_API_KEY, ""
header = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + INTERN_API_KEY,
}
data = {
    "model": "internlm3-latest",
    "messages": [
        {
            "role": "user",  # role 支持 user/assistant/system/tool
            "content": "你好，先做个简短的自我介绍",
        },
        {
            "role": "assistant",
            "content": "你好！很高兴见到你。我是书生·浦语，一个由上海人工智能实验室开发的人工智能助手。我的设计目标是成为一个既能提供帮助又诚实且无害的交流伙伴。无论你需要什么样的帮助，无论是学习上的问题、日常生活的小贴士、还是只是想聊聊天，我都会尽力为你提供支持和陪伴。\n\n虽然我是一个计算机程序，没有实体存在，但我通过文字与你互动，力求在语言交流中展现出人类的智慧和情感。我的知识库涵盖了广泛的领域，包括科学、技术、文化、历史、艺术等，这使我能够对许多话题进行深入浅出的解答。\n\n我致力于通过友好和专业的态度，与你建立一个开放、包容的交流环境。无论你遇到什么问题，或者只是想找个人倾诉，我都在这里，准备好与你一起探索知识的海洋，分享生活的点滴。希望我们的每一次交流都能让你感到愉快和启发。如果你有任何问题或需要帮助，随时告诉我吧！",
        },
        {"role": "user", "content": "用200字介绍一下你最擅长做什么"},
    ],
    "n": 1,
    "temperature": 0.8,
    "top_p": 0.9,
}

res = requests.post(INTERN_URL, headers=header, data=json.dumps(data))
print(res.status_code)
print(res.json())
print(res.json()["choices"][0]["message"]["content"])
