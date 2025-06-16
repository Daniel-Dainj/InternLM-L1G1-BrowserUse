from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
INTERN_API_KEY = os.getenv("INTERN_API_KEY")
INTERN_ENDPOINT = os.getenv("INTERN_ENDPOINT")
client = OpenAI(
    api_key=INTERN_API_KEY,
    base_url=INTERN_ENDPOINT,
)

chat_rsp = client.chat.completions.create(
    model="internvl-latest",
    messages=[
        {
            "role": "user",  # role 支持 user/assistant/system/tool
            "content": [
                {
                    "type": "text",  # 支持 text/image_url
                    "text": "描述一下这张图片",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.apple.com.cn/mac/college-students/images/campaign/hero_mac_student__b14mk6idt8xe_xlarge.jpg"  # 支持互联网公开可访问的图片 url 或图片的 base64 编码
                    },
                },
            ],
        },
    ],
    stream=False,
    temperature=0.8,  # 控制回复的创造性
    max_tokens=500,  # 限制回复长度
)

for choice in chat_rsp.choices:
    print(choice.message.content)

# 若使用流式调用(stream=True)，则使用下面这段代码
# for chunk in chat_rsp:
#     print(chunk.choices[0].delta.content)
