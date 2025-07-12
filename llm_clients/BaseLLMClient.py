import openai


class BaseLLMClient():
    def __init__(self):
        self.client = openai.OpenAI(
            api_key="sk-ZhouYanZhen_xw0k51BG2hHR",
            base_url="https://chat.noc.pku.edu.cn/v1"
        )
        self.model = None
        self.temperature = 0.0
        self.show = False

    def _chat_with_messages(self, messages):
        response_full = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        response_content = response_full.choices[0].message.content
        if self.show:
            print(response_content)
        return response_content
