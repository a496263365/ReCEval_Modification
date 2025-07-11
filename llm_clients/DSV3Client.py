from llm_clients.BaseLLMClient import BaseLLMClient


class DSV3Client(BaseLLMClient):
    def __init__(self):
        super(DSV3Client, self).__init__()
        self.model = "deepseek-v3"
        self.temperature = 0.0
        pass

    def __chat_with_only_prompt(self,prompt):
        response_content = self._chat_with_messages([
            {'role':'user', 'content': prompt},
        ])
        return response_content

    def chat(self,prompt):
        return self.__chat_with_only_prompt(prompt)


