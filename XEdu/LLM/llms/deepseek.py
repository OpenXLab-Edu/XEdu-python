from .base import BaseLLM

class Deepseek(BaseLLM):
    def __init__(self, api_key, model="deepseek-chat"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.deepseek.com"
