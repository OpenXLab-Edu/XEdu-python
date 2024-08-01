from .base import BaseLLM

class Moonshot(BaseLLM):
    def __init__(self, api_key, model="moonshot-v1-8k"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.moonshot.cn/v1"

