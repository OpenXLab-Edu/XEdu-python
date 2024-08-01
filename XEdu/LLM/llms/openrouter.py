from .base import BaseLLM

class Openrouter(BaseLLM):
    def __init__(self, api_key, model="mistralai/mistral-7b-instruct:free"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
  