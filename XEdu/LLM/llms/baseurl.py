from .base import BaseLLM

class BaseURL(BaseLLM):
    def __init__(self, api_key, base_url,model):
        super().__init__(api_key)
        self.model = model
        self.base_url = base_url