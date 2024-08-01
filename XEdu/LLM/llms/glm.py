from .base import BaseLLM
import requests
import json
class GLM(BaseLLM):
    def __init__(self, api_key, model="glm-4"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"

    def _stream_infer(self, message,**kwargs):
        """
        实时推理，返回推理结果
        """
        if isinstance(message,str):
            messages = [{"role": "user", "content": message}]
        else:
            messages = message
        senf_messages = {
            "model": self.model, # Optional
            "messages": messages,
            "stream":True
        }
        response = requests.post(
            url=self.base_url+"/chat/completions",
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps({**senf_messages,**kwargs}),
            stream=True,
        )
        for i in response.iter_lines():
            text = i.decode("UTF-8")
            if text.startswith("data: ") and not text.endswith("[DONE]"):
                a = json.loads(text[6:])
                content = a["choices"][0]["delta"]["content"]
                if content is not None:
                    yield content

    def list_models(self):
        # glm-4-0520、glm-4 、glm-4-air、glm-4-airx、 glm-4-flash \glm-4v ,glm-3-turbo
        id_list = ["glm-4","glm-3-turbo"]
        return id_list