import requests
import json
from .base import BaseLLM
 

# class ERNIE:
# 实现一个ERNIE类，通过model参数选择以下不同的类，默认使用ERNIE_speed_128k模型
class ERNIE:
    def __new__(cls, api_key, secret_key,model="ERNIE_speed_8k"):
        model = "ERNIE_speed_8k" if model not in ["ERNIE_speed_128k","ERNIE_speed_8k","ERNIE_lite_8k_0922","ERNIE_lite_8k_0308"] else model
        if model == "ERNIE_speed_128k":
            return ERNIE_speed_128k(api_key, secret_key)
        elif model == "ERNIE_speed_8k":
            return ERNIE_speed(api_key, secret_key)
        elif model == "ERNIE_lite_8k_0922":
            return ERNIE_lite_8k_0922(api_key, secret_key)
        elif model == "ERNIE_lite_8k_0308":
            return ERNIE_lite_8k_0308(api_key, secret_key)
        else:
            raise ValueError("Invalid model name")


class ERNIE_speed(BaseLLM):
    def __init__(self, api_key, secret_key):
        super().__init__(api_key, secret_key)
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = self.get_access_token()
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"


    def get_access_token(self):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=" + self.api_key + "&client_secret=" + self.secret_key
        payload = json.dumps("")    
        headers = {
            'Content-Type': 'application/json',  
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        try:
            access_token = response.json()["access_token"]
        except:
            raise Exception(response.json())
        return access_token



    def _stream_infer(self, message,**kwargs):
        """
        实时推理，返回推理结果
        """
        if isinstance(message,str):
            messages = [{"role": "user", "content": message}]
        else:
            messages = message
        senf_messages = {
            "messages": messages,
            "stream":True
        }
        url = self.url + "?access_token=" + self.access_token
        payload = json.dumps({**senf_messages,**kwargs})
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload,stream=True)
        for i in response.iter_lines():
            if i:
                a = json.loads(i.decode("UTF-8")[6:])['result']
                yield a


    def _infer(self, message,**kwargs):
        """
        单轮推理，返回推理结果
        """
        url = self.url + "?access_token=" + self.access_token
        if isinstance(message,str):
            messages = [{"role": "user", "content": message}]
        else:
            messages = message
        senf_messages = {
            "messages": messages,
        }
        payload = json.dumps({**senf_messages,**kwargs})

        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        try:
            content = json.loads(response.text)['result']
        except:
            content = response.json()
        return content

    def list_models(self):
        return ["ERNIE_speed_8k","ERNIE_speed_128k","ERNIE_lite_8k_0308"]

class ERNIE_speed_128k(ERNIE_speed):
    def __init__(self, api_key, secret_key):
        super().__init__(api_key, secret_key)
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k"


class ERNIE_lite_8k_0308(ERNIE_speed):
    def __init__(self, api_key, secret_key):
        super().__init__(api_key, secret_key)
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k"

    