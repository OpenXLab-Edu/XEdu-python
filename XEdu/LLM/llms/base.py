import requests
import json


class BaseLLM:
    def __init__(self, api_key, model=None):
        self.api_key = api_key
        self.model = model
        self.base_url = None


    def inference(self, message,stream=False,**kwargs):
        if stream:
            return self._stream_infer(message,**kwargs)
        else:
            return self._infer(message,**kwargs)

    def _set_message(self, messages,stream=True):
        if stream:
            send_messages = {
                "model": self.model, # Optional
                "messages": messages,
                "stream":True
            }
        else:
            send_messages = {
                "model": self.model, # Optional
                "messages": messages,
            }
        return send_messages

    def _set_url(self):
        return self.base_url + "/chat/completions"
    def _stream_infer(self, message,**kwargs):
        """
        实时推理，返回推理结果
        """
        if isinstance(message,str):
            messages = [{"role": "user", "content": message}]
        else:
            messages = message
        senf_messages = self._set_message(messages,stream=True)
        response = requests.post(
            # url=self.base_url+"/chat/completions",
            url=self._set_url(),
            headers=self._set_headers(stream=True),
            data=json.dumps({**senf_messages,**kwargs}),
            stream=True,
        )
        return self._analyse_stream_response(response)

    def _analyse_stream_response(self, response):
        """
        解析推理结果
        """
        for i in response.iter_lines():
            text = i.decode("UTF-8")
            if text.startswith("data: ") and not text.endswith("[DONE]"):
                a = json.loads(text[6:])
                if a["choices"][0]["finish_reason"] != "stop":
                    content = a["choices"][0]["delta"]["content"]
                    if content is not None:
                        yield content

    def _set_content(self, response):
        return response.json()["choices"][0]["message"]["content"]

    def _set_headers(self,stream=False):
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            "Authorization": f"Bearer {self.api_key}",
        }
        return headers
    
    def _infer(self, message,**kwargs):
        """
        单轮推理，返回推理结果
        """
        if isinstance(message,str):
            messages = [{"role": "user", "content": message}]
        else:
            messages = message
        # send_messages = {
        #     "model": self.model, # Optional
        #     "messages": messages,
        # }
        send_messages = self._set_message(messages,stream=False)
        response = requests.post(
        # url=self.base_url+"/chat/completions",
            url=self._set_url(),
            headers=self._set_headers(),
            data=json.dumps({**send_messages,**kwargs})
        )
        try:
            content = self._set_content(response)
        except Exception as e:
            content = response.json()
        return content

    def list_models(self):
        url = self.base_url + "/models"
        payload={}
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer {}'.format(self.api_key),
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        id_list = []
        try:
            for i in response.json()['data']:
                id_list.append(i['id'])
        except:
            raise Exception(response.json())
        return id_list