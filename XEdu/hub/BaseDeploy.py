# import BaseDT
from .BaseDT.data import ModelData, ImageData
import numpy as np
from typing import Union, List
import os
import cv2
import matplotlib.pyplot as plt
from .BaseDT.utils import mmpose_preprocess, mmpose_postprocess

def get_host_ip():
    try:
        import socket
    except ImportError:
        print("The library 'socket' does not exist. Please use 'pip3 install socket' to install library.")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


class _BaseDeploy:

    def __init__(self, model_path, device='cpu', backend='ort'):
        assert isinstance(model_path, str)
        assert backend in ['ort']
        # 分类 onnx推理框架、api层（siot、fastapi）
        self.backend = backend
        self.device = device
        self.model_path = None
        self.model = None
        self.info = None
        self.class_names = None
        self.backbone = None
        self.codebase = None
        self.ip_addr = None
        self.siot = None
        self.score = None
        # 用以判断siot一次至显示或回传一张图片
        self.is_show = False

        if backend == 'ort':
            self.model_path = model_path
            self.model = self._use_ort_backend()
            self.device = device
            self.info = ModelData(model_path)
            self.class_names = self.info.get_labels()
            self.backbone = self.info.get_modelname()
            self.codebase = self.info.get_codebase()
        elif backend == 'siot':
            self.ip_addr = model_path
            self.run_siot(self.ip_addr, 'upload')
        elif backend == 'fastapi':
            self.ip_addr = model_path

    def _use_ort_backend(self):
        import onnxruntime as ort
        sess = ort.InferenceSession(self.model_path, None)
        return sess

    def inference(self, 
                input_data: Union[str, List[str], ImageData, List[ImageData]], 
                show: bool = False, 
                get_img: str = None,
                score: float = None,
                show_path: bool = False):
        if self.backend == 'siot':
            try:
                import base64
            except ImportError:
                print("The library 'base64' does not exist. Please use 'pip3 install base64' to install library.")
            self.is_show = True
            IOT_Topic  = 'XEdu/model_infer'
            IOT_Result_Topic = 'XEdu/model_result'
            print('Inference using a remote siot server.')
            def rec_cb(client, userdata, msg):
                if not self.is_show:
                    return
                payload = msg.payload.decode()
                decoded_data = base64.b64decode(payload)
                decoded_image = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
                plt.imshow(image_rgb)
                plt.axis('off')
                plt.show()
                self.is_show = False
            if isinstance(input_data, str):
                img = cv2.imread(input_data)
                _, encoded_image = cv2.imencode('.jpg', img)
                encoded_string = base64.b64encode(encoded_image).decode('utf-8')
                self.siot.publish(IOT_Topic, encoded_string)
                self.siot.subscribe(IOT_Result_Topic, rec_cb)
        if self.backend == 'ort':
            if self.backbone == '':
                input_shape = self.model.get_inputs()[0].shape
                if input_shape[0] == 'unk__606':
                    img_size = tuple(input_shape[1:3])
                else:
                    img_size = tuple(input_shape[-2:])
                if isinstance(input_data, str):
                    if os.path.isfile(input_data):
                        input_data = ImageData(input_data, size = img_size)
                    elif os.path.isdir(input_data):
                        input_data = [ImageData(os.path.join(input_data, f), size = img_size) for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if isinstance(input_data, np.ndarray):
                    input_data = ImageData(input_data, size = img_size)
            else:
                if isinstance(input_data, str):
                    if os.path.isfile(input_data):
                        input_data = ImageData(input_data, backbone = self.backbone)
                    elif os.path.isdir(input_data):
                        input_data = [ImageData(os.path.join(input_data, f), backbone = self.backbone) for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if isinstance(input_data, np.ndarray):
                    input_data = ImageData(input_data, backbone = self.backbone)
            if isinstance(input_data, ImageData):
                input_data = [input_data]
            assert isinstance(input_data, list)
            input_name = self.model.get_inputs()[0].name
            output_names = [o.name for o in self.model.get_outputs()]
            results = []
            for dt in input_data:
                assert isinstance(dt, ImageData)
                if self.codebase == 'MMCls':
                    if score is None:
                        score = 0
                    self.score = score
                    pred_onx = self.model.run(output_names, {input_name: dt.to_tensor()})
                    results.append(pred_onx)
                    result = self.print_result(pred_onx, score)
                    if show_path:
                        result['路径'] = dt.data_source
                    #results.append(result)
                    if show:
                        if result['置信度'] >= score:
                            from .BaseDT.plot import show_cls
                            show_cls([dt], [result])
                            print(result)
                        else:
                            print("The accuracy is lower than the preset value, "
                                  "if you want to draw a picture, "
                                  "please set score={:.2f}, or lower.".format(result['置信度']))
                elif self.codebase == 'MMDet':
                    if score is None:
                        score = 0.65
                    self.score = score
                    pred_onx = self.model.run(output_names, {input_name: dt.to_tensor()})
                    pred_onx[0][0] = dt.map_orig_coords(pred_onx[0][0])
                    results.append(pred_onx)
                    result = self.print_result(pred_onx, score)
                    if show_path:
                        for item in result:
                            item['路径'] = dt.data_source
                    if show:
                        from .BaseDT.plot import show_det
                        show_det([dt], [result])        
                        print(result)
                elif self.codebase == 'TFJS':
                    if score is None:
                        score = 0
                    self.score = score
                    import copy
                    copy_dt = copy.deepcopy(dt)
                    input_tensor = copy_dt.to_tensor()
                    input_data_tfjs = np.transpose(input_tensor, (0, 2, 3, 1))
                    output_data = self.model.run(output_names, {input_name: input_data_tfjs})
                    results.append(output_data)
                    result = self.print_result(output_data)
                    if show_path:
                        result['路径'] = dt.data_source
                    if show:
                        if result['置信度'] >= score:
                            from .BaseDT.plot import show_cls
                            show_cls([dt], [result])
                            print(result)
                        else:
                            print("The accuracy is lower than the preset value, "
                                  "if you want to draw a picture, "
                                  "please set score={:.2f}, or lower.".format(result['置信度']))
                elif self.codebase == 'MMPose':
                    h, w = self.model.get_inputs()[0].shape[2:]
                    model_input_size = (w, h)
                    img_shape = dt.raw_value.shape[:2]
                    img = dt.raw_value
                    bbox = np.array([0, 0, img_shape[1], img_shape[0]])
                    resized_img, center, scale = mmpose_preprocess(img, model_input_size)
                    input_tensor = [resized_img.transpose(2, 0, 1)]
                    outputs = self.model.run(output_names, {input_name: input_tensor})
                    keypoints, scores = mmpose_postprocess(outputs, model_input_size, center, scale)
                    results.append([keypoints, scores])
                    result = {'关键点':keypoints, '得分':scores}
                    if show_path:
                        result['路径'] = dt.data_source
                    if show:
                        from .BaseDT.plot import show_pose
                        show_pose([dt], [result])
                        print(result)
                else:
                    input_shape = self.model.get_inputs()[0].shape
                    print('The onnx model is not exported by the XEdu tool.'\
						' BaseDeploy calls BaseDT to adapt the input to {}.'\
						' \'mean\': [123.675, 116.28, 103.53], \'std\': [58.395, 57.12, 57.375], \'normalize\': True'.format(input_shape))
                    if input_shape[0] == 'unk__606':
                        self.codebase = 'TFJS'
                        if score is None:
                            score = 0
                        self.score = score
                        import copy
                        copy_dt = copy.deepcopy(dt)
                        input_tensor = copy_dt.to_tensor()
                        input_data_tfjs = np.transpose(input_tensor, (0, 2, 3, 1))
                        output_data = self.model.run(output_names, {input_name: input_data_tfjs})
                        results.append(output_data)
                        result = self.print_result(output_data)
                        if show_path:
                            result['路径'] = dt.data_source
                        if show:
                            if result['置信度'] >= score:
                                from .BaseDT.plot import show_cls
                                show_cls([dt], [result])
                                print(result)
                            else:
                                print("The accuracy is lower than the preset value, "
                                    "if you want to draw a picture, "
                                    "please set score={:.2f}, or lower.".format(result['置信度']))
                    else:
                        pred_onx = self.model.run(output_names, {input_name: dt.to_tensor()})
                        if len(pred_onx) == 1:
                            # consider as cls
                            self.codebase = 'MMCls'
                            if score is None:
                                score = 0
                            self.score = score
                            results.append(pred_onx)
                            result = self.print_result(pred_onx)
                            if show_path:
                                result['路径'] = dt.data_source
                            if show:
                                if result['置信度'] >= score:
                                    from .BaseDT.plot import show_cls
                                    show_cls([dt], [result])
                                    print(result)
                                else:
                                    print("The accuracy is lower than the preset value, "
                                        "if you want to draw a picture, "
                                        "please set score={:.2f}, or lower.".format(result['置信度']))
                        elif len(pred_onx) == 2:
                            # consider as det
                            self.codebase = 'MMDet'
                            if score is None:
                                score = 0.65
                            self.score = score
                            #pred_onx = self.model.run(output_names, {input_name: dt.to_tensor()})
                            pred_onx[0][0] = dt.map_orig_coords(pred_onx[0][0])
                            results.append(pred_onx)
                            result = self.print_result(pred_onx, score)
                            print(result)
                            if show_path:
                                for item in result:
                                    item['路径'] = dt.data_source
                            if show:
                                from .BaseDT.plot import show_det
                                show_det([dt], [result])
                                print(result)
                        else:
                            return pred_onx
            if get_img == 'pil':
                imgs = self._get_img(input_data, self.print_result(results), score, show)
                results = results[0] if len(results) == 1 else results
                imgs = imgs[0] if len(imgs) == 1 else imgs
                return results, imgs
            elif get_img == 'cv2':
                imgs = self._get_img(input_data, self.print_result(results), score, show)
                for i in range(len(imgs)):
                    imgs[i] = cv2.cvtColor(np.array(imgs[i]), cv2.COLOR_RGB2BGR)
                results = results[0] if len(results) == 1 else results
                imgs = imgs[0] if len(imgs) == 1 else imgs
                return results, imgs
            results = results[0] if len(results) == 1 else results
            return results

    def run_gradio(self):
        try:
            import gradio as gr
        except ImportError:
            print("The library 'gradio' does not exist. Please use 'pip3 install gradio' to install library.")
        def predict(input_img, score):
            dt = ImageData(input_img, backbone=self.backbone)
            result, img = self.inference(dt, score=score, get_img='pil')
            return img, result
        gr_infer = gr.Interface(
            fn=predict,
            inputs=[gr.Image(type="filepath"),
                    gr.Slider(minimum=0, maximum=1, value=0.0, step=0.01, label="Score")],
            outputs = [gr.Image(type="numpy"), "text"]
        )
        gr_infer.launch()

    def run_fastapi(self, 
                    port: int = 1956,
                    mode: str = 'json',
                    score: float = 0.65):
        try:
            from fastapi import FastAPI, UploadFile, File
            import shutil
            import uvicorn
        except ImportError:
            print("The library 'gradio' does not exist. Please use 'pip3 install fastapi shutil uvicorn' to install library.")
        app = FastAPI()
        @app.post('/upload')
        async def upload_file(files: UploadFile = File(...)):
            fileUpload = f"./{files.filename}"
            with open(fileUpload, "wb") as buffer:
                shutil.copyfileobj(files.file, buffer)
                dt = ImageData(fileUpload, backbone=self.backbone)
                results = self.inference(dt, score=score)
                print(results)
                return str(results)
        ip = str(get_host_ip())
        print("EasyAPI的接口已启动，地址为http://"+ ip + ":{}/upload".format(port))
        uvicorn.run(app=app,host="0.0.0.0", port=port, workers=1)

    def run_pywebio(self,
                    port: int = 1956):
        try:
            from pywebio import start_server
            from pywebio.input import file_upload, slider, input_group
            from pywebio.output import put_image, put_text
            from PIL import Image
        except ImportError:
            print("The library 'pywebio' does not exist. Please use 'pip3 install pywebio' to install library.")
        def img_infer():
            info = input_group("", [
                file_upload("选择要进行推理的图片", name='img_file'),
                slider("请设置阈值", min_value=0, max_value=1, step=0.01, value=0.65, name='score')
            ])
            image_data = np.frombuffer(info['img_file']['content'], np.uint8)
            score = info['score']
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            dt = ImageData(img, backbone=self.backbone)
            result, img = self.inference(dt, score=score, get_img='pil')
            img = Image.fromarray(img)
            put_image(img)
            put_text('推理结果如下：', result)
        start_server(img_infer, port=port, debug=True)

    def run_siot(self, ip: str = None,
                mode: str = None):
        if ip is None:
            ip = self.ip_addr
        assert ip is not None, 'Please set ip, such as x.x.x.x'
        assert mode in ['infer', 'upload'], 'Please set mode, the choice is mode=\'upload\' or mode=\'infer\''
        try:
            import siot
            import base64
        except ImportError:
            print("The library 'siot'、'base64' does not exist. Please use 'pip3 install siot base64' to install library.")
        msg = 'Hallo siot, this connection supported by XEdu'
        def sub_cb(client, userdata, msg):
            payload = msg.payload.decode()
            decoded_data = base64.b64decode(payload)
            decoded_image = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_COLOR)
            result, img = self.inference(decoded_image, get_img='cv2')
            print(result)
            IOT_Result_Topic = 'XEdu/model_result'
            _, encoded_image = cv2.imencode('.jpg', img)
            encoded_string = base64.b64encode(encoded_image).decode('utf-8')
            self.siot.publish(IOT_Result_Topic, encoded_string)
        if mode == 'infer':
            SERVER = ip
            CLIENT_ID = "XEdu"
            IOT_Topic  = 'XEdu/model_infer'
            IOT_UserName ='siot'
            IOT_PassWord ='dfrobot'
            siot.init(CLIENT_ID, SERVER, user=IOT_UserName, password=IOT_PassWord)
            siot.connect()
            self.siot = siot
            print(msg)
            siot.subscribe(IOT_Topic, sub_cb)
            siot.loop()
        else:
            SERVER = ip
            CLIENT_ID = "XEdu"
            IOT_Topic  = 'XEdu/model_infer'
            IOT_UserName ='siot'
            IOT_PassWord ='dfrobot'
            siot.init(CLIENT_ID, SERVER, user=IOT_UserName, password=IOT_PassWord)
            siot.connect()
            siot.loop()
            print(msg)
            self.use_siot = True
            self.siot = siot


    def stop_siot(self):
        if self.use_siot:
            self.siot.stop()
            self.siot = None
            print('The siot\'s run has been terminated')
        else:
            print('The siot\'s run has already been terminated')


    def run_flask(self):
        pass

    def _get_img(self, input_data, results, score, show):
        imgs = [] 
        for i, dt in enumerate(input_data):
            if self.codebase == 'MMCls':
                if show == False:
                    if results[i]['置信度'] < score:
                        print("The accuracy is lower than the preset value, "
                          "if you want to draw a picture, "
                          "please set score={:.2f}, or lower.".format(results[i]['置信度']))
                        dt.init_plt()
                    else:
                        from .BaseDT.plot import draw_single_cls
                        draw_single_cls(dt, results[i])
            elif self.codebase == 'MMDet':
                if show == False:
                    from .BaseDT.plot import draw_single_det
                    draw_single_det(dt, results[i])
            elif self.codebase == 'TFJS':
                if show == False:
                    if results[i]['置信度'] < score:
                        print("The accuracy is lower than the preset value, "
                          "if you want to draw a picture, "
                          "please set score={:.2f}, or lower.".format(results[i]['置信度']))
                        dt.init_plt()
                    else:
                        from .BaseDT.plot import draw_single_cls
                        draw_single_cls(dt, results[i])
            elif self.codebase == 'MMPose':
                if show == False:
                    from .BaseDT.plot import draw_single_pose
                    draw_single_pose(dt, results[i])
            imgs.append(dt.get_image())
        return imgs
    
    def diy_inference(self, input_data):
        assert isinstance(input_data, np.ndarray)
        input_name = self.model.get_inputs()[0].name
        output_names = [o.name for o in self.model.get_outputs()]
        pred_onx = self.model.run(output_names, {input_name: input_data})
        return pred_onx

    def print_single_cls(self, pred_onx: list):
        if pred_onx[0].ndim == 1:
            pred_onx[0] = pred_onx[0][np.newaxis, :]
        idx = np.argmax(pred_onx[0], axis=1)[0]
        result = {'标签': idx, '置信度': pred_onx[0][0][idx]}
        if isinstance(self.class_names, list):
            result['预测结果'] = self.class_names[idx]
        return result

    def print_single_det(self, pred_onx: list, score):
        boxes = pred_onx[0][0]
        labels = pred_onx[1][0]
        result = []
        for box, label in zip(boxes, labels):
            if box[4] >= score:
                box_int = box.astype(np.int32)
                tmp_dict = {'标签': label, '置信度': box[4],
                            '坐标': {'x1': box_int[0],
                                     'y1': box_int[1],
                                     'x2': box_int[2],
                                     'y2': box_int[3]}}
                if isinstance(self.class_names, list):
                    tmp_dict['预测结果'] = self.class_names[label]
                result.append(tmp_dict)
        return result

    def print_single_pose(self, pred_onx: list):
        keypoints, scores = pred_onx
        result = {'关键点': keypoints, '得分': scores}
        return result

    def print_result(self, pred_onx: list, score: float = None):
        if self.codebase == 'MMCls':
            if isinstance(pred_onx[0], list):
                result = []
                for item in pred_onx:
                    result.append(self.print_single_cls(item))
            else:
                result = self.print_single_cls(pred_onx)
        elif self.codebase == 'MMDet':
            score = self.score
            if score is None:
               score = 0.65
            if isinstance(pred_onx[0], list):
                result = []
                for item in pred_onx:
                    result.append(self.print_single_det(item, score))
            else:
                result = self.print_single_det(pred_onx, score)
        elif self.codebase == 'TFJS':
            if isinstance(pred_onx[0], list):
                result = []
                for item in pred_onx:
                    result.append(self.print_single_cls(item))
            else:
                result = self.print_single_cls(pred_onx)
        elif self.codebase == 'MMPose':
            if isinstance(pred_onx[0], list):
                result = []
                for item in pred_onx:
                    result.append(self.print_single_pose(item))
            else:
                result = self.print_single_pose(pred_onx)
        else:
            result = pred_onx
        return result
