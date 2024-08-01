#coding=utf-8

from XEdu.hub import Workflow as wf
import cv2
import numpy as np 


def pose_infer_demo():
    # a = time.time()
    img = 'pose2.jpg' # 指定进行推理的图片路径
    img = cv2.imread(img)

    det = wf(task='det_body')#,checkpoint="checkpoints/bodydetect_l.onnx")
    pose = wf(task='pose_body17')# ,checkpoint="rtmpose-l-19c9d1.onnx")# "rtmpose-m-80e511.onnx") # 实例化mmpose模型
    
    import time
    a = time.time()
    bbox = det.inference(data=img)
    print(time.time()-a)
    for i in bbox:
        result,img = pose.inference(data=img,img_type='cv2',bbox=i) # 在CPU上进行推理
        pose.show(img)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # pose.save(img,"pimg_ou.png")
    
    # result = pose.format_output(lang="zh")
    # print(result)

def video_infer_demo():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("gyt.mp4")
    
    pose = wf(task='pose_body')
    det = wf(task='det_body')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        bboxs = det.inference(data=frame,thr=0.3) # 在CPU上进行推理
        img = frame
        for i in bboxs:
            keypoints,img =pose.inference(data=img,img_type='cv2',bbox=i) # 在CPU上进行推理
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()

def det_infer_demo():
    # a = time.time()
    from XEdu.hub import Workflow as wf
    img = 'hand1.jpeg' # 指定进行推理的图片路径

    det = wf(task='det_hand',checkpoint='det_fire.onnx')

    bboxs,im_ou = det.inference(data=img,img_type='pil',show=False) # 在CPU上进行推理
    # print(bboxs)
    # det.save(im_ou,"im_ou_d.jpg")

    det.format_output(lang="zh")
    # print(result)

def hand_video_demo():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("pose.mp4")

    pose = wf(task='pose_hand21')# ,checkpoint="rtmpose-m-80e511.onnx") # 实例化pose模型

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints,img =pose.inference(data=frame,img_type='cv2') # 在CPU上进行推理
        
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()

def coco_det_demo():
    img = 'cat.jpg' # 指定进行推理的图片路径
    det = wf(task='det_coco')#,checkpoint="checkpoints/cocodetect_l.onnx") # 实例化mmpose模型

    result,img = det.inference(data=img,img_type='pil',thr=0.3) # 在CPU上进行推理
    det.show(img)
    det.save(img,"pimg_ou.png")
    
    # re = det.format_output(lang="zh")

def face_det_demo():
    img = 'banniang.jpg' # 指定进行推理的图片路径
    # img = 'face2.jpeg' # 指定进行推理的图片路径

    det = wf(task='det_face' )
    face = wf(task="pose_face")

    result,img = det.inference(data=img,img_type='cv2',scaleFactor=1.4,minNeighbors=8) # 在CPU上进行推理
    re =  det.format_output(lang="zh")

    det.show(img)
    # det.save(img,"banniang1.jpg")
    # for i in result:
    #     ky,img = face.inference(data=img, img_type="cv2",bbox=i)#,erase=False)
    #     face.show(img)
    

def ocr_demo():
    img = 'ocr.jpg' # 指定进行推理的图片路径
    ocr = wf(task='ocr' )#,checkpoint="rtmdet-coco.onnx") # 实例化mmpose模型

    result,img = ocr.inference(data=img,img_type='pil',show=True) # 在CPU上进行推理
    print(result)
    ocr.show(img)
    ocr.save(img,"img_ou.png")
    
    re = ocr.format_output(lang="zh")

def mmedu_demo():
    mm = wf(task='mmedu',checkpoint="det_fire.onnx")
    result, img = mm.inference(data='fire.jpg',img_type='pil',thr=0.6)
    # mm.show(img)
    # print(result)
    re = mm.format_output(lang="zh")

    # mm = wf(task='mmedu',checkpoint="convert_model.onnx")
    # result, img = mm.inference(data='fire.jpg',img_type='pil',thr=0.6)
    # # mm.show(img)
    # # print(result)
    # re = mm.format_output(lang="zh")

def basenn_demo():
    nn = wf(task='basenn',checkpoint="checkpoints/basenn.onnx") # iris act 
    result,img = nn.inference(data='6.jpg',img_type='cv2')
    re = nn.format_output(lang="zh")
    nn.show(img)

def hand_det_demo():
    img = 'hand4.jpeg' # 指定进行推理的图片路径
    det = wf(task='det_hand') # 实例化mmpose模型
    hand = wf(task='pose_hand')

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result,img = det.inference(data=frame,img_type='cv2',thr=0.3) # 在CPU上进行推理
        for i in result:
            ky, img = hand.inference(data=img, img_type='cv2',bbox=i)
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()
    # det.save(img,"pimg_ou.png")
    # re = det.format_output(lang="zh")

def custom_demo():
    def pre(path):
        img = cv2.imread(path) 
        img = img.astype(np.float32)
        img = np.expand_dims(img,0) # 增加batch维
        img = np.transpose(img, (0,3,1,2)) # [batch,channel,width,height]
        return img
    
    def post(res,data):
        
        idx = np.argmax(res[0])
        # print(xxx)
        return idx, res[0][0][idx]

    img_path = "ele.jpg"
    mm = wf(task='custom',checkpoint="mobileone-s3-46652f.onnx") # iris act 
    result = mm.inference(data=img_path,preprocess=pre,postprocess=post)
    print(result)

def cls_demo():
    cls = wf(task='cls_imagnet')#checkpoint="mobileone-s3-46652f.onnx") # iris act 
    img = cv2.imread('ele.jpg')
    result,img = cls.inference(data="ele.jpg",img_type="pil")
    # cls.show(img)
    # print(result)
    re = cls.format_output(lang="zh")

def baseml_demo():
    # ml = wf(task='baseml',checkpoint="baseml_ckpt.pkl") # iris act 
    ml = wf(task='baseml',checkpoint="polynormial.pkl") # iris act 

    # result = ml.inference(data=[[1,0.5,-1,0]])
    # result = ml.inference(data=[1,0.5,-1,0])
    result = ml.inference(data=np.array([[0.5,0,1,1]]))


    re = ml.format_output(lang="zh")
def style_demo():
    for i in range(5):
        style = wf(task='gen_style' ,style="ele.jpg") # iris act 
        result = style.inference(data="ele.jpg" ,img_type='cv2')
        style.show(result)
        style.save(result,"ele_{}.jpg".format(i))
    # # 打开摄像头，实时风格迁移
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     style = random.choice(styles)
    #     result,img = style.inference(data=frame,img_type='cv2')
    #     cv2.imshow('video', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

def qa_demo():
    qa = wf(task='nlp_qa')# ,checkpoint="checkpoints/bertsquad-8s.onnx")  # /home/user/下载/chatgpt_项目/bertsquad-12-int8.onnx
    # qa = wf(task='nlp_qa',checkpoint="/home/user/下载/chatgpt_项目/bertsquad-12-int8.onnx")
    # 整理以下字符串
    context_glm = """In life, we often encounter challenges that make us question our abilities and motivate us to give up. However, the secret to success lies in persistence. Only by persevering through difficult times can we unlock our true potential and achieve our goals.
Michael Jordan, one of the greatest basketball players of all time, was once cut from his high school team. Instead of giving up, he practiced relentlessly and eventually earned a spot on the team. His perseverance and determination are what made him a legendary athlete.
Another example of persistence is Elon Musk, the CEO of SpaceX and Tesla. He has faced countless setbacks in his entrepreneurial ventures, but he hasn't let them stop him. His vision for the future and relentless pursuit of innovation have made him a global sensation.
In life, we will all face obstacles. It's how we respond to these challenges that defines us. Instead of surrendering to defeat, let's embrace the power of persistence and strive for greatness. Remember, success is not a destination but a journey, and the only way to achieve it is to keep moving forward.
"""
    q11 = "Who is Michael Jordan?"
    q22 = "Who is Elon Musk?"
    context = "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space"
    q1 = "By what year many Silicon Valley businesses were choosing the Moscone Center?"
    q2 = "how may votes did the ballot measure need?"
    q3 = "how many square feet did the South Hall add?"
    qa.load_context(context)
    result = qa.inference(data=q11, context=context_glm)
    result = qa.inference(data=q2)
    result = qa.inference(data=q22, context=context_glm)
    result = qa.inference(data=q3)

    print(result)
    res = qa.format_output(lang="en",show_context=False)
    # result, r = qa.inference(data=q4)
    # print(result)

def sim():
    from onnxsim import simplify
    import onnx
    model = onnx.load("udnie-9.onnx")
    model,_ = simplify(model)
    onnx.save(model, "gen_style_udnie.onnx")

def drive_demo():
    from XEdu.hub import Workflow as wf
    drp = wf(task='drive_perception')
    # result = drp.inference(data="demo.jpg")

    result,image = drp.inference(data="cat1.jpg",img_type='pil')
    drp.format_output(lang="zh",isprint=False)

    drp.show(image)
    # drp.save(image,"demo_ou.jpg")

def embedding_demo():
    # 导入依赖库
    from XEdu.hub import Workflow as wf
    from XEdu.utils import get_similarity,visualize_similarity,visualize_probability
    # 实例化图像嵌入模型
    img_emb = wf(task='embedding_image')
    # 实例化文本嵌入模型
    text_emb = wf(task='embedding_text')
    images = ["cat.png","cat1.jpg","cat2.jpg","cat3.jpg","ele.jpg","dog.jpg"]
    texts_zh = ["猫", "狗"]
    texts_en = ["cat", "dog","room","elephant"]

    # image_embeddings = img_emb.inference(data=images)
    # text_zh_embeddings = text_emb.inference(data=texts_zh)
    text_en_embeddings = text_emb.inference(data=texts_en)
    print(text_en_embeddings.shape)
    # # 图像 - 文本相似度
    # logits = get_similarity(image_embeddings,text_en_embeddings,use_softmax=False)
    # visualize_similarity(logits, images, texts_en,figsize=(10,20),)
    # visualize_probability(logits, images, texts_en,topk=6,figsize=(10,10))  

    # 文本 - 文本相似度
    logits = get_similarity(text_en_embeddings,text_en_embeddings ,method='cosine')
    # visualize_similarity(logits, texts_en, texts_en)

    # # 文本 - 图像相似度
    # logits = get_similarity(text_en_embeddings,image_embeddings,method='cosine',use_softmax=False)
    # visualize_similarity(logits, texts_en, images)

    # 图像 - 图像相似度
    # logits = get_similarity(image_embeddings,images_embeddings,method='cosine',use_softmax=False)
    # visualize_similarity(logits, images, imagess)

    print("cosine", logits)


def openvino_demo():
    import openvino as ov 
    from openvino.runtime import Core
    ie = Core()
    devices = ie.available_devices

    for device in devices:
        device_name = ie.get_property(device,'FULL_DEVICE_NAME')
        print(f'{device}:{device_name}')

    from openvino.runtime import Core

    ie =  Core()
    model_xml = './checkpoints/body17.onnx'

    model  = ie.read_model(model=model_xml)
    compiled_model = ie.compile_model(model=model,device_name='CPU')


    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    import cv2
    import numpy as np

    image_filename = "pose1.jpg"
    image = cv2.imread(image_filename)
    image.shape

    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = input_layer.shape
    # OpenCV resize expects the destination size as (width, height).
    resized_image = cv2.resize(src=image, dsize=(W, H))
    resized_image.shape

    input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
    input_data.shape


    # for single input models only
    result = compiled_model(input_data)[output_layer]

    # for multiple inputs in a list
    result = compiled_model([input_data])[output_layer]

    # or using a dictionary, where the key is input tensor name or index
    result = compiled_model({input_layer.any_name: input_data})[output_layer]

    print(result)


def color_demo():
    from XEdu.hub import Workflow as wf
    color = wf(task='gen_color')# ,checkpoint='/home/user/下载/colorization-master/colorizer_siggraph17.onnx')
    img = cv2.imread('/home/user/下载/colorization-master/imgs/ansel_adams3.jpg')
    result,xxx = color.inference(data='/home/user/下载/colorization-master/imgs/ansel_adams3.jpg',img_type='cv2',show=True)
    # color.show(xxx)
    color.format_output(lang="zh")
    color.save(xxx,"color_ou.jpg")

def llm_demo(stream=False):
    from XEdu.LLM import Client
    # print(Client.support_provider())
    # print(Client.support_provider('zh'))
    client = Client(provider="openrouter",api_key="xxx")


    # print(client.support_model())
    # res = client.inference("你好,用中文介绍一下你自己,20字以内",stream=True)
    # for i in res:
    #     print(i, flush=True, end='')
    # print("\n ------------非流式-------------\n")
    # res = client.inference("你好,用中文介绍一下你自己,20字以内")
    # print(res)
    # client.set_system('你的名字叫做xedu小助手，每次回复不会超过20个字且句末加一个"喵"。')
    # text = """
    # 你的名字叫做xedu小助手，你每次回复不会超过20个字。
    # """
    # client.set_system(text)
    # message = [
    #     {"role":"system","content":"你的名字叫做xedu小助手，你每次回复不会超过20个字。"},
    #     {"role":"user","content":"你好,用中文介绍一下你自己,20字以内"},
    # ]
    # res = client.inference(message,stream=False)
    # print(res)
    # res = client.inference("你好,用中文介绍一下你自己,20字以内",stream=True)

    client.run()

def seg_demo():
    from XEdu.hub import Workflow as wf
    seg = wf(task='segment_anything') # iris act 
    img_p = 'car_plate.jpg'
    # img = 'ele.jpg'
    # img = 'fire.jpg'
    # result, img = seg.inference(data=img_p,img_type='cv2',show=True,prompt=[[170,70],[380,270]])
    # result, img = seg.inference(data=img_p,img_type='cv2',show=True,prompt=[380,270])
    result, img = seg.inference(data=img_p,img_type='cv2',show=True)
    # seg.show(img)
    import time
    # start = time.time()
    # result,img = seg.inference(data=img_p,mode='box',img_type='cv2',show=True)
    result,img = seg.inference(data=img_p,mode='box',prompt=[[170,70,380,270]],img_type='pil',show=True)
    # result,img = seg.inference(data=img_p,mode='box',prompt=[[170,70,380,270]],img_type='cv2',show=True)
    # print(time.time()-start)
    # print(result.shape)
    # seg.show(result[0])
    # seg.show(img)
    # seg.format_output(lang="zh")
    seg.save(img,"seg_ou.bmp")

def mde_demo():
    from XEdu.hub import Workflow as wf
    mde = wf(task='depth_anything')
    import time
    a = time.time()
    res = mde.inference(data="det.jpg")
    print(time.time()-a)
    mde.show(res)
    res, img = mde.inference(data="ele.jpg",img_type='cv2',show=True)
    mde.show(img)
    mde.show(res)
    # mde.show(img)
    mde.save(img,"mde_ou1.jpg")
    mde.save(res,"mde_ou2.jpg")
    mde.format_output(lang="zh")

def embedding_audio_demo():
    from XEdu.hub import Workflow as wf
    from XEdu.utils import get_similarity
    import time
    audio_emb = wf(task='embedding_audio')
    audio_files_3 = ['/home/user/桌面/pip测试7/Project/Spokendigit-master/content/原始0-9/three/da5dadb9_nohash_0.wav',
                    '/home/user/桌面/pip测试7/Project/Spokendigit-master/content/原始0-9/three/0f7205ef_nohash_0.wav'
                    ]
    audio_files_1 = ['/home/user/桌面/pip测试7/Project/Spokendigit-master/content/原始0-9/one/0c40e715_nohash_0.wav']

    audio_files_7 = ['/home/user/桌面/pip测试7/Project/Spokendigit-master/content/原始0-9/seven/0a0b46ae_nohash_0.wav']

    audio_files_9 = ['/home/user/桌面/pip测试7/Project/Spokendigit-master/content/原始0-9/nine/0b77ee66_nohash_0.wav']
    a1 = [audio_files_3[1] ]
    a2 = [audio_files_3[0] , audio_files_1[0]]    
    a = time.time()
    emb_1 = audio_emb.inference(data=a1[0])
    print(time.time()-a)
    # print("emb_1",emb_1.shape,emb_1[0][:10])
    b = time.time()
    emb_2 = audio_emb.inference(data=a2)
    print(time.time()-b)
    # print("emb_3",emb_2.shape,emb_2[0][:10],emb_2[1][:10])
    a = get_similarity(emb_1,emb_2)
    print("a",a)

def repo_infer():
    from XEdu.hub import Workflow as wf
    img_path = "ele.jpg"
    mm = wf(repo="fhl123/mobileone_test",download_path="new_repo1") # iris act 
    result = mm.inference(data=img_path,xxx="input")
    print(result)

if __name__ == "__main__":
    # pose_infer_demo()
    # det_infer_demo()
    # video_infer_demo()
    # hand_video_demo()
    # coco_det_demo()
    # hand_det_demo()
    # face_det_demo()
    # ocr_demo()
    # mmedu_demo()
    # custom_demo()
    # basenn_demo()
    # cls_demo()
    # baseml_demo()
    style_demo()
    # qa_demo()
    # drive_demo()
    # embedding_demo()
    # color_demo()
    # llm_demo()
    # seg_demo()
    # mde_demo()
    # embedding_audio_demo()
    # repo_infer()
