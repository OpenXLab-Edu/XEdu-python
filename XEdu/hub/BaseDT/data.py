# 导入工具库
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# import jieba
import json
import numbers
import codecs
from .utils import (check_type, check_type_and_length,
                                          color_val_matplotlib,
                                          img_from_canvas,
                                          value2list, wait_continue)

if TYPE_CHECKING:
    from matplotlib.font_manager import FontProperties


class ImageData(object):
    '''
    目前支持图片路径, np array, 文本
    后续支持 PIL
    '''
    _defaults = {
        "data_type": 'uint8',
        # 图像默认属性
        "to_rgb": True,
        "normalize": True,
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
    }
    _backbone_args = {
        "LeNet": {"size": (32, 32), "to_rgb": False, "mean": [33.46], "std":[78.87]},
        "ResNet18": {"size": (256, 256), "crop_size": (224, 224)},
        "ResNet50": {"size": (256, 256), "crop_size": (224, 224)},
        "ResNeXt": {"size": (256, 256), "crop_size": (224, 224)},
        "ShuffleNet_v2": {"size": (256, 256), "crop_size": (224, 224)},
        "VGG": {"size": (256, 256), "crop_size": (224, 224)},
        "RepVGG": {"size": (256, 256), "crop_size": (224, 224)},
        "MobileNet": {"size": (256, 256), "crop_size": (224, 224)},
        "SSD_Lite": {"size": (320, 320), "pad_size_divisor": 320},
        "FasterRCNN": {"size": (800, 1333)},#, "size_keep_ratio": True,  "pad_size_divisor": 32},
        "Mask_RCNN": {"size": (1333, 800), "size_keep_ratio": True, "pad_size_divisor": 32},
        "RegNet": {"size": (1333, 800), "size_keep_ratio": True, "pad_size_divisor": 32},
        "Yolov3": {"size": (416, 416)},# , "size_keep_ratio": True, "pad_size_divisor": 32},
        "Simcc_MobileNet_v2": {"size":(192, 256)},
    }

    def get_attribute(self, n):
        if n in self.__dict__:
            return self.__dict__[n]
        else:
            return None

    def __init__(self, data_source, **kwargs):
        self.vct = None
        self.data_source = data_source
        # 装在默认参数进入私有变量列表  
        self.__dict__.update(self._defaults)
        # 将关键词参数装入私有变量列表
        for name, value in kwargs.items():
            setattr(self, name, value)

        if type(self.data_source) == str:
            self.value = cv2.imdecode(np.fromfile((self.data_source),dtype=np.uint8),-1)
        if type(self.data_source) == np.ndarray:
            self.value = self.data_source
        else:
            # TODO 检查合法性
            pass
        self.value = self.value.astype(self.get_attribute("data_type"))
        self.raw_value = self.value
        self.init_by_backbone = False
        self.fig_save = None
        self.fig_save_cfg = None
        self.fig_show_cfg = None
        self.fig_save_canvas = None
        self.fig_save = None
        self.ax_save = None
        self.dpi = None
        self.width = None
        self.height = None
        self._image = None

        if self.get_attribute("backbone"):
            # for key, value in self._backbone_args[self.get_attribute("backbone")].items():
            #     print(key,value)
            self.init_by_backbone = True
            self.value = ImageData(data_source, **self._backbone_args[self.get_attribute("backbone")]).value
            self.__dict__.update(self._backbone_args[self.get_attribute("backbone")])#更新私有变量列表
        else:
            if self.get_attribute("to_rgb") == False and len(self.value.shape)>=3:
                self._rgb2gray()
            elif self.get_attribute("to_rgb") == True:
                self._to3channel()
            #需要考虑各种操作顺序的灵活性，循环遍历私有变量列表来实现
            for key in self.__dict__.keys():
                if key == "size" or key == "size_keep_ratio":
                    size = self.get_attribute("size")
                    size_keep_ratio = self.get_attribute("size_keep_ratio")
                    self._resize(size, size_keep_ratio)
                if key == "crop_size":
                    crop_size = self.get_attribute("crop_size")
                    self._crop(crop_size)
                if key == "normalize":
                    #TODO 需检查维度
                    if self.get_attribute("normalize") == True:
                        mean = self.get_attribute("mean")
                        std = self.get_attribute("std")
                        self._normalize(mean, std)
                if key == "pad_size_divisor":
                    pad_size_divisor = self.get_attribute("pad_size_divisor")
                    self._pad(size_divisor=pad_size_divisor)

    def to_tensor(self):
        if self.get_attribute("to_rgb"):
            return np.expand_dims(np.transpose(self.value, (2,0,1)), 0)
        else:
            return np.expand_dims(np.expand_dims(self.value, 0), 0)
    
    def init_plt(self, fig_save_cfg=dict(frameon=False),
                  fig_show_cfg=dict(frameon=False)):
        self.fig_save = None
        self.fig_save_cfg = fig_save_cfg
        self.fig_show_cfg = fig_show_cfg
        (self.fig_save_canvas, self.fig_save,
         self.ax_save) = self._initialize_fig(fig_save_cfg)
        self.dpi = self.fig_save.get_dpi()
        if self.init_by_backbone:
            image_rgb = cv2.cvtColor(self.raw_value, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = self.raw_value
        self._set_image(image_rgb)

    #保护方法，不给用户调用
    def _rgb2gray(self):
        gray = np.dot(self.value, [0.2989, 0.5870, 0.1140])
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        self.value = gray

    def _to3channel(self, background=(255, 255, 255)):
        if len(self.value.shape) == 2:
            self.value = cv2.cvtColor(self.value, cv2.COLOR_GRAY2RGB)
        elif len(self.value.shape) == 3 and self.value.shape[-1] > 3:
            row, col, ch = self.value.shape
            rgb = np.zeros((row, col, 3), dtype='float32')
            r, g, b, a = self.value[:, :, 0], self.value[:, :, 1], self.value[:, :, 2], self.value[:, :, 3]
            a = np.asarray(a, dtype='float32') / 255.0
            R, G, B = background
            rgb[:, :, 0] = r * a + (1.0 - a) * R

            rgb[:, :, 1] = g * a + (1.0 - a) * G

            rgb[:, :, 2] = b * a + (1.0 - a) * B
            self.value = np.asarray(rgb, dtype='uint8')

    def _normalize(self, mean, std):
        # assert self.value.dtype != np.uint8
        img = self.value
        mean = np.float64(np.array(mean).reshape(1, -1))
        stdinv = 1 / np.float64(np.array(std).reshape(1, -1))
        if self.get_attribute("to_rgb"):
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        img = img.astype(np.float32)
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        self.value = img

    def _resize(self, size, keep_ratio = False):
        #TODO 支持padding
        if keep_ratio:
            h, w = self.value.shape[:2]
            img_shape = self.value.shape[:2]
            max_long_edge = max(size)
            max_short_edge = min(size)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
            # scale_factor = [1.0*size[0]/w, 1.0*size[1]/h]
            new_size = self._scale_size(img_shape[::-1], scale_factor)
            # print(new_size)
            self.value = cv2.resize(self.value, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            self.value = cv2.resize(self.value, size, interpolation=cv2.INTER_LINEAR)
        #self.value = np.resize(self.value, size, interp='bilinear')

    def _scale_size(self, size, scale):
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        w, h = size
        return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

    def _bbox_clip(self, bboxes, img_shape):
        assert bboxes.shape[-1] % 4 == 0
        cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
        cmin[0::2] = img_shape[1] - 1
        cmin[1::2] = img_shape[0] - 1
        clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
        return clipped_bboxes

    def _bbox_scaling(self, bboxes, scale, clip_shape=None):
        if float(scale) == 1.0:
            scaled_bboxes = bboxes.copy()
        else:
            w = bboxes[..., 2] - bboxes[..., 0] + 1
            h = bboxes[..., 3] - bboxes[..., 1] + 1
            dw = (w * (scale - 1)) * 0.5
            dh = (h * (scale - 1)) * 0.5
            scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
        if clip_shape is not None:
            return self._bbox_clip(scaled_bboxes, clip_shape)
        else:
            return scaled_bboxes

    def _bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]

        # get bbox center and scale
        x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding

        if dim == 1:
            center = center[0]
            scale = scale[0]

        return center, scale

    def _crop(self, size, scale=1.0, pad_fill=None):
        if isinstance(size, int):
            crop_size = (size, size)
        else:
            crop_size = size

        img = self.value
        img_height, img_width = img.shape[:2]

        crop_height, crop_width = crop_size

        if crop_height > img_height or crop_width > img_width:
            #TODO 可选择pad_mod
            pass
        else:
            crop_height = min(crop_height, img_height)
            crop_width = min(crop_width, img_width)

        y1 = max(0, int(round((img_height - crop_height) / 2.)))
        x1 = max(0, int(round((img_width - crop_width) / 2.)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1
        bboxes = np.array([x1, y1, x2, y2])

        chn = 1 if img.ndim == 2 else img.shape[2]
        if pad_fill is not None:
            if isinstance(pad_fill, (int, float)):
                pad_fill = [pad_fill for _ in range(chn)]
            assert len(pad_fill) == chn

        _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = self._bbox_scaling(_bboxes, scale).astype(np.int32)
        clipped_bbox = self._bbox_clip(scaled_bboxes, img.shape)

        patches = []
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
            if pad_fill is None:
                patch = img[y1:y2 + 1, x1:x2 + 1, ...]
            else:
                _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
                if chn == 1:
                    patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
                else:
                    patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
                patch = np.array(
                    pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
                x_start = 0 if _x1 >= 0 else -_x1
                y_start = 0 if _y1 >= 0 else -_y1
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                patch[y_start:y_start + h, x_start:x_start + w,
                ...] = img[y1:y1 + h, x1:x1 + w, ...]
            patches.append(patch)

        if bboxes.ndim == 1:
            self.value = patches[0]
        else:
            self.value = patches

    def _pad(self, size=None, size_divisor=None, pad_val=0, padding_mode='constant'):
        img = self.value
        if size_divisor is not None:
            if size is None:
                size = (img.shape[0], img.shape[1])
            pad_h = int(np.ceil(
                size[0] / size_divisor)) * size_divisor
            pad_w = int(np.ceil(
                size[1] / size_divisor)) * size_divisor
            size = (pad_h, pad_w)

        shape = size
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

        # check pad_val
        if isinstance(pad_val, tuple):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError('pad_val must be a int or a tuple. '
                            f'But received {type(pad_val)}')

        # check padding
        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                             f'But received {padding}')

        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val)
        self.value = img

    def _flip(self):
        pass
    
    def show(self, raw = False):
        if raw: img = self.raw_value
        else: img = self.value
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
       
    def map_orig_coords(self, boxes):
        #TODO 暂时只支持box的映射，后续应该支持单纯的坐标映射
        original_w, original_h = self.raw_value.shape[:2]
        processed_w, processed_h = self.value.shape[:2]
        sx, sy = original_w / processed_w, original_h / processed_h
        new_boxes = np.array([]).reshape(0, 5)
        for box in boxes:
            new_box = [box[0] * sy, box[1] * sx, box[2] * sy, box[3] * sx, box[4]]
            new_boxes = np.concatenate([new_boxes, [new_box]], axis=0)
        return new_boxes

    def _initialize_fig(self, fig_cfg) -> tuple:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(**fig_cfg)
        ax = fig.add_subplot()
        ax.axis(False)

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvasAgg(fig)
        return canvas, fig, ax
    
    def imshow(self,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: float = 0.,
             continue_key: str = ' ',
             backend: str = 'matplotlib') -> None:
        """Show the drawn image.

        Args:
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (float): Delay in seconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
            backend (str): The backend to show the image. Defaults to
                'matplotlib'. `New in version 0.7.3.`
        """
        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            is_inline = 'inline' in plt.get_backend()

            img = self.get_image() if drawn_img is None else drawn_img
            self._init_manager(win_name)
            #fig = self.manager.canvas.figure
            # remove white edges by set subplot margin
            #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            #fig.clear()
            #ax = fig.add_subplot()
            plt.axis(False)
            plt.imshow(img)
            self.manager.canvas.draw()
            plt.show()
            # Find a better way for inline to show the image
            #if is_inline:
            #    return fig
            #wait_continue(fig, timeout=wait_time, continue_key=continue_key)

        elif backend == 'cv2':
            # Keep images are shown in the same window, and the title of window
            # will be updated with `win_name`.
            cv2.namedWindow(winname=f'{id(self)}')
            cv2.setWindowTitle(f'{id(self)}', win_name)
            cv2.imshow(
                str(id(self)),
                self.get_image() if drawn_img is None else drawn_img)
            cv2.waitKey(int(np.ceil(wait_time * 1000)))
        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    def _set_image(self, image: np.ndarray) -> None:
        assert image is not None
        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        self.fig_save.set_size_inches(  # type: ignore
            (self.width + 1e-2) / self.dpi, (self.height + 1e-2) / self.dpi)

        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')
    
    def get_image(self) -> np.ndarray:
        assert self._image is not None, 'Please set image using `set_image`'
        return img_from_canvas(self.fig_save_canvas)  # type: ignore
    
    def draw_texts(
        self,
        texts: Union[str, List[str]],
        positions: np.ndarray,
        font_sizes: Optional[Union[int, List[int]]] = None,
        colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        vertical_alignments: Union[str, List[str]] = 'top',
        horizontal_alignments: Union[str, List[str]] = 'left',
        font_families: Union[str, List[str]] = 'sans-serif',
        bboxes: Optional[Union[dict, List[dict]]] = None,
        font_properties: Optional[Union['FontProperties',
                                        List['FontProperties']]] = None
    ) -> 'Visualizer':
        from matplotlib.font_manager import FontProperties
        check_type('texts', texts, (str, list))
        if isinstance(texts, str):
            texts = [texts]
        num_text = len(texts)
        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape == (num_text, 2), (
            '`positions` should have the shape of '
            f'({num_text}, 2), but got {positions.shape}')
        positions = positions.tolist()

        if font_sizes is None:
            font_sizes = self._default_font_size
        check_type_and_length('font_sizes', font_sizes, (int, float, list),
                              num_text)
        font_sizes = value2list(font_sizes, (int, float), num_text)

        check_type_and_length('colors', colors, (str, tuple, list), num_text)
        colors = value2list(colors, (str, tuple), num_text)
        colors = color_val_matplotlib(colors)  # type: ignore

        check_type_and_length('vertical_alignments', vertical_alignments,
                              (str, list), num_text)
        vertical_alignments = value2list(vertical_alignments, str, num_text)

        check_type_and_length('horizontal_alignments', horizontal_alignments,
                              (str, list), num_text)
        horizontal_alignments = value2list(horizontal_alignments, str,
                                           num_text)

        check_type_and_length('font_families', font_families, (str, list),
                              num_text)
        font_families = value2list(font_families, str, num_text)

        if font_properties is None:
            font_properties = [None for _ in range(num_text)]  # type: ignore
        else:
            check_type_and_length('font_properties', font_properties,
                                  (FontProperties, list), num_text)
            font_properties = value2list(font_properties, FontProperties,
                                         num_text)

        if bboxes is None:
            bboxes = [None for _ in range(num_text)]  # type: ignore
        else:
            check_type_and_length('bboxes', bboxes, (dict, list), num_text)
            bboxes = value2list(bboxes, dict, num_text)

        for i in range(num_text):
            self.ax_save.text(
                positions[i][0],
                positions[i][1],
                texts[i],
                size=font_sizes[i],  # type: ignore
                bbox=bboxes[i],  # type: ignore
                verticalalignment=vertical_alignments[i],
                horizontalalignment=horizontal_alignments[i],
                family=font_families[i],
                fontproperties=font_properties[i],
                color=colors[i])
        return self

    def draw_bboxes(
        self,
        bboxes: np.ndarray,
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[int, float] = 0.8,
    ) -> 'Visualizer':
        if len(bboxes.shape) == 1:
            bboxes = bboxes[None]
        assert bboxes.shape[-1] == 4, (
            f'The shape of `bboxes` should be (N, 4), but got {bboxes.shape}')

        assert (bboxes[:, 0] <= bboxes[:, 2]).all() and (bboxes[:, 1] <=
                                                         bboxes[:, 3]).all()
        poly = np.stack(
            (bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 1],
             bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 3]),
            axis=-1).reshape(-1, 4, 2)
        poly = [p for p in poly]
        return self.draw_polygons(
            poly,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)
    
    def draw_lines(
        self,
        x_datas: np.ndarray,
        y_datas: np.ndarray,
        colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2
    ) -> 'Visualizer':
        from matplotlib.collections import LineCollection
        #check_type('x_datas', x_datas, (np.ndarray, torch.Tensor))
        #x_datas = tensor2ndarray(x_datas)
        #check_type('y_datas', y_datas, (np.ndarray, torch.Tensor))
        #y_datas = tensor2ndarray(y_datas)
        assert x_datas.shape == y_datas.shape, (
            '`x_datas` and `y_datas` should have the same shape')
        assert x_datas.shape[-1] == 2, (
            f'The shape of `x_datas` should be (N, 2), but got {x_datas.shape}'
        )
        if len(x_datas.shape) == 1:
            x_datas = x_datas[None]
            y_datas = y_datas[None]
        colors = color_val_matplotlib(colors)  # type: ignore
        lines = np.concatenate(
            (x_datas.reshape(-1, 2, 1), y_datas.reshape(-1, 2, 1)), axis=-1)
        if not self._is_posion_valid(lines):
            warnings.warn(
                'Warning: The line is out of bounds,'
                ' the drawn line may not be in the image', UserWarning)
        line_collect = LineCollection(
            lines.tolist(),
            colors=colors,
            linestyles=line_styles,
            linewidths=line_widths)
        self.ax_save.add_collection(line_collect)
        return self
    
    def draw_circles(
        self,
        center: np.ndarray,
        radius: np.ndarray,
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[float, int] = 0.8,
    ) -> 'Visualizer':
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle
        #check_type('center', center, (np.ndarray, torch.Tensor))
        #center = tensor2ndarray(center)
        #check_type('radius', radius, (np.ndarray, torch.Tensor))
        #radius = tensor2ndarray(radius)
        if len(center.shape) == 1:
            center = center[None]
        assert center.shape == (radius.shape[0], 2), (
            'The shape of `center` should be (radius.shape, 2), '
            f'but got {center.shape}')
        if not (self._is_posion_valid(center -
                                      np.tile(radius.reshape((-1, 1)), (1, 2)))
                and self._is_posion_valid(
                    center + np.tile(radius.reshape((-1, 1)), (1, 2)))):
            warnings.warn(
                'Warning: The circle is out of bounds,'
                ' the drawn circle may not be in the image', UserWarning)

        center = center.tolist()
        radius = radius.tolist()
        edge_colors = color_val_matplotlib(edge_colors)  # type: ignore
        face_colors = color_val_matplotlib(face_colors)  # type: ignore
        circles = []
        for i in range(len(center)):
            circles.append(Circle(tuple(center[i]), radius[i]))

        if isinstance(line_widths, (int, float)):
            line_widths = [line_widths] * len(circles)
        line_widths = [
            min(max(linewidth, 1), self._default_font_size / 4)
            for linewidth in line_widths
        ]
        p = PatchCollection(
            circles,
            alpha=alpha,
            facecolors=face_colors,
            edgecolors=edge_colors,
            linewidths=line_widths,
            linestyles=line_styles)
        self.ax_save.add_collection(p)
        return self
    
    def color_val_matplotlib(
            colors: Union[str, tuple, List[Union[str, tuple]]]
            ) -> Union[str, tuple, List[Union[str, tuple]]]:
        if isinstance(colors, str):
            return colors
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            for channel in colors:
                assert 0 <= channel <= 255
            colors = [channel / 255 for channel in colors]
            return tuple(colors)
        elif isinstance(colors, list):
            colors = [
                color_val_matplotlib(color)  # type:ignore
                for color in colors
            ]
            return colors
        else:
            raise TypeError(f'Invalid type for color: {type(colors)}')

    def draw_polygons(
        self,
        polygons,
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[int, float] = 0.8,
    ):
        from matplotlib.collections import PolyCollection
        edge_colors = color_val_matplotlib(edge_colors)  # type: ignore
        face_colors = color_val_matplotlib(face_colors)  # type: ignore

        if isinstance(polygons, np.ndarray):
            polygons = [polygons]
        if isinstance(polygons, list):
            for polygon in polygons:
                assert polygon.shape[1] == 2, (
                    'The shape of each polygon in `polygons` should be (M, 2),'
                    f' but got {polygon.shape}')
                
        if isinstance(line_widths, (int, float)):
            line_widths = [line_widths] * len(polygons)
        line_widths = [
            min(max(linewidth, 1), self._default_font_size / 4)
            for linewidth in line_widths
        ]
        polygon_collection = PolyCollection(
            polygons,
            alpha=alpha,
            facecolor=face_colors,
            linestyles=line_styles,
            edgecolors=edge_colors,
            linewidths=line_widths)

        self.ax_save.add_collection(polygon_collection)
        return self
    
    def _init_manager(self, win_name: str) -> None:
        """Initialize the matplot manager.

        Args:
            win_name (str): The window name.
        """
        from matplotlib.figure import Figure
        from matplotlib.pyplot import new_figure_manager
        if getattr(self, 'manager', None) is None:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)

        try:
            self.manager.set_window_title(win_name)
        except Exception:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)
            self.manager.set_window_title(win_name)

    def _is_posion_valid(self, position: np.ndarray) -> bool:
        """Judge whether the position is in image.

        Args:
            position (np.ndarray): The position to judge which last dim must
                be two and the format is [x, y].

        Returns:
            bool: Whether the position is in image.
        """
        flag = (position[..., 0] < self.width).all() and \
               (position[..., 0] >= 0).all() and \
               (position[..., 1] < self.height).all() and \
               (position[..., 1] >= 0).all()
        return flag


   
class ModelData(object):

    def __init__(self, model_path):
        self.model_info = self._get_model_info(model_path)

    def _get_model_info(self, model_path):
        import onnxruntime
        sess = onnxruntime.InferenceSession(model_path)
        model_meta = sess.get_modelmeta()
        key='MODEL_INFO'
        if key in model_meta.custom_metadata_map:
            unicode_string = model_meta.custom_metadata_map[key]
            print(f'Success load model info generate by MMEdu>=0.1.15: {unicode_string[:100] + "..." if len(unicode_string) > 100 else unicode_string}')
        else:   
            print('Please input onnx model path convert by MMEdu>=0.1.15 for a better experience.')
            return ''
        #return json.loads(unicode_string)
        #print('ok')
        return json.loads(codecs.decode(unicode_string, 'unicode_escape'))
    def get_model_info(self):
        if self.model_info == '':
            #print('Please input onnx model path convert by MMEdu>=0.1.15.')
            return ''
        else:
            return self.model_info

    def get_codebase(self):
        key = 'codebase'
        if self.model_info == '':
            #print('Please input onnx model path convert by MMEdu>=0.1.15.')
            return ''
        else:
            if key in self.model_info:
                return self.model_info[key]
            else:
                print(f'The model info does not contain key:{key}')

    def get_modelname(self):
        key = 'modelname'
        if self.model_info == '':
            #print('Please input onnx model path convert by MMEdu>=0.1.15.')
            return ''
        else:
            if key in self.model_info:
                return self.model_info[key]
            else:
                print(f'The model info does not contain key:{key}')

    def get_labels(self):
        key = 'classes'
        if self.model_info == '':
            #print('Please input onnx model path convert by MMEdu>=0.1.15.')
            return ''
        else:
            if key in self.model_info:
                return self.model_info[key]
            else:
                print(f'The model info does not contain key:{key}')
    
    

def get_fold_ImageData(data_source: str, **kwargs) -> Union[ImageData, List[ImageData]]:
    if os.path.isfile(data_source):
        return ImageData(data_source, **kwargs)
    elif os.path.isdir(data_source):
        return [ImageData(os.path.join(data_source, f), **kwargs) for f in os.listdir(data_source) if os.path.isfile(os.path.join(data_source, f))]
    else:
        raise ValueError('Invalid path. Please provide a valid file or directory path.')
