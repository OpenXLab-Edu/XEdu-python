import cv2
import os
from typing import Tuple,TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import json


if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg


DEFAULT_TEXT_CFG = {
    'family': 'monospace',
    'color': 'white',
    'bbox': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
    'verticalalignment': 'top',
    'horizontalalignment': 'left',
}

# def draw_boxes(image, boxes, labels = None, color=(0, 0, 255), thickness=2):
#     font_scale = image.shape[0] / 500
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     for box, label in zip(boxes, labels):
#         x1, y1, x2, y2 = box
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
#         text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
#         text_x = x1 + (x2 - x1 - text_size[0]) / 2
#         text_y = y1 + (y2 - y1 + text_size[1]) / 2
#         cv2.putText(image, label, (int(text_x), int(text_y)), font, font_scale, color, thickness)
#     return image
class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if isinstance(color, str):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')

def imshow(img, need_win = False, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    # 转RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if need_win:
        cv2.imshow(win_name, img)
        if wait_time == 0:  # prevent from hanging if windows was closed
            while True:
                ret = cv2.waitKey(1)

                closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
                # if user closed window or if some key pressed
                if closed or ret != -1:
                    break
        else:
            ret = cv2.waitKey(wait_time)
    else:
        plt.imshow(img)
        plt.show()
        
def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color=None,
                      text_color=None,
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    assert isinstance(img,str) or isinstance(img, np.ndarray)
    if isinstance(img, str):
        img = cv2.imread(img)
    img = np.ascontiguousarray(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    if bbox_color or text_color is None:
        bbox_color = []
        text_color = []
        for c in Color:
            bbox_color.append(color_val(c))
            text_color.append(color_val(c))

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        idx = label % len(bbox_color)
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color[idx], thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color[idx])

    if show:
        imshow(img, win_name, wait_time)
    return img

def plot_log(log, title='Loss Graph', plot_list='loss'):

    if isinstance(plot_list, str):
        plot_list = [plot_list]
    elif isinstance(plot_list, (list, tuple)):
        assert len(plot_list) == len(set(plot_list)), (
            'Find duplicate elements in "plot_list".')
    for key in plot_list:
        assert key in ['loss','loss_cls','loss_bbox']

    iter = []
    loss_dict = {
        'loss': [],
        'loss_cls':[],
        'loss_bbox':[]
    }

    iter_num = 0

    if isinstance(log, str):
        with open(log,'r') as f:
            content = [json.loads(line) for line in f.readlines()]
    elif isinstance(log, list):
        content = log
    
    assert isinstance(content, list)
    for log in content:
        if 'mode' in log.keys() and log["mode"] == 'train':
            iter_num += 10
            iter.append(iter_num)
            non_empty_keys = log.keys()
            for axis_y in plot_list:
                loss_dict[axis_y].append(log[axis_y])

    plt.figure(figsize=(8,6))
    for axis_y in plot_list:
        plt.plot(iter,loss_dict[axis_y],'',label=axis_y)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xlabel('iter')
    plt.ylabel('')
    plt.grid(True)
    plt.show()
    non_empty_keys = [key for key in non_empty_keys if key.startswith('loss')]
    print('The loss function graph is drawn. If you want to add y-axis parameters,'
          f' please set `{non_empty_keys}` in the plot_list')


def img_from_canvas(canvas: 'FigureCanvasAgg') -> np.ndarray:
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype('uint8')

def get_adaptive_scale(img_shape: Tuple[int, int],
                       min_scale: float = 0.3,
                       max_scale: float = 3.0) -> float:
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)

DEFAULT_TEXT_CFG = {
    'family': 'monospace',
    'color': 'white',
    'bbox': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
    'verticalalignment': 'top',
    'horizontalalignment': 'left',
}
def draw_single_cls(image, result):
    texts = []
    max_length = 12 # 文本长度限制
    if '标签' in result:
        texts.append('Pred_label: {}'.format(result['标签']))
    if '置信度' in result:
        texts.append('Pred_score: {:.2f}'.format(result['置信度']))
    if '预测结果' in result:
        texts.append('Pred_result: {}'.format(result['预测结果'][:max_length]))
    text = '\n'.join(texts)
    #fig = plt.figure()
    image.init_plt()
    img_tmp = image._image
    img_scale = get_adaptive_scale(img_tmp.shape[:2])
    DEFAULT_TEXT_CFG['size'] = int(img_scale * 5)
    pos = np.array([[img_scale*5, img_scale*5]], dtype=int)
    image.draw_texts(texts=text, positions=pos, colors='w', font_sizes=int(img_scale * 12), font_families='monospace',
                            bboxes=[{
                                'facecolor': 'black',
                                'alpha': 0.5,
                                'boxstyle': 'Round'
                            }])

    '''
    image = image._image
    img_scale = get_adaptive_scale(image.shape[:2])
    DEFAULT_TEXT_CFG['size'] = int(img_scale * 5)
    plt.text(img_scale*5, img_scale*5, text, fontdict=DEFAULT_TEXT_CFG)
    #canvas = FigureCanvasAgg(fig)
    #image = img_from_canvas(canvas)
    plt.imshow(image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    #print(text)
    '''

# 将vis绘图内容与mmlab对齐
def show_cls(images, results):
    # 探讨传入dt还是img_path
    # 判断是文件夹还是文件
    if isinstance(images, list) and isinstance(results, list):
        for image, result in zip(images, results):
            draw_single_cls(image, result)
            image.imshow()
    else:
        draw_single_cls(images, results)
        images.imshow()
        

def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales

def get_palette(num_classes: int) -> List[Tuple[int]]:
    state = np.random.get_state()
    np.random.seed(1956)
    palette = np.random.randint(0, 168, size=(num_classes, 3))
    np.random.set_state(state)
    dataset_palette = [tuple(c) for c in palette]
    assert len(dataset_palette) >= num_classes, \
        'The length of palette should not be less than `num_classes`.'
    return dataset_palette

def draw_single_det(image, result, line_width: int =3) -> None:
    assert isinstance(result, list), "Expected 'result' to be a list"
    image.init_plt()
    if len(result) == 0:
        return
    bboxes = []
    labels = []
    label_texts = []
    for bbox in result:
        assert isinstance(bbox, dict), "Expected each element in 'result' to be a dictionary"
        assert '坐标' in bbox and '标签' in bbox, "Each dictionary in 'result' should have '坐标' and '标签' keys"
        
        coords = bbox['坐标']
        assert isinstance(coords, dict), "Expected '坐标' to be a dictionary"
        assert all(key in coords for key in ['x1', 'y1', 'x2', 'y2']), "The '坐标' dictionary should have 'x1', 'y1', 'x2', 'y2' keys"
        
        bboxes.append([coords['x1'], coords['y1'], coords['x2'], coords['y2']])
        labels.append(bbox['标签'])
        if '预测结果' in bbox:
            label_texts.append('{}:{}'.format(bbox['预测结果'],round(float(bbox['置信度']) * 100, 1)))
        else:
            label_texts.append('{}:{}'.format(bbox['标签'],round(float(bbox['置信度']) * 100, 1)))
    bboxes = np.array(bboxes)
    labels = np.array(labels)


    max_label = int(max(labels) if len(labels) > 0 else 0)

    bbox_palette = get_palette(max_label + 1)
    colors = [bbox_palette[label] for label in labels]

    image.draw_bboxes(bboxes=bboxes, edge_colors=colors, line_widths=line_width)

    positions = bboxes[:, :2] + line_width
    areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
    scales = _get_adaptive_scales(areas)
    for i, (pos, label_text) in enumerate(zip(positions, label_texts)):    
            image.draw_texts(
                            label_text,
                            pos,
                            colors=(200,200,200),
                            font_sizes=int(13 * scales[i]),
                            bboxes=[{
                                'facecolor': 'black',
                                'alpha': 0.8,
                                'pad': 0.7,
                                'edgecolor': 'none'
                            }])

def show_det(images, results):
    if isinstance(images, list) and isinstance(results, list):
        for image, result in zip(images, results):
            draw_single_det(image=image, result= result)
            image.imshow()
    else:
        draw_single_det(image=images, result= results)
        images.imshow()

def draw_single_pose(image, result, thr=0.3) -> None:
    image.init_plt()
    keypoints = result['关键点']
    scores = result['得分']
    #result = {'关键点':keypoints, '得分':scores}
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
            (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    link_color = [
        1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]
    point_color = [
        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4,
        4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]

    # draw keypoints and skeleton
    for kpts, score in zip(keypoints, scores):
        for kpt, color in zip(kpts, point_color):
            image.draw_circles(center=kpt,
                                radius=np.array([1.5]),
                                edge_colors=tuple(palette[color]))
            #cv2.circle(img, tuple(kpt.astype(np.int32)), 1, palette[color], 1,
            #           cv2.LINE_AA)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                x_data = [kpts[u][0], kpts[v][0]]
                y_data = [kpts[u][1], kpts[v][1]]
                image.draw_lines(x_datas=np.array(x_data),
                                y_datas=np.array(y_data),
                                colors=tuple(palette[color]))
                #cv2.line(img, tuple(kpts[u].astype(np.int32)),
                #         tuple(kpts[v].astype(np.int32)), palette[color], 2,
                #         cv2.LINE_AA)
    

def show_pose(images, results):
    if isinstance(images, list) and isinstance(results, list):
        for image, result in zip(images, results):
            draw_single_pose(image=image, result= result)
            image.imshow()
    else:
        draw_single_pose(image=images, result= results)
        images.imshow()