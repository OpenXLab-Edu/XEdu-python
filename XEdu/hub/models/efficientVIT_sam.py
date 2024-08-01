from copy import deepcopy
from typing import Any, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


class SamEncoder:
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        # print(f"loading encoder model from {model_path}...")
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)
        self.input_name = self.session.get_inputs()[0].name

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        feature = self.session.run(None, {self.input_name: tensor})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)


class SamDecoder:
    def __init__(
        self, model_path: str, device: str = "cpu", target_size: int = 1024, mask_threshold: float = 0.0, **kwargs
    ):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        # print(f"loading decoder model from {model_path}...")
        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def run(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: Union[list, tuple],
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        return_logits: bool = False,
    ):
        input_size = self.get_preprocess_shape(*origin_image_size, long_side_length=self.target_size)

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)

            prompts, labels = point_coords, point_labels

        if boxes is not None:
            boxes = self.apply_boxes(boxes, origin_image_size, input_size).astype(np.float32)
            box_labels = np.array([[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32).reshape((-1, 2))

            if point_coords is not None:
                prompts = np.concatenate([prompts, boxes], axis=1)
                labels = np.concatenate([labels, box_labels], axis=1)
            else:
                prompts, labels = boxes, box_labels

        input_dict = {"image_embeddings": img_embeddings, "point_coords": prompts, "point_labels": labels}
        low_res_masks, iou_predictions = self.session.run(None, input_dict)

        masks = mask_postprocessing(low_res_masks, origin_image_size)

        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes

def preprocess(x, img_size):
    img = Image.fromarray(x)

    orig_width, orig_height = img.size
    resized_width, resized_height = img.size

    if orig_width > orig_height:
        resized_width = img_size
        resized_height = int(img_size / orig_width * orig_height)
    else:
        resized_height = img_size
        resized_width = int(img_size / orig_height * orig_width)

    img = img.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    input_tensor = np.array(img)

    # Normalize input tensor numbers
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([[58.395, 57.12, 57.375]])
    input_tensor = (input_tensor - mean) / std

    input_tensor = input_tensor.transpose(2,0,1)[None,:,:,:].astype(np.float32)
    if resized_height < resized_width:
        input_tensor = np.pad(input_tensor,((0,0),(0,0),(0,img_size-resized_height),(0,0)))
    else:
        input_tensor = np.pad(input_tensor,((0,0),(0,0),(0,0),(0,img_size-resized_width)))
    return input_tensor

def resize_longest_image_size(input_image_size: np.ndarray, longest_side: int) -> np.ndarray:
    scale = longest_side / max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = np.floor(transformed_size + 0.5).astype(np.int64)
    return transformed_size

def mask_postprocessing(masks: np.ndarray, orig_im_size: np.ndarray) -> np.ndarray:
    img_size = 1024
    masks = cv2.resize(np.transpose(masks[0], (1, 2, 0)), (img_size, img_size), interpolation=cv2.INTER_LINEAR)[None, :, :]
    masks = np.transpose(masks, (0, 3, 1, 2))
    prepadded_size = resize_longest_image_size(np.array(orig_im_size), img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
    h, w = orig_im_size[0], orig_im_size[1]
    masks = cv2.resize(np.transpose(masks[0], (1, 2, 0)), (int(w), int(h)), interpolation=cv2.INTER_LINEAR)[None, :, :]
    masks = np.transpose(masks, (0, 3, 1, 2))
    return masks
