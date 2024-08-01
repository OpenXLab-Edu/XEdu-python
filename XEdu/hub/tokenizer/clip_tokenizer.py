from typing import Union

import cv2 as cv
import numpy as np
from PIL import Image
import gzip
import html
import os
from typing import Dict, List, Set, Union, Iterable

import ftfy
import numpy as np
import regex as re

class Preprocessor:
    "Preprocesses images for CLIP."
    CLIP_INPUT_SIZE = 224
    NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1, 1, 3))
    NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1, 1, 3))

    @staticmethod
    def _crop_and_resize(img: np.ndarray) -> np.ndarray:
        """Resize and crop an image to a square, preserving the aspect ratio."""
        # Current height and width
        h, w = img.shape[0:2]

        if h * w == 0:
            raise ValueError(
                f"Height and width of the image should both be non-zero but got shape {h, w}"
            )

        target_size = Preprocessor.CLIP_INPUT_SIZE

        if h < w:
            resized_h = target_size
            resized_w = int(resized_h * w / h)
        else:
            resized_w = target_size
            resized_h = int(resized_w * h / w)

        # PIL resizing behaves slightly differently than OpenCV because of
        # antialiasing. See also
        # https://pytorch.org/vision/main/generated/torchvision.transforms.functional.resize.html
        # CLIP uses PIL, so we do too to match its results. But if you don't
        # want to have PIL as a dependency, feel free to change the code to
        # use the other branch.
        use_pil_for_resizing = True

        if use_pil_for_resizing:
            # https://github.com/pytorch/vision/blob/7cf0f4cc1801ff1892007c7a11f7c35d8dfb7fd0/torchvision/transforms/functional_pil.py#L240
            # We're working with float images but PIL uses uint8, so convert
            # there and back again afterwards
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize(
                (resized_w, resized_h), resample=Image.BICUBIC
            )
            img = np.array(img_pil).astype(np.float32) / 255
        else:
            img = cv.resize(
                img, (resized_w, resized_h), interpolation=cv.INTER_CUBIC
            )

        # Now crop to a square
        y_from = (resized_h - target_size) // 2
        x_from = (resized_w - target_size) // 2
        img = img[
            y_from : y_from + target_size, x_from : x_from + target_size, :
        ]

        return img

    @staticmethod
    def _image_to_float_array(img: Union[Image.Image, np.ndarray]):
        """Converts a PIL image or a NumPy array to standard form.

        Standard form means:
        - the shape is (H, W, 3)
        - the dtype is np.float32
        - all values are in [0, 1]
        - there are no NaN values

        Args:
            img: The image to convert.

        Returns:
            The image converted to a NumPy array in standard form.

        Raises:
            ValueError if the image is invalid (wrong shape, invalid
                values...).
        """
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise TypeError(
                f"Expected PIL Image or np.ndarray but instead got {type(img)}"
            )

        if isinstance(img, Image.Image):
            # Convert to NumPy
            img = np.array(img)

        if len(img.shape) > 3:
            raise ValueError(
                f"The image should have 2 or 3 dimensions but got "
                f"{len(img.shape)} dimensions"
            )
        if len(img.shape) == 3 and img.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image but got image with "
                f"{img.shape[2]} channels"
            )

        # Handle grayscale
        if len(img.shape) == 2:
            # The model doesn't support HxWx1 images as input
            img = np.expand_dims(img, axis=2)  # HxWx1
            img = np.concatenate((img,) * 3, axis=2)  # HxWx3

        # At this point, `img` has the shape (H, W, 3).

        if np.min(img) < 0:
            raise ValueError(
                "Images should have non-negative pixel values, "
                f"but the minimum value is {np.min(img)}"
            )

        if np.issubdtype(img.dtype, np.floating):
            if np.max(img) > 1:
                raise ValueError(
                    "Images with a floating dtype should have values "
                    f"in [0, 1], but the maximum value is {np.max(img)}"
                )
            img = img.astype(np.float32)
        elif np.issubdtype(img.dtype, np.integer):
            if np.max(img) > 255:
                raise ValueError(
                    "Images with an integer dtype should have values "
                    f"in [0, 255], but the maximum value is {np.max(img)}"
                )
            img = img.astype(np.float32) / 255
            img = np.clip(img, 0, 1)  # In case of rounding errors
        else:
            raise ValueError(
                f"The image has an unsupported dtype: {img.dtype}."
            )

        if np.isnan(img).any():
            raise ValueError(f"The image contains NaN values.")

        try:
            # These should never trigger, but let's do a sanity check
            assert np.min(img) >= 0
            assert np.max(img) <= 1
            assert img.dtype == np.float32
            assert len(img.shape) == 3
            assert img.shape[2] == 3
        except AssertionError as e:
            raise RuntimeError(
                "Internal preprocessing error. "
                "The image does not have the expected format."
            ) from e

        return img

    def encode_image(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocesses the images like CLIP's preprocess() function:
        https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L79

        Args:
            img: PIL image or numpy array

        Returns:
            img: numpy image after resizing, center cropping and normalization.
        """
        img = Preprocessor._image_to_float_array(img)

        img = Preprocessor._crop_and_resize(img)

        # Normalize channels
        img = (img - Preprocessor.NORM_MEAN) / Preprocessor.NORM_STD

        # Mimic the pytorch tensor format for Model class
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)

        return img



default_bpe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"bpe_simple_vocab_16e6.txt.gz")


def bytes_to_unicode() -> Dict[int, str]:
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L16
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word) -> Set[str]:
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L38
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text) -> str:
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L50
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text) -> str:
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L56
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class Tokenizer(object):
    """
    Taken from CLIP.
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L62
    """

    def __init__(self, bpe_path: str = default_bpe_path):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def encode_text(
        self,
        texts: Union[str, Iterable[str]],
        context_length: int = 77,
        truncate: bool = False,
    ) -> np.array:
        """
        Taken from CLIP and reformatted to replace pytorch zeros with numpy zeros.
        Furthermore, this has been wrapped inside the Tokenizer class instead of being
        a separate function.
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L197
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = np.zeros((len(all_tokens), context_length), dtype=np.int32)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context length {context_length}"
                    )
            result[i, : len(tokens)] = np.array(tokens)

        return result
