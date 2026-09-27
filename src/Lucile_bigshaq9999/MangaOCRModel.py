import gc
import logging
from math import floor, ceil
import numpy as np
from PIL import Image
import torch
from manga_ocr import MangaOcr

logger = logging.getLogger(__name__)


def transform_img_to_PIL(img):
    return Image.fromarray(img)


class MangaOCRModel:
    def __init__(self):
        self.mocr = None
        self.device = "cpu"

    def load_model(self, device="auto", force_cpu=False):
        if force_cpu:
            target_device = "cpu"
        elif device == "auto" or device is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device

        logger.info(f"Initializing MangaOCR with target device: {target_device}")
        self.mocr = MangaOcr(force_cpu=(target_device == "cpu"))
        self.device = "cpu"

        if target_device != "cpu":
            try:
                self.mocr.model.to(target_device)
                self.device = target_device
                logger.info(f"MangaOCR successfully loaded on {self.device}")
            except Exception as e:
                logger.warning(
                    f"Failed to move MangaOCR model to {target_device}: {e}. Falling back to CPU."
                )
                self.mocr.model.to("cpu")
                self.device = "cpu"

    def predict(self, img, bboxes):
        if self.mocr is None:
            raise TypeError("Model is not loaded yet")

        image_rgb = np.array(img)
        cropped_image_list = []

        for box in bboxes:
            cropped_image = image_rgb[
                floor(box[1]): ceil(box[3]), floor(box[0]): ceil(box[2]), :
            ]
            cropped_image_list.append(cropped_image)

        text_ocr_list = []

        for cropped_img in cropped_image_list:
            if isinstance(cropped_img, Image.Image):
                pil_img = cropped_img
            else:
                pil_img = transform_img_to_PIL(cropped_img)

            try:
                text = self.mocr(pil_img)
            except Exception as e:
                if self.device != "cpu":
                    logger.warning(
                        f"MangaOCR prediction failed on {self.device}: {e}. Falling back to CPU."
                    )
                    self.device = "cpu"
                    try:
                        self.mocr.model.to("cpu")
                        text = self.mocr(pil_img)
                    except Exception as cpu_e:
                        logger.error(f"MangaOCR failed on CPU fallback: {cpu_e}")
                        text = ""
                else:
                    logger.error(f"MangaOCR prediction error: {e}")
                    text = ""

            text_ocr_list.append(text)

        return text_ocr_list

    def unload_model(self):
        del self.mocr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.mocr = None
        self.device = "cpu"

            
