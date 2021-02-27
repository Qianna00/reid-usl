import inspect
import math
import numbers
import random

import torch
import torchvision.transforms as T
from mmcv.utils import build_from_cfg

from ..builder import PIPELINES

for m in inspect.getmembers(T, inspect.isclass):
    PIPELINES.register_module(name=m[0], module=m[1])


@PIPELINES.register_module()
class RandomAppliedTrans(object):

    def __init__(self, transforms, p=0):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.transforms = T.RandomApply(t, p=p)

    def __call__(self, img):
        return self.transforms(img)


@PIPELINES.register_module()
class RandomResizedCropV2(T.RandomResizedCrop):

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = img.size
        area = height * width

        for _ in range(100):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)

                return y1, x1, h, w
        # Fallback
        return 0, 0, h, w


@PIPELINES.register_module()
class RandomErasingV2(T.RandomErasing):

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random
                erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for _ in range(100):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w],
                                    dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(
                        value, dtype=torch.float32).view(-1, 1,
                                                         1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img
