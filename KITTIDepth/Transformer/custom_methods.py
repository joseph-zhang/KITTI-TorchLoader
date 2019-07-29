import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from .base_methods import BaseMethod


"""
This file defines some example transforms.
Each transform method is defined by using BaseMethod class
"""


class TransToPIL(BaseMethod):
    """
    Transform method to convert images as PIL Image.
    """
    def __init__(self):
        BaseMethod.__init__(self)
        self.to_pil = transforms.ToPILImage()

    def __call__(self, data_item):
        self.set_data(data_item)

        if not self._is_pil_image(self.img):
            data_item['img'] = self.to_pil(self.img)
        if not self._is_pil_image(self.depth):
            data_item['depth'] = self.to_pil(self.depth)

        return data_item


class Scale(BaseMethod):
    def __init__(self, mode, size):
        BaseMethod.__init__(self, mode)
        self.scale = transforms.Resize(size, Image.BILINEAR)

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode in ["pair", "Img"]:
            data_item['img'] = self.scale(self.img)
        if self.mode in ["pair", "depth"]:
            data_item['depth'] = self.scale(self.depth)

        return data_item


class RandomHorizontalFlip(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def __call__(self, data_item):
        self.set_data(data_item)

        if random.random() < 0.5:
            data_item['img'] = self.img.transpose(Image.FLIP_LEFT_RIGHT)
            data_item['depth'] = self.depth.transpose(Image.FLIP_LEFT_RIGHT)

        return data_item


class RandomRotate(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    @staticmethod
    def rotate_pil_func():
        degree = random.randrange(-500, 500)/100
        return (lambda pil, interp : F.rotate(pil, degree, interp))

    def __call__(self, data_item):
        self.set_data(data_item)

        if random.random() < 0.5:
            rotate_pil = self.rotate_pil_func()
            data_item['img'] = rotate_pil(self.img, Image.BICUBIC)
            data_item['depth'] = rotate_pil(self.depth, Image.BILINEAR)

        return data_item


class ImgAug(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    @staticmethod
    def adjust_pil(pil):
        brightness = random.uniform(0.8, 1.0)
        contrast = random.uniform(0.8, 1.0)
        saturation = random.uniform(0.8, 1.0)

        pil = F.adjust_brightness(pil, brightness)
        pil = F.adjust_contrast(pil, contrast)
        pil = F.adjust_saturation(pil, saturation)

        return pil

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item['img'] = self.adjust_pil(self.img)

        return data_item


class ToTensor(BaseMethod):
    def __init__(self, mode):
        BaseMethod.__init__(self, mode=mode)
        self.totensor = transforms.ToTensor()

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode == "Img":
            data_item['img'] = self.totensor(self.img)
        if self.mode == "depth":
            data_item['depth'] = self.totensor(self.depth)

        return data_item


class ImgNormalize(BaseMethod):
    def __init__(self, mean, std):
        BaseMethod.__init__(self)
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item['img'] = self.normalize(self.img)

        return data_item
