import torch
from torchvision import utils
import torchvision.transforms as T
import torch.nn as nn

import numpy as np
import math
import cv2 as cv
from astropy.convolution import Gaussian2DKernel

from abc import ABC, abstractmethod

from .u_net.model import UNet

from .self_net import Self_net_architecture

def exists(x):
    return x is not None
def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 抽象父类
class Operator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass


# 去噪退化
class DenoiseOperator(Operator):
    def __init__(self):
        pass
    def forward(self, x, **kwags):
        return x

# 高斯模糊
class GaussialBlurOperator(Operator):
    def __init__(self, img_shape: tuple, sigma: float=3.0) -> None:
        b, c, h, w = img_shape

        kernel_size = math.ceil(2*3*sigma)
        if kernel_size % 2 == 0:
            kernel_size+=1

        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, x, **kwags):
        x = self.blur(x)
        return x
    
# 各向异性退化
class AnisotropicOperator(Operator):
    def __init__(self, img_shape: tuple, sigma: tuple=(3, 1), scale: tuple=(1,3), noise_sigma=0.1) -> None:
        b, c, h, w = img_shape
        self.noise_sigma = noise_sigma

        kernel_size = math.ceil(2*3*max(sigma))
        if kernel_size % 2 == 0:
            kernel_size+=1
        kernel = Gaussian2DKernel(x_stddev=sigma[0], y_stddev=sigma[1], x_size=kernel_size, y_size=kernel_size)
        kernel.normalize()
        kernel = torch.from_numpy(kernel.array).float()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        self.kernel = kernel.repeat(c, c, 1, 1)

        scale_h, scale_w = scale
        down_size = (h//scale_h, w//scale_w)
        up_size = (h, w)
        self.resize = T.Compose([
            T.Resize(size=down_size, antialias=False),
            T.Resize(size=up_size, antialias=False),
        ])
        # self.normlize = T.Normalize(mean=[0.5 for _ in range(c)], std=[0.5 for _ in range(c)])

    def forward(self, x, **kwags):
        x = torch.nn.functional.conv2d(x, self.kernel.to(x.device), padding='same')
        x = self.resize(x)
        x = x+torch.randn_like(x, device=x.device)*self.noise_sigma
        # x = self.normlize(x)
        return x
