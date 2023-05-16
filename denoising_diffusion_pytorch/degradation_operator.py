import torch
from torchvision import utils
import torchvision.transforms as T

import numpy as np
import math
import cv2 as cv
from astropy.convolution import Gaussian2DKernel

from accelerate import Accelerator
from ema_pytorch import EMA

from abc import ABC, abstractmethod

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

class Operator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

class InverseProblemOperator(Operator):
    def __init__(
        self,
        diffusion_model,
        checkpoint_path,
        *,
        ema_update_every = 10,
        ema_decay = 0.995,
        amp = False,
        fp16 = False,
        split_batches = True,
    ):
        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.model= self.accelerator.prepare(self.model)

        self.load(checkpoint_path)

    @property
    def device(self):
        return self.accelerator.device
    
    def load(self, path=''):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(path, map_location=device)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from: [version]:{data['version']}; [step]:{data['step']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def ddpm_p_sample(self, x, t: int):
        img, x_start = self.model.p_sample_with_grad(x=x, t=t)
        return img

    def ddim_p_sample(self, x, t: int):
        pass

    def forward(self, x, t: int, **kwags):
        img, x_start = self.model.p_sample_with_grad(x=x, t=t)
        return img

class AnisotropicOperator(Operator):
    def __init__(self, img_shape: tuple, sigma: tuple=(1.5, 0.5), scale: tuple=(3,12), noise_sigma=0.1,) -> None:
        b, c, h, w = img_shape
        self.noise_sigma = noise_sigma

        kernel_size = math.ceil(2*3*max(sigma))
        if kernel_size % 2 == 0:
            kernel_size+=1
        kernel = Gaussian2DKernel(x_stddev=sigma[0], y_stddev=sigma[1], x_size=kernel_size, y_size=kernel_size)
        kernel.normalize()
        kernel = torch.from_numpy(kernel.array).float()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        self.kernel = kernel.repeat(1, c, 1, 1)

        scale_h, scale_w = scale
        down_size = (h//scale_h, w//scale_w)
        up_size = (h, w)
        self.resize = T.Compose([
            # T.Lambda(lambda x:x+torch.randn_like(x, device=x.device)*self.noise_sigma),
            T.Resize(size=down_size, antialias=False),
            T.Resize(size=up_size, antialias=False),
        ])

    def forward(self, x, **kwags):
        x = torch.nn.functional.conv2d(x, self.kernel.to(x.device))
        x = self.resize(x)
        x = x+torch.randn_like(x, device=x.device)*self.noise_sigma
        return x

class DenoiseOperator(Operator):
    def __init__(self):
        pass
    def forward(self, x, **kwags):
        return x