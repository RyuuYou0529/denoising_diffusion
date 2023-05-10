import torch
from torchvision import utils
import torchvision.transforms as T

import math
import cv2 as cv

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

class InverseProblemOperator:
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
            print(f"loading from version {data['version']}")

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

class AnisotropicOperator:
    def __init__(self, img_size: tuple=(128, 128), sigma: tuple=(1.5, 0.5), scale_h: int=3, scale_w: int=9) -> None:
        
        # kernel_size = math.ceil(2*3*max(sigma))
        # if kernel_size % 2 == 0:
        #     kernel_size+=1
        # self.kernel_size = (kernel_size, kernel_size)

        self.down_size = (img_size[0]//scale_h, img_size[1]//scale_w)
        self.up_size = img_size

        self.transform = T.Compose([
            T.GaussianBlur(kernel_size=(5,5), sigma=1),
            T.Resize(size=self.down_size),
            T.Resize(size=self.up_size)
        ])

    def forward(self, x, noise_sigma=0.1, **kwags):
        return self.transform(x+torch.randn_like(x, device=x.device) * noise_sigma)

class DenoiseOperator:
    def __init__(self, device):
        self.device = device
    
    def forward(self, x, **kwags):
        return x