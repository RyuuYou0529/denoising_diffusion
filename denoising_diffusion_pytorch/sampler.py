import torch
from torchvision import utils

from ema_pytorch import EMA
from accelerate import Accelerator

import math

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

class Sampler(object):
    def __init__(
        self,
        diffusion_model,
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
    
    def sample(self, num_samples = 25, batch_size = 16, return_all_timesteps=False, return_ndarr=False):
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        batches = num_to_groups(num_samples, batch_size)
        
        with torch.no_grad():
            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, return_all_timesteps=return_all_timesteps), batches))
        all_images = torch.cat(all_images_list, dim = 0)
        if return_all_timesteps:
            all_images = torch.moveaxis(all_images, 0, 1)
        
        if return_ndarr:
            return all_images.cpu().numpy()
        else:
            return all_images
        
    def save(self, images, *, path, nrow=None):
        if nrow is None:
            nrow = int(math.sqrt(images.shape[0]))
        # 该方法会把灰度图保存成RGB图
        utils.save_image(images, path, nrow=nrow)