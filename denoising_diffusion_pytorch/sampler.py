import torch
from torchvision import utils

from ema_pytorch import EMA
from accelerate import Accelerator

import math
import os
import tifffile as tiff
import numpy as np

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
            print(f"loading from: [version]:{data['version']}; [step]:{data['step']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def sample(self, num_samples = 25, batch_size = 16, return_all_timesteps=False, return_ndarr=False):
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        batches = num_to_groups(num_samples, batch_size)
        
        with torch.no_grad():
            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, return_all_timesteps=return_all_timesteps), batches))
        # [n, (t), c, h, w]
        all_images = torch.cat(all_images_list, dim = 0)
        if return_all_timesteps:
            # [n, t, c, h, w] -> [t, n, c, h, w]
            all_images = torch.moveaxis(all_images, 0, 1)
        
        if return_ndarr:
            return all_images.cpu().numpy()
        else:
            return all_images
    
    def dps(self, measurement, operator, num_samples=16, scale=1, return_all_timesteps=False, return_ndarr=False):
        # [n, (t), c, h, w] or [b, n, (t), c, h, w]
        all_images = self.ema.ema_model.dps(measurement=measurement, operator=operator, 
                                            num_samples=num_samples, scale=scale, 
                                            return_all_timesteps=return_all_timesteps)
        
        if return_all_timesteps:
            if measurement.shape[0] == 1:
                # [n, t, c, h, w] -> [t, n, c, h, w]
                all_images = torch.moveaxis(all_images, 0, 1)
            else:
                # [b, n, t, c, h, w] -> [b, t, n, c, h, w]
                all_images = torch.moveaxis(all_images, 1, 2)
        
        if return_ndarr:
            return all_images.cpu().numpy()
        else:
            return all_images

    def save_png(self, images, *, path, nrow=None):
        # images [n(b), c, h, w]
        if nrow is None:
            nrow = int(math.sqrt(images.shape[0]))
        # 该方法会把灰度图保存成RGB图
        utils.save_image(images, path, nrow=nrow)

    def save_tif(self, images, *, path, **kwags):
        # images [n(b), c, h, w]
        res = images.detach().cpu().numpy()
        tiff.imwrite(path, res, **kwags)
    
    def save_png_with_records(self, images, *, folder, nrow=None, step=1):
        # common: [t, n, c, h, w]
        # dps: [t, n, c, h, w] or [b, t, n, c, h, w]

        # [t, n, c, h, w]
        if len(images.shape) == 5:
            if nrow is None:
                nrow = int(math.sqrt(images.shape[1]))
            for t, item in enumerate(images):
                if t%step==0:
                    utils.save_image(item, os.path.join(folder, f'{t}.png'), nrow=nrow)
        
        # [b, t, n, c, h, w]
        elif len(images.shape) == 6:
            if nrow is None:
                nrow = int(math.sqrt(images.shape[2]))
            for index, batch in enumerate(images):
                for t, item in enumerate(batch):
                    if t%step==0:
                        utils.save_image(item, os.path.join(folder, f'batch_{index}/',f'{t}.png'), nrow=nrow)
        
    def save_tif_with_records(self, images, *, folder, step=1, padding=0, **kwargs):
        # common: [t, n, c, h, w]
        # dps: [t, n, c, h, w] or [b, t, n, c, h, w]

        # [t, n, c, h, w]
        res = []
        if len(images.shape) == 5:
            t, n, c, h, w = images.shape
            for time, item in enumerate(images):
                if time%step==0 or time==t-1:
                    grid = utils.make_grid(item, nrow=int(math.sqrt(n)), padding=padding)
                    if c == 1:
                        grid=grid[0]
                    res.append(grid.detach().cpu().numpy())
            tiff.imwrite(os.path.join(folder, 'records_one_batch.tif'), np.asarray(res), **kwargs)
        
        # [b, t, n, c, h, w]
        elif len(images.shape) == 6:
            b, t, n, c, h, w = images.shape
            for index, batch in enumerate(images):
                for time, item in enumerate(batch):
                    if time%step==0 or time==t-1:
                        grid = utils.make_grid(item, nrow=int(math.sqrt(n)), padding=padding)
                        if c == 1:
                            grid=grid[0]
                        res.append(grid.detach().cpu().numpy())
                tiff.imwrite(os.path.join(folder, f'multi_batch/', f'batch_{index}_{t}.png'), np.asarray(res), **kwargs)

        