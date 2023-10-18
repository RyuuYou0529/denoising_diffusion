import torch
from torchvision import utils

from ema_pytorch import EMA
from accelerate import Accelerator

import math
import os
import tifffile as tiff
import numpy as np
from matplotlib import pyplot as plt

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
        normalize_result=True
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

        self.normalize_result = normalize_result

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

        # [-1,1] -> [0,1]
        if self.normalize_result:
            all_images = (all_images+1)*0.5
        
        if return_ndarr:
            return all_images.cpu().numpy()
        else:
            return all_images
    
    def dps(self, measurement, operator, num_samples=16, scale=1, return_all_timesteps=False, return_ndarr=False):
        self.step_size = scale
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

        # [-1,1] -> [0,1]
        if self.normalize_result:
            all_images = (all_images+1)*0.5
        
        if return_ndarr:
            return all_images.cpu().numpy()
        else:
            return all_images
    
    def save_tif(self, images, *, folder: str, file_name: str=None, make_grid: bool=True,  **kwags):
        # images [n(b), c, h, w]

        # check folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        # check file_name
        if file_name is not None:
            assert file_name.endswith(('tif','tiff')), 'filename should end with "tif" or "tiff".'
        else:
            sample_type = 'ddim' if self.model.is_ddim_sampling else 'ddpm'
            file_name = f'res_{sample_type}_{self.model.sampling_timesteps}s.tif'

        save_path = os.path.join(folder, file_name)

        # save result stack
        res = images.detach().cpu().numpy()
        tiff.imwrite(save_path, res, **kwags)

        # save result as grid
        if make_grid:
            grid = utils.make_grid(images, nrow=int(np.sqrt(images.shape[0])), padding=0)
            if images.shape[1]==1:
                grid=grid[0]
            index = save_path.find('.tif')
            save_path = save_path[:index]+'_grid'+save_path[index:]
            tiff.imwrite(save_path, grid.detach().cpu().numpy(), **kwags)
        
    def save_tif_with_records(self, images, *, folder: str, file_name: str=None, step=1, padding=0, **kwargs):
        # common: [t, n, c, h, w]
        # dps: [t, n, c, h, w] or [b, t, n, c, h, w]

        # check file_name
        if file_name is not None:
            assert file_name.endswith(('tif','tiff')), 'filename should end with "tif" or "tiff".'
        else:
            sample_type = 'ddim' if self.model.is_ddim_sampling else 'ddpm'
            file_name = f'batch0_{sample_type}_{self.model.sampling_timesteps}s.tif'
            
        # check folder
        if not os.path.exists(folder):
            os.makedirs(folder)

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
                
            tiff.imwrite(os.path.join(folder, file_name), np.asarray(res), **kwargs)
        
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
                        
                file_name = f'batch{index}_{sample_type}_{self.model.sampling_timesteps}s.tif'
                tiff.imwrite(os.path.join(folder, file_name), np.asarray(res), **kwargs)
    
    def save_histc(self, t: torch.Tensor, bins=256, save_path: str=None, file_name=None, if_show: bool=True):
        assert len(t.shape) == 4, "The length of the tensor's shape must be 5."
        b, c, h, w = t.shape

        if file_name is not None:
            assert file_name.endswith(('jpg','png', 'jpeg')), 'filename should end with "jpg" or "png".'
        else:
            sample_type = 'ddim' if self.model.is_ddim_sampling else 'ddpm'
            file_name = f'histc_{sample_type}_{self.model.sampling_timesteps}s.png'

        res = []
        ranges = []
        width = []
        for i in range(b):
            res.append(torch.histc(t[i], bins=bins).cpu().numpy())
            min = t[i].min().cpu().item()
            max = t[i].max().cpu().item()
            ranges.append(np.linspace(min,  max, bins))
            width.append((max-min)/bins)
        res = np.asarray(res)
        
        num = int(np.sqrt(b))
        plt.figure(figsize=(15, 10))
        for row in range(num):
            for col in range(num):
                index = row*num+col
                plt.subplot(num, num, index+1)
                plt.bar(x=ranges[index] ,height=res[index], width=width[index])
        if save_path is not None:
            plt.savefig(os.path.join(save_path, file_name))
        if if_show:
            plt.show()
    
    def ddnm(self, measurement, deg, sigma_y, num_samples=16, return_all_timesteps = False):
        return self.model.ddnm(measurement, deg, sigma_y, num_samples, return_all_timesteps)
        