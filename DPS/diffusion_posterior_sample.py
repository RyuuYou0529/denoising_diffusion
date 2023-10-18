import sys
sys.path.insert(0, '/home/ryuuyou/Project/denoising-diffusion')

from denoising_diffusion_pytorch import GaussianDiffusion, Sampler
from degradation_model import *

import torch

from tqdm.auto import tqdm

# ====================
# DPS Class
# ====================
class DPS(GaussianDiffusion):
    # 带梯度的p_sample
    # todo: original_func = decorated_func.__wrapped__
    def p_sample_with_grad(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # 计算梯度
    def grad_and_value(self, x_prev, x_0_hat, measurement, t):
        difference = measurement - self.operator.forward(x=x_0_hat, t=t)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
             
        return norm_grad, norm
    
    # DDPM
    def dps_ddpm(self, shape, measurement, scale=1, return_all_timesteps = False):
        batch, device = shape[0], self.betas.device

        x_t = torch.randn(shape, device = device)
        imgs = [x_t]

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None

            x_t = x_t.requires_grad_()

            x_tm1, x_start = self.p_sample_with_grad(x_t, t, self_cond)
            norm_grad, norm  = self.grad_and_value(x_prev=x_t, x_0_hat=x_start, measurement=measurement, t=t)
            x_tm1 -= norm_grad*scale
            
            x_tm1 = x_tm1.detach_()
            imgs.append(x_tm1)
            x_t = x_tm1

        res = x_start if not return_all_timesteps else torch.stack(imgs, dim = 1)

        return res

    # DDIM
    def dps_ddim(self, shape, measurement, scale=1, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_t = torch.randn(shape, device = device)
        imgs = [x_t]

        x_start = None
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            
            x_t = x_t.requires_grad_()

            pred_noise, x_start, *_ = self.model_predictions(x_t, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                x_tm1 = x_start
                imgs.append(x_tm1)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)

            x_tm1 = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            measurement_t = measurement
            # measurement_t = self.sqrt_alphas_cumprod[time]*measurement \
            #                 + self.sqrt_one_minus_alphas_cumprod[time]*self.operator.forward(x=noise)

            norm_grad, norm  = self.grad_and_value(x_prev=x_t, x_0_hat=x_start, measurement=measurement_t, t=time)
            x_tm1 -= norm_grad*scale
            # x_tm1 -= norm_grad*scale*(time/total_timesteps)
            # x_tm1 = self.operator.forward(x=x_tm1, t=time)
            x_tm1 = x_tm1.detach_()

            imgs.append(x_tm1)
            x_t = x_tm1

        res = x_tm1 if not return_all_timesteps else torch.stack(imgs, dim = 1)

        return res

    # DPS采样的封装
    def dps(self, measurement, operator, num_samples=16, scale=1, return_all_timesteps = False):
        b, c, h, w = measurement.shape
        self.operator = operator
        # sample steps小于1k时用DDIM，大于等于1k时用DDPM
        sample_fn = self.dps_ddpm if not self.is_ddim_sampling else self.dps_ddim
        
        # [n, (t), c, h, w]
        if b == 1:
            return sample_fn(shape=(num_samples, c, h, w), measurement=measurement, 
                             scale=scale, return_all_timesteps = return_all_timesteps)
        # [b, n, (t), c, h, w]
        else:
            res = []
            for i in range(b):
                res.append(sample_fn(shape=(num_samples, c, h, w), measurement=measurement[i],
                                     scale=scale, return_all_timesteps = return_all_timesteps))
            return torch.cat(res, dim=0)