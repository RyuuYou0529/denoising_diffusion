import torch

import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import os

from DPS.helper_function import *
from DPS.conditioning_method import *
from DPS.mean_processer import *
from DPS.variance_processer import *
from DPS.noiser import *
from DPS.operator import *

class GaussianDiffusion:
    def __init__(self, 
                 model_hr, operator, noiser,
                 timesteps, 
                 beta_schedule,
                 given_betas = None,
                 schedule_fn_kwargs=dict()) -> None:
        
        self.model = model_hr
        self.operator = operator
        self.noiser = noiser,

        if given_betas is None:
            if beta_schedule == 'linear':
                beta_schedule_fn = linear_beta_schedule
            elif beta_schedule == 'cosine':
                beta_schedule_fn = cosine_beta_schedule
            elif beta_schedule == 'sigmoid':
                beta_schedule_fn = sigmoid_beta_schedule
            else:
                raise ValueError(f'unknown beta schedule {beta_schedule}')
            self.betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs).cpu().numpy()
        else:
            self.betas = given_betas

        self.num_timesteps = int(self.betas.shape[0])

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = EpsilonXMeanProcessor(betas=self.betas)
        self.var_processor = LearnedRangeVarianceProcessor(betas=self.betas)

    # 计算后验均值和方差
    def p_mean_variance(self, x, t):
        model_output = self.model(x=x, time=t)

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)

        model_var_values = model_output
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    # 单次后验采样,调用了p_mean_variance
    def p_sample(self, model, x, t):
        raise NotImplementedError
    
    # 后验采样循环,调用了p_sample
    def p_sample_loop(self, x_start, measurement, record, save_root):
        img = x_start       
        device = x_start.device

        cond_method = PosteriorSampling(operator=self.operator, noiser=self.noiser)
        measurement_cond_fn = cond_method.conditioning

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            
            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time)

            img, distance = measurement_cond_fn(x_t=out['sample'],
                                      measurement=measurement,
                                      x_prev=img,
                                      x_0_hat=out['pred_xstart'],
                                      t=idx)
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))
            
            img = img.detach_()
        
        return img

# ===============================================================================

class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_steps = original_num_steps

    def __call__(self, x, time, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=time.device, dtype=time.dtype)
        new_ts = map_tensor[time]
        return self.model(x, new_ts, **kwargs)

def space_timesteps(num_timesteps, section_counts):
    section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

# ===============================================================================

class DDPM(GaussianDiffusion):
    def p_sample(self, x, t):
        out = self.p_mean_variance(x, t)
        sample = out['mean']

        noise = torch.randn_like(x)
        if t[0] != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out['log_variance']) * noise

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

class DDIM(GaussianDiffusion):
    def __init__(self, ddim_steps: int, **kwargs):
        use_timesteps = space_timesteps(kwargs["timesteps"], ddim_steps)
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = kwargs["timesteps"]

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        
        kwargs["given_betas"] = np.array(new_betas)

        super().__init__(**kwargs)

        self.model = self._wrap_model(self.model)

    def p_mean_variance(
        self, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(*args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.original_num_steps
        )

    def p_sample(self, x, t, eta=0.0):
        out = self.p_mean_variance(x, t)
        
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        
        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t[0] != 0:
            sample += sigma * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

# ===============================================================================
