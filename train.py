import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch import npz_dataset

import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
import os

## === config ===
data_path = "/home/share/CARE/Isotropic_Liver/train_data/data_label.npz"

load_checkpoint = True
use_specified_path = False
if use_specified_path:
    load_path = '/home/share_ssd/ryuuyou/denoising-diffusion/pretrained_y/model_150k_steps_lr1e-5.pt'
else:
    milestone=10

results_folder = os.path.join('./checkpoints')
## === === ===

## === model ===
model = Unet(
    dim = 64,
    channels=1,
    dim_mults = (1, 2, 4, 8)
)
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,
    sampling_timesteps = 250,
    loss_type = 'l1',
    auto_normalize=True
)
## === === ===

## === data ===
dataset = npz_dataset(path=data_path,
                      npz_file_name='Y',
                      normalize_mode='min_max')
## === === ===

## === trainer ===
trainer = Trainer(
    diffusion,
    dataset=dataset,
    train_batch_size = 32,
    train_lr = 1e-5,
    train_num_steps = 150000,
    save_and_sample_every = 10000,
    gradient_accumulate_every = 2,
    calculate_fid = True,
    results_folder = results_folder
)

fh = open('log.txt', 'w')
original_stderr = sys.stderr
sys.stderr = fh
original_stdout = sys.stdout
sys.stdout = fh

if load_checkpoint:
    if use_specified_path:
        trainer.load(use_path=use_specified_path, path=load_path)
    else:
        trainer.load(milestone=milestone)
## === === ===

trainer.train()

sys.stderr = original_stderr
sys.stdout = original_stdout
fh.close()
