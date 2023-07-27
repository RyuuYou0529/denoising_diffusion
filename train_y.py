import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import warnings
warnings.simplefilter("ignore", UserWarning)

import sys

## config

# data_path = "E:\\Project\\datasets\\CARE\\Isotropic_Liver\\train_data\\data_label.npz"
data_path = "/home/share/CARE/Isotropic_Liver/train_data/data_label.npz"

load_checkpoint = False
if_use_path = False
load_path = '/home/share_ssd/ryuuyou/denoising-diffusion/pretrained_y/model_150k_steps_lr1e-5.pt'
##

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
    loss_type = 'l1'
)

trainer = Trainer(
    diffusion,
    path=data_path,
    if_npz=True,
    npz_file_name='Y',
    if_lr_scheduler = False,
    train_batch_size = 32,
    train_lr = 1e-5,
    train_num_steps = 100000,
    save_and_sample_every = 10000,
    gradient_accumulate_every = 2,
    calculate_fid = True,
    results_folder = './pretrained_y'
)

fh = open('log.txt', 'w')
original_stderr = sys.stderr
sys.stderr = fh
original_stdout = sys.stdout
sys.stdout = fh

if load_checkpoint:
    trainer.load(milestone=10, use_path=if_use_path, path=load_path)

trainer.train()

sys.stderr = original_stderr
sys.stdout = original_stdout
fh.close()
