import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter("ignore", UserWarning)

import sys

## config

# data_path = "E:\\Project\\datasets\\CARE\\Isotropic_Liver\\train_data\\data_label.npz"
data_path = "/home/share/CARE/Isotropic_Liver/train_data/data_label.npz"

load_milestone = True
# load_path = './pretrained/model_100k_steps_lr1e-5.pt'
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
    npz_file_name='X',
    if_lr_scheduler = False,
    train_batch_size = 32,
    train_lr = 1e-5,
    train_num_steps = 150000,
    save_and_sample_every = 10000,
    gradient_accumulate_every = 2,
    calculate_fid = True,
    results_folder = './results_x'
)

fh = open('log.txt', 'w')
original_stderr = sys.stderr
sys.stderr = fh
original_stdout = sys.stdout
sys.stdout = fh

if load_milestone:
    trainer.load(milestone=10, use_path=False)
trainer.train()

sys.stderr = original_stderr
sys.stdout = original_stdout
fh.close()
