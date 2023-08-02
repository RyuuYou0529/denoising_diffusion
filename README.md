A diffusion model demo based on https://github.com/lucidrains/denoising-diffusion-pytorch

denoising_diffusion_pytorch/

|--- denoising_diffusion_pytorch.py # 网络结构、Diffusion过程、训练

|--- diffusion_posterior_sample.py # DPS过程

|--- degradation_operator.py # 退化模型

|--- sampler.py # 采样

|--- version.py # 保存模型时的version信息

train.py #训练高分辨率图像的diffusion model

inference.ipynb # DPS推理

sample.ipynb # 测试sampler.py

env.yaml # conda环境配置
