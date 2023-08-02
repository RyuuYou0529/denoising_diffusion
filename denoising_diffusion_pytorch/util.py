import torch
import numpy as np
from matplotlib import pyplot as plt

def tensor_info(t: torch.Tensor):
    print(f'shape: {t.shape} \ntype: {t.dtype} \nmax: {t.max()} \
          \nmin: {t.min()} \nmean: {t.mean()} \nstd: {t.std()}')

def histc(t: torch.Tensor, save_path: str=None, if_show: bool=True):
    b, c, h, w = t.shape
    res = []
    for i in range(b):
        res.append(torch.histc(t[i], bins=256).cpu().numpy())
    res = np.asarray(res)
    
    num = int(np.sqrt(b))
    plt.figure(figsize=(15, 10))
    for row in range(num):
        for col in range(num):
            plt.subplot(num, num, row*num+col+1)
            plt.bar(x=np.arange(256) ,height=res[row*num+col])
    if save_path is not None:
        plt.savefig(save_path)
    if if_show:
        plt.show()

