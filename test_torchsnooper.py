import torch
import torchsnooper

def myfunc(mask, x):
    y = torch.zeros(6)
    y.masked_scatter_(mask, x)
    return y

mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda')
source = torch.tensor([1.0, 2.0, 3.0], device='cuda')