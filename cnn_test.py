import torch
from torch import nn
import time
#加载datetime模块
import datetime

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
torch.cuda.device_count()

x = torch.tensor([1, 2, 3])
x.device

x = torch.ones(2, 3, device='cuda')
x.device

x=torch.arange(12)
x

#可以通过张量的shape属性来访问张量（沿每个轴的长度）的形状 。
x.shape
















