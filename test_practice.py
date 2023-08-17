# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:38:19 2023

@author: yue
"""
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn
# dir(models)

alexnet = models.AlexNet()
resnet = models.resnet101(weights=10)



preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# img=Image.open("jm.jpg")
img2=Image.open("panda2.jpg")
img2
print(img2)



img_t = preprocess(img2)    #预处理
batch_t =torch.unsqueeze(img_t,0)
resnet.eval()       #网络变成推理模式
out=resnet(batch_t)
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index=torch.max(out,1)
percentage=torch.nn.functional.softmax(out,dim=1)[0]*100
labels[index[0]],percentage[index[0]].item()

_,indices=torch.sort(out,descending=True)
print([(labels[idx],percentage[idx].item()) for idx in indices[0][:5]])   #输出前5可能性

################################################################################

