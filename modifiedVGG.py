import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        features=list(vgg16(pretrained=True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features=nn.ModuleList(features).eval()
    def forward(self,x):
        results=[]
        for index,sub_module in enumerate(self.features):
            x=sub_module(x)
            if index in {3,8,15,22}:
                results.append(x)

        middle_outputs=namedtuple("middle_outputs",['relu1_2','relu2_2','relu3_3','relu4_3'])
        return middle_outputs(*results)
