import glob
import random
import os
import numpy as np
import time
import datetime
import sys

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        
        self.model = nn.Sequential(
            # 入力ch：input_nc, 出力ch：64, カーネルサイズ：4, ストライド=2, パディング=1
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, padding=1)
        )
        
    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten, 
        # x.size()[2:]=xは[バッチ,チャンネル,縦,横]の形状なので、『縦,横』を取り出してカーネルサイズとしている,
        # つまり、平均プーリングのカーネルの大きさは識別器の出力データの大きさと同じと思われる。
        # .view()がflatten
        # 念のためチェック
#        print("x.size()[2:]",x.size()[2:])
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)