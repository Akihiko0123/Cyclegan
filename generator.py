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

# ResidualBlockの定義
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)
    
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()
        
        self.model = nn.Sequential(
            ### エンコーダ部分
            # 入力データに対して、鏡のように左右逆さのデータを指定数行列前後上下に追加する。
            # 例：nn.ReflectionPad2d(3)の場合、左右逆さの3行3列を元のデータの上下左右斜め上下に追加する。
            # 参考：https://qiita.com/students/items/f5a2486eeeeb89e6df1d
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            # バッチ正規化の改善選択肢の1つ
            # 参考:https://qiita.com/sho12333/items/8a23f0fcdc03b91cbc04
            nn.InstanceNorm2d(64),
            # inplace=Trueで、入力値そのものを書き換える。
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            ### トランスフォーマ部分
            
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            
            ### デコーダ部分

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)
    

