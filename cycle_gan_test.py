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
#from visdom import Visdom

import itertools
from PIL import Image

# グラフ
import matplotlib.pyplot as plt
import japanize_matplotlib
japanize_matplotlib.japanize()

## 識別器
from discriminator import Discriminator

## 生成器
from generator import ResidualBlock, Generator

### テスト用のパラメータ
class Opts_test():
    def __init__(self):
        self.batch_size = 1 # 23.5.1設定
#        self.batch_size = 10
        self.dataroot = './img_patches'
        self.size = 256
#        self.size = 1000
        self.input_nc = 3
        self.output_nc = 3
        self.cpu = False
        self.n_cpu = 8
        self.device_name = "cuda:0"
        self.device = torch.device(self.device_name)
        self.load_weight = False
        self.generator_A2B = './output/netG_A2B.pth'
        self.generator_B2A = './output/netG_B2A.pth'
        self.cuda = True

## テスト用パラメータのインスタンス
opt2 = Opts_test()

### ドメインAとドメインBの画像データセット生成クラス
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        
        self.item_A_o = Image.open(self.files_A[index % len(self.files_A)])        
        item_A = self.transform(self.item_A_o.convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:           
            self.item_B_o = Image.open(self.files_B[index % len(self.files_B)]) 
            item_B = self.transform(self.item_B_o.convert('RGB'))

        return {'A': item_A, 'B': item_B}
        
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == '__main__':
    ### ネットワークの呼び出し
    # 出力フォルダ
    # 3microと6micro
#    A_output = "output/A_256"
#    B_output = "output/B_256"
    # 学習してない4micro
    A_output = "output/A"
    B_output = "output/B"
    # 生成器G
    netG_A2B = Generator(opt2.input_nc, opt2.output_nc)
    netG_B2A = Generator(opt2.output_nc, opt2.input_nc)

    # CUDA
    if opt2.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt2.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt2.generator_B2A))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt2.cuda else torch.Tensor
    input_A = Tensor(opt2.batch_size, opt2.input_nc, opt2.size, opt2.size)
    input_B = Tensor(opt2.batch_size, opt2.output_nc, opt2.size, opt2.size)

    # Dataset loader
    transforms_ = [transforms.Resize(int(opt2.size*1.0), Image.BICUBIC), 
                    transforms.RandomCrop(opt2.size), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    # 3microと6microから生成する場合
#    dataset = ImageDataset(opt2.dataroot, transforms_=transforms_, mode='train')
    # 4microからFAKE 3micro or FAKE 6microを生成する場合
    dataset = ImageDataset(opt2.dataroot, transforms_=transforms_, mode='test')
    dataloader = DataLoader(dataset, batch_size=opt2.batch_size, shuffle=False, num_workers=opt2.n_cpu)


    # 出力用フォルダ生成
    if not os.path.exists(A_output):
        os.makedirs(A_output)
    if not os.path.exists(B_output):
        os.makedirs(B_output)

    ### テストの実行
    ##### 生成器Gによる画像生成　#####
    from torchvision.utils import save_image

    num_create = 100

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

#        out_img1 = torch.cat([real_A, fake_B], dim=2)
#        out_img2 = torch.cat([real_B, fake_A], dim=2)
        # 生成画像を保管するだけにする
        out_img1 = fake_B
        out_img2 = fake_A
               
        # Save image files
        save_image(out_img1, os.path.join(A_output,'%04d.png' % (i+1)))
        save_image(out_img2, os.path.join(B_output,'%04d.png' % (i+1)))

'''        
        ### 画像保存
#        A_origin,B_origin = dataset.origin()
        ## 3μ=>6μ
        fig = plt.figure(figsize=(6,6))        
        fig.suptitle("3μ → 6μ変換")

        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        
##        real_A_save = (((input_A.copy_(batch['A'])).to("cpu").detach().numpy()).squeeze().transpose(1,2,0)).astype(np.uint8)
#        real_A_save = (((A_origin).to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8)
#        old_0,old_1,old_2 = real_A_save[:,:,0].copy(),real_A_save[:,:,1].copy(),real_A_save[:,:,2].copy()
#        real_A_save[:,:,0], real_A_save[:,:,1], real_A_save[:,:,2] = old_1, old_2, old_0 
        real_A_save = ((real_A.to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8)
        fake_B_save = ((fake_B.to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8)
 #       if i == 0:
 #           print("real_A:\n",real_A_save)
 #           print("fake_B:\n",fake_B_save)
        
        ax1.imshow(real_A_save) # 元画像
        ax1.set_title("original")
        ax1.set_axis_off()
        
        ax2.imshow(fake_B_save) # 元画像
        ax2.set_title("fake")
        ax2.set_axis_off()
        
        # 保存
        fig.savefig(f'output/A/{i+1}_figures.png',facecolor="white")        
        
        ## 6μ=>3μ
        fig2 = plt.figure(figsize=(6,12))
        fig2.suptitle("6μ → 3μ変換")
        
        ax3 = fig2.add_subplot(2,1,1)
        ax4 = fig2.add_subplot(2,1,2)
        
#        real_B_save = (((input_B.copy_(batch['B'])).to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8)
        real_B_save = ((real_B.to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8)
        fake_A_save = ((fake_A.to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8)
#        if i == 0:
#            print("real_B:\n",real_B_save)
#            print("fake_A:\n",fake_A_save)
        
        ax3.imshow(real_B_save) # 元画像
        ax3.set_title("original")
        ax3.set_axis_off()
        
        ax4.imshow(fake_A_save) # 元画像
        ax4.set_title("fake")
        ax4.set_axis_off()
        
        # 保存
        fig2.savefig(f'output/B/{i+1}_figures.png',facecolor="white")        
        
        # メモリ解放
        plt.clf()
        plt.close()
        

        if i > num_create:
            break

    # A->Bの生成画像
    import matplotlib.pyplot as plt
    img = plt.imread('output/A/0001.png')
    plt.imshow(img)

    # B->Aの生成画像
    img = plt.imread('output/B/0001.png')
    plt.imshow(img)

'''
        