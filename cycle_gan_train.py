from __future__ import absolute_import, division, print_function, unicode_literals
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

## 識別器
from discriminator import Discriminator
## 生成器
from generator import Generator, ResidualBlock

### TensorboardでＬｏｓｓ確認用
from torch.utils.tensorboard import SummaryWriter
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

#try:
#  # %tensorflow_version only exists in Colab.
#  %tensorflow_version 2.x
##except Exception:
#    pass
  
#%load_ext tensorboard
#%tensorboard --logdir './logs'


### パラメータの設定クラス
class Opts():
    def __init__(self):
        self.start_epoch = 0
#        self.n_epochs = 5 # 23.5.13_2 エポック数
        self.n_epochs = 80 # 23.5.13_1 エポック数
        self.batch_size = 1
        self.dataroot = './img_patches'　# 入力パッチ画像が入ったフォルダ(./img_patches/train/AやBの中にパッチが入っている)
        self.lr = 0.0002 # 学習率
        self.decay_epoch = 40 # 23.5.13_1学習率減衰が始まるエポック(0.0002からスタートし、40epoch~n_epochsの間に学習率が0となる)
#        self.decay_epoch = 3 # 23.5.13_2
        self.size = 256   # サイズ
        self.input_nc = 3 # 入力チャンネル
        self.output_nc = 3# 出力チャンネル
#        self.cpu = True
        self.cpu = False
        self.n_cpu = 8
        self.device_name = "cuda:0"
#        self.device_name = "cpu"
        self.device = torch.device(self.device_name)
        self.load_weight = False

## パラメータのインスタンス
opt = Opts()

### ドメインAとドメインBの画像データセット生成クラス
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        # rootディレクトリ内の./train/A/内のファイルをドメインAのファイルとして呼び出す。
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        # rootディレクトリ内の./train/B/内のファイルをドメインBのファイルとして呼び出す。
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


### 生成画像のバッファ
# 過去の生成データ(50iter分)を保持しておく
class ReplayBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            #
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

### ラムダ
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    
### 重みの初期化関数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

### ロスの保存関数
def save_loss(train_info, batches_done):
    """
    lossの保存
    """
    for k, v in train_info.items():
        writer.add_scalar(k, v, batches_done)
        
if __name__ == '__main__':
    ## 生成器 2つ
    # A から Bを生成するもの
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    # B から Aを生成するもの
    netG_B2A = Generator(opt.output_nc, opt.input_nc)

    ## 識別器
    # Aを識別するもの
    netD_A = Discriminator(opt.input_nc)
    # Bを識別するもの
    netD_B = Discriminator(opt.output_nc)

    # opt.cpuがFalseの場合は、GPUを使用
    if not opt.cpu:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # 重みパラメータ初期化
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # 保存したモデルのロード
    if opt.load_weight is True:
        netG_A2B.load_state_dict(torch.load("./output/netG_A2B.pth", map_location="cuda:0"), strict=False)
        netG_B2A.load_state_dict(torch.load("./output/netG_B2A.pth", map_location="cuda:0"), strict=False)
        netD_A.load_state_dict(torch.load("./output/netD_A.pth", map_location="cuda:0"), strict=False)
        netD_B.load_state_dict(torch.load("./output/netD_B.pth", map_location="cuda:0"), strict=False)

    # 損失関数
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)

    # 入出力メモリ確保
    Tensor = torch.cuda.FloatTensor if not opt.cpu else torch.Tensor
    input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

    # 過去データ分のメモリ確保
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # データローダー
    transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                    transforms.RandomCrop(opt.size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    print("num dataloader= {}".format(len(dataloader)))

    ### ネットワークのサマリー表示
    import torchsummary
    # AからBを生成する生成器のサマリー
    torchsummary.summary(netG_A2B, (opt.input_nc, opt.size, opt.size))
    # Aを識別する識別器のサマリー
    torchsummary.summary(netD_A, (opt.input_nc, opt.size, opt.size))

    for epoch in range(opt.start_epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # モデルの入力
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ##### 生成器A2B、B2Aの処理 #####
            optimizer_G.zero_grad()

            # 同一性損失の計算（Identity loss)
            # G_A2B(B)はBと一致
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A)はAと一致
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # 敵対的損失（GAN loss）
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # サイクル一貫性損失（Cycle-consistency loss）
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # 生成器の合計損失関数（Total loss）
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()

            ##### ドメインAの識別器 #####
            optimizer_D_A.zero_grad()

            # ドメインAの本物画像の識別結果（Real loss）
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # ドメインAの生成画像の識別結果（Fake loss）
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # 識別器（ドメインA）の合計損失（Total loss）
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            ##### ドメインBの識別器 #####
            optimizer_D_B.zero_grad()

            # ドメインBの本物画像の識別結果（Real loss）
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # ドメインBの生成画像の識別結果（Fake loss）
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # 識別器（ドメインB）の合計損失（Total loss）
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            if i % 20 == 0:
                print('Epoch[{}]({}/{}) loss_G: {:.4f} loss_G_identity: {:.4f} loss_G_GAN: {:.4f} loss_G_cycle: {:.4f} loss_D: {:.4f}'.format(
                    epoch, i, len(dataloader), loss_G, (loss_identity_A + loss_identity_B),
                    (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)
                    ))

            train_info = {
                'epoch': epoch, 
                'batch_num': i, 
                'lossG': loss_G.item(),
                'lossG_identity': (loss_identity_A.item() + loss_identity_B.item()),
                'lossG_GAN': (loss_GAN_A2B.item() + loss_GAN_B2A.item()),
                'lossG_cycle': (loss_cycle_ABA.item() + loss_cycle_BAB.item()),
                'lossD': (loss_D_A.item() + loss_D_B.item()), 
                }

            batches_done = (epoch - 1) * len(dataloader) + i
            save_loss(train_info, batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B.pth')       
