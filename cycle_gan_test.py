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

## PIL変換用
import torchvision

### テスト用のパラメータ
class Opts_test():
    def __init__(self):
        self.size_AX = 11928 # 未染色画像の幅
        self.size_AY = 10368 # 未染色画像の高さ
        self.size_BX = 11648 # HE染色画像の幅
        self.size_BY = 10152 # HE染色画像の高さ
        self.batch_size = 1  # バッチサイズは1に固定する
        self.dataroot = './img_patches' # 画像が置いてあるディレクトリ(この中にAとBのフォルダがあり、それぞれの元画像が置いてある)
        self.size = 256 # パッチのサイズ
        self.input_nc = 3 # 入力チャンネル数
        self.output_nc = 3# 出力チャンネル数
        self.cpu = False 
        self.n_cpu = 8    # cpu数
        self.device_name = "cuda:0" # device_name
        self.device = torch.device(self.device_name)
        self.load_weight = False
        self.generator_A2B = './output/netG_A2B.pth'   # AからBへ変換するモデルのパス
        self.generator_B2A = './output/netG_B2A.pth'   # BからAへ変換するモデルのパス
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
        # チェック用
#        print("self.files_A:",self.files_A) #["./img_patches/test/A/undyed_TMA_kidney_patch_10x10.png","./img_patches/test/A/undyed_TMA_kidney_patch_10x11.png",...]
#        print("self.files_B:",self.files_B) # ["./img_patches/test/A/HE_TMA_kidney_patch_10x10.png","./img_patches/test/A/HE_TMA_kidney_patch_10x11.png",...]


    def __getitem__(self, index):
        file_path_A = self.files_A[index % len(self.files_A)]
        # パッチAの番号(位置情報)
        i_A,j_A = os.path.splitext(os.path.basename(file_path_A))[0].split("_")[-1].split("x")
        file_path_B = self.files_B[index % len(self.files_B)]
        # パッチBの番号(位置情報)
        i_B,j_B = os.path.splitext(os.path.basename(file_path_B))[0].split("_")[-1].split("x")
        
        # チェック用
#        print("Image_file_name:",self.files_A[index % len(self.files_A)])
        self.item_A_o = Image.open(file_path_A)
        item_A = self.transform(self.item_A_o.convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:           
            self.item_B_o = Image.open(file_path_B) 
            item_B = self.transform(self.item_B_o.convert('RGB'))
        # チェック用
#        print("item_A.size:",item_A.size())
        # Aの１枚絵のファイル名
        A_name = "_".join(os.path.splitext(os.path.basename(file_path_A))[0].split("_")[0:3])# 最終1枚絵出力ファイル名
        # Bの１枚絵のファイル名
        B_name = "_".join(os.path.splitext(os.path.basename(file_path_B))[0].split("_")[0:3])# 最終1枚絵出力ファイル名
        
        # パッチA、Bと位置情報A、B、一枚絵の名前A、Bを辞書形式で返す
        return {'A': item_A, 'B': item_B, 'ij_A':[int(i_A),int(j_A)], 'ij_B':[int(i_B),int(j_B)], 'A_name':A_name, 'B_name':B_name}
        
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == '__main__':
    # PILで扱える最大ピクセル数を拡張しておく
    Image.MAX_IMAGE_PIXELS = 10000000000

    ## 合成用の白い画像を用意
    # 腎臓のスライドの大きさ
    # im_A = 未染色＝＞HE染色変換
    im_A  = Image.new("RGB", (opt2.size_AX, opt2.size_AY), (255, 255, 255))
    # im_B = HE染色＝＞未染色変換
    im_B = Image.new("RGB", (opt2.size_BX, opt2.size_BY), (255, 255, 255))
    
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
    input_A = Tensor(opt2.batch_size, opt2.input_nc, opt2.size, opt2.size) # バッチ単位で画像データを貼り付ける器を用意
    input_B = Tensor(opt2.batch_size, opt2.output_nc, opt2.size, opt2.size)# バッチ単位で画像データを貼り付ける器を用意

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
#        print("batch:",batch)
        # Set model input
        # input_Aにbatch["A"]の画像データを貼り付ける(バッチサイズ1なら1枚分,バッチサイズ2なら2枚分)
        # input_Bにbatch["B"]の画像データを貼り付けて(バッチサイズ1なら1枚分,バッチサイズ2なら2枚分)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        
        # 各パッチの位置情報(例：[tensor([10]), tensor([10])])
        i_A = batch['ij_A'][0].numpy()[0] # パッチAのx番号
        j_A = batch['ij_A'][1].numpy()[0] # パッチAのy番号
        x_A = batch['A'].size()[2]        # パッチAの横幅
        y_A = batch['A'].size()[3]        # パッチAの縦幅
        i_B = batch['ij_B'][0].numpy()[0] # パッチBのx番号
        j_B = batch['ij_B'][1].numpy()[0] # パッチBのy番号
        x_B = batch['B'].size()[2]        # パッチBの横幅
        y_B = batch['B'].size()[3]        # パッチBの縦幅
        
        ## 最初の1回だけ実行
        if i == 0:
            # １枚絵の名称
            A_name = batch['A_name'][0]
            B_name = batch['B_name'][0]
            print("A_name:",A_name)
            print("B_name:",B_name)
        
#        print("i_A:",i_A)
#        print("j_A:",j_A)
#        print("i_B:",i_B)
#        print("j_B:",j_B)
#        print("x_A:",x_A)
#        print("y_A:",y_A)
#        print("x_B:",x_B)
#        print("y_B:",y_B)

        # Generate output
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

#        out_img1 = torch.cat([real_A, fake_B], dim=2)
#        out_img2 = torch.cat([real_B, fake_A], dim=2)
        # 生成画像を保管するだけにする
        out_img1 = fake_B # Aから変換した画像
        out_img2 = fake_A # Bから変換した画像
        
#        print("out_img1.size:",out_img1.size())
        ## 画像を白い元画像に貼り付ける
        # バッチ数1でないとエラーになるが、バッチ次元を削減
        out_img1x = out_img1.view(out_img1.size()[1],out_img1.size()[2],out_img1.size()[3])
        out_img2x = out_img2.view(out_img2.size()[1],out_img2.size()[2],out_img2.size()[3])
        # PIL形式に変換
        out_img1_PIL = torchvision.transforms.functional.to_pil_image(out_img1x)        
        out_img2_PIL = torchvision.transforms.functional.to_pil_image(out_img2x)        
        im_A.paste(out_img1_PIL, (int(i_A * x_A), int(j_A * y_A)))
        im_B.paste(out_img2_PIL, (int(i_B * x_B), int(j_B * y_B)))
        
        # Save image files
        save_image(out_img1, os.path.join(A_output,'%04d.png' % (i+1)))
        save_image(out_img2, os.path.join(B_output,'%04d.png' % (i+1)))
        
    # 1枚に合成した画像を保存
    im_A.save(os.path.join(A_output,f'cyclegan_A_{A_name}.png'))
    im_B.save(os.path.join(B_output,f'cyclegan_B_{B_name}.png'))        

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
        