'''

JPEG画像を指定サイズに切り分ける

'''
'''
インポート
'''
import argparse
import numpy as np
import os                # パスからファイル名取得など
import sys
import glob              # フォルダからファイル一覧取得
import cv2               # 背景除去、画像変換
from PIL import Image   # 画像読み込み (CV2でのjpeg読み込み時の値が環境により変化する可能性があるという記事を確認したため)
import yaml              # configファイル読み込み用

'''
引数設定
'''
parser = argparse.ArgumentParser(description='get patches from jpg script')
parser.add_argument('--config_file', type=str, default="get_patches_config.yaml")
args = parser.parse_args() # 引数を解析

'''
関数
'''
#画像における背景（白い部分）割合カウント用関数(レベル0ベース)
def count_white_area (img_w):
    
    ## グレースケール変換
    img_grayimg = cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY)

    ## ぼかし
    kernel = np.ones((10,10),np.float32)/100
    blur_img = cv2.filter2D(img_grayimg,-1,kernel)

    ## 二値化
    # 画像サイズにより適宜調整
    # 高解像度の場合
    bi_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 41, 7)

    ## 近傍の定義
    neiborhood = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],np.uint8)

    ## 縮小、膨張
    # 高解像度の場合
    img_dilate = cv2.dilate(bi_img,neiborhood,iterations=3)
    img_erode = cv2.erode(img_dilate,neiborhood,iterations=20)
    
    ##白（背景）のピクセル数を算出
    # 2値化＋処理したデータの0でないピクセル数をカウント
    white = cv2.countNonZero(img_erode)
    # white_area = パッチ全体のピクセル数の内、白い部分のピクセルの割合
    white_area = white /(img_w.shape[0]*img_w.shape[1])
    
    return white_area


# 画像切り取りの関数(切り取り幅は引数で設定)
def get_patches(base_path, input_name, output_name, x_csize, y_csize, A_pref, B_pref, train_suf, test_suf, use_white_area, w_th, sep):
    
    # 入力フォルダ名の用意
    input_dir = os.path.join(base_path, input_name)
    # 入力フォルダが無ければ作成
    os.makedirs(input_dir, exist_ok = True)

    # 出力フォルダの用意
    out_dir = os.path.join(base_path, output_name)                
    # 出力フォルダが無ければ作成
    os.makedirs(out_dir, exist_ok = True)
    
    # 入力フォルダ内のjpgファイルを取得
    files = glob.glob(input_dir + '/*.jpg')

    '''
    入力フォルダから各ファイルを読み込む
    '''
    for k,file in enumerate(files):
        # 最初の保存パッチだけを検知して、txtファイルにファイル名を記録する
        first_save_flag = 1
        # 入力ファイル名(接尾辞なし)
        file_name = os.path.splitext(os.path.basename(file))[0]
        
        '''
        画像読み込み、PIL -> cv2変換
        '''
        # jpgファイルをPILで開く
        im = Image.open(file)
        
        # 推論のために、読み込んだイメージをRGBA➡RGB変換する
        im = im.convert("RGB")
#        im = im_rgba # 推論しないのでRGBAからRGBに変換する必要なし
        print("im_size:",im.size)

        ## 背景削除用にCV2形式に変換
        # numpy 8bit整数変換
        im_new = np.array(im, dtype=np.uint8)
        # RGB -> BGR変換
#        im_cv2 = cv2.cvtColor(im_new, cv2.COLOR_RGBA2BGRA)
        im_cv2 = cv2.cvtColor(im_new, cv2.COLOR_RGB2BGR)

        # im画像サイズ
        # 余白の端は切り捨てて良い。(全体サイズ÷切り取りサイズ(余り切り捨て)×切り取りサイズ)
        # 例: x = 23040 ÷ 1000 = 23(余り40を切り捨て), 23　× 1000 = 23000
        #     y = 13824 ÷ 1000 = 13(余り824を切り捨て), 13 × 1000 = 13000
        # こうすることで、端を切りとる場合分けのif文を省略することができる。
        img_x = im.size[0]//x_csize * x_csize
        img_y = im.size[1]//y_csize * y_csize
                
        # 画像サイズのyを切り取りサイズで割った回数繰り返す。(例: 画像のy=32を8ずつ切りたい場合=> 32/8 = 4回,事前に割り切れる形に変換済)
        for i in range(int(img_y / y_csize)):
            # 画像サイズのxを切り取りサイズで割った回数繰り返す。(例: 画像のx=24を8ずつ切りたい場合=> 24/8 = 3回,事前に割り切れる形に変換済)
            for j in range(int(img_x / x_csize)):
                
                # cv2画像から1つのパッチを作成(j背景除去に使用)
                # 指定サイズに切り取る(img[top : bottom, left : right])
                im_cv2_c = im_cv2[y_csize * i : y_csize * i + y_csize, x_csize * j : x_csize * j + x_csize]
                
                # 背景除外の関数を用いる場合は、count_white_area関数を呼び出す。
                if use_white_area:
                    white_area = count_white_area(img_w = im_cv2_c)
                else:
                    # 背景除外しないなら、0にしておく。
                    white_area = 0

                ## 白の割合がw_th以上のものは推論～合成の対象外とする。
                if white_area < w_th:
                    ## cycleGAN用に、ファイル名の接頭辞と接尾辞で振り分けるフォルダを４つに分ける。
                    suf  = file_name.split(sep)[-1]     # 接尾辞
                    pref = file_name.split(sep)[0]    # 接頭辞
                    
                    # 全ファイル中１つ目のパッチの場合、ファイルを開く
                    if (k == 0) and first_save_flag == 1:
#                        print(f"k=={k},i=={i},j=={j}")
                        f = open(txt_path,"w")
                    else:
                        f = open(txt_path,"a")
                    # 各ファイルの１つ目の保存パッチの場合
                    if first_save_flag == 1:
                        f.write(f"{file_name}\n")
                        first_save_flag = 0
                    f.write( f"{file_name}_patch_{j}x{i}.png\n")
                    f.write(f"white_area:{white_area}\n")
                    f.close()
                    
                    
                    # 学習用に設定した接尾辞のリスト(train_suf)にファイル名の接尾辞(suf)が含まれている場合、学習用フォルダに振り分ける
                    if suf in train_suf:
                        os.makedirs(os.path.join(out_dir, "train"),exist_ok=True) # 学習用フォルダ作成
                        # ドメインAの接頭辞リスト(A_pref)にファイル名の接頭辞(pref)が含まれている場合、Aフォルダに保存する
                        if pref in A_pref:
                            out_train_A = os.path.join(out_dir, "train", "A")
                            os.makedirs(out_train_A, exist_ok=True)  # ドメインA用フォルダ作成                           
                            cv2.imwrite(os.path.join(out_train_A, f"{file_name}_patch_{j}x{i}.png"), im_cv2_c)                                                        
                        # ドメインBの接頭辞リスト(B_pref)にファイル名の接頭辞(pref)が含まれている場合、Bフォルダに保存する
                        elif pref in B_pref:
                            out_train_B = os.path.join(out_dir, "train", "B")
                            os.makedirs(out_train_B, exist_ok=True)  # ドメインB用フォルダ作成
                            cv2.imwrite(os.path.join(out_train_B, f"{file_name}_patch_{j}x{i}.png"), im_cv2_c)                                                        
                        # ファイル名の接頭辞(pref)がA_prefにもB_prefにも含まれない場合、メッセージを出力してプログラムを終了する
                        else:
                            print(f"pref={pref},A_pref={A_pref},B_pref={B_pref}。ファイル名の接尾辞と接頭辞が正しく設定されていません。\nファイルの接頭辞、接尾辞、セパレータを確認してみてください")
                            sys.exit()
                    # テスト用に設定した接尾辞のリスト(test_suf)にファイル名の接尾辞(suf)が含まれている場合、テスト用フォルダに振り分ける                        
                    elif suf in test_suf:
                        os.makedirs(os.path.join(out_dir, "test"),exist_ok=True)  # テスト用フォルダ作成
                        # ドメインAの接頭辞リスト(A_pref)にファイル名の接頭辞(pref)が含まれている場合、Aフォルダに保存する
                        if pref in A_pref:
                            out_test_A = os.path.join(out_dir, "test", "A")
                            os.makedirs(out_test_A, exist_ok=True)  # ドメインA用フォルダ作成                                  
                            cv2.imwrite(os.path.join(out_test_A, f"{file_name}_patch_{j}x{i}.png"), im_cv2_c)                            
                        # ドメインBの接頭辞リスト(B_pref)にファイル名の接頭辞(pref)が含まれている場合、Bフォルダに保存する
                        elif pref in B_pref:
                            out_test_B = os.path.join(out_dir, "test", "B")
                            os.makedirs(out_test_B, exist_ok=True)  # ドメインB用フォルダ作成
                            cv2.imwrite(os.path.join(out_test_B, f"{file_name}_patch_{j}x{i}.png"), im_cv2_c)                                                        
                        # ファイル名の接頭辞(pref)がA_prefにもB_prefにも含まれない場合、メッセージを出力してプログラムを終了する                            
                        else:
                            print(f"pref={pref},A_pref={A_pref},B_pref={B_pref}。ファイル名の接尾辞と接頭辞が正しく設定されていません。\nファイルの接頭辞、接尾辞、セパレータを確認してみてください")
                            sys.exit()
                    # ファイル名の接尾辞(suf)がtrain_sufにもtest_sufにも含まれない場合、メッセージを出力してプログラムを終了する                                                        
                    else:
                        print(f"suf={pref},train_suf={train_suf},test_suf={test_suf}。ファイル名の接尾辞と接頭辞が正しく設定されていません。\nファイルの接頭辞、接尾辞、セパレータを確認してみてください")
                        sys.exit()

'''
プログラム実行部分
'''
# importされた時に自動的に実行されないようにする。
if __name__ == '__main__':
    # configファイルへのパス(カレントディレクトリベース)
    config_path = os.path.join("./", args.config_file)
    # config読み込み
    config = yaml.safe_load(open(config_path, 'r'))
    # 各変数をconfigファイルから代入
    base_path  = config['path']['base_path']
    input_name = config['path']['input_name']
    output_name= config['path']['output_name']
    txt_path = config['path']['txt_path']                   ## テキストファイルに保存するパッチの情報を記録する
    sep = config['data']['sep']
    A_pref = config['data']['A_pref']
    B_pref = config['data']['B_pref']
    train_suf = config['data']['train_suf']
    test_suf  = config['data']['test_suf']
    x_csize = config['cut']['x_csize']
    y_csize = config['cut']['y_csize']
    w_th = config['background']['w_th']
    use_white_area = config['background']['use_white_area']    

    
    # プログラム実行
    get_patches(base_path=base_path, input_name=input_name, output_name = output_name, x_csize = x_csize, y_csize = y_csize, A_pref=A_pref, B_pref=B_pref, train_suf=train_suf, test_suf=test_suf, use_white_area=use_white_area, w_th=w_th, sep=sep)
