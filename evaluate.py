import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import os

from pytorch_fid import fid_score
import numpy as np
from tqdm import tqdm

# 计算两张图像的psnr和ssim
def psnr_ssim(ori_img, test_img):
    psnr = peak_signal_noise_ratio(ori_img, test_img)
    ssim = structural_similarity(ori_img, test_img, channel_axis=2)
    mse = mean_squared_error(ori_img, test_img)

    return psnr, ssim, mse

def calculate_cf(img, **kwargs):
    """Calculate Colorfulness.
    """
    (B, G, R) = cv2.split(img.astype('float'))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R+G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def fid(image_paths):
    fid_value = fid_score.calculate_fid_given_paths(image_paths, batch_size=32, 
                                                    device='cuda:0', 
                                                    dims=2048, 
                                                    num_workers=0)
    
    return fid_value

def eval(image_paths, txt_path):
    real_imgs = os.listdir(image_paths[0])
    fake_imgs = os.listdir(image_paths[1])

    f = open(txt_path, 'w')

    psnr_sum = 0
    ssim_sum = 0
    mse_sum = 0
    cf_sum = 0

    pbda = tqdm(enumerate(fake_imgs), total=len(fake_imgs), bar_format="{l_bar}{bar:10}{r_bar}")

    for i, name in pbda:

        real_img = cv2.imread(image_paths[0] + os.sep + real_imgs[i])
        fake_img = cv2.imread(image_paths[1] + os.sep + fake_imgs[i])

        psnr, ssim, mse = psnr_ssim(real_img, fake_img)  # 每个通道的psnr, ssim, mse

        cf = calculate_cf(fake_img)

        f.write(f'{name}:\tpsnr: {psnr:.4f}\tssim: {ssim:.4f}\tcf: {cf:.4f}\n')

        # print(paths[1] + fake_imgs[i])

        psnr_sum += psnr
        ssim_sum += ssim
        mse_sum += mse
        cf_sum += cf

    psnr_mean = psnr_sum / len(fake_imgs)
    ssim_mean = ssim_sum / len(fake_imgs)
    mse_mean = mse_sum / len(fake_imgs)
    cf_mean = cf_sum / len(fake_imgs)

    # FID计算
    fid_value = fid_score.calculate_fid_given_paths(image_paths, batch_size=32, 
                                                    device='cuda:0', 
                                                    dims=2048, 
                                                    num_workers=0)

    f.write(f'psnr_mean: {psnr_mean:.4f}\nssim_mean: {ssim_mean:.4f}\n'
            f'cf_mean: {cf_mean:.4f}\nfid: {fid_value:.4f}\n')

    print(f'psnr_mean: {psnr_mean:.4f}\nssim_mean: {ssim_mean:.4f}\n'
          f'cf_mean: {cf_mean:.4f}\nfid: {fid_value:.4f}\n')

if __name__ == "__main__":
    paths = ['E:/datasets/GF7/test_label/', 
             './checkpoints/GF7_0808/results/']
    
    txt_path = './checkpoints/GF7_0808/evaluate.txt'

    eval(paths, txt_path)

    

