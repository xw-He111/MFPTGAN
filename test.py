import os
import datetime
import torch
import numpy as np
from torch.cuda.amp import GradScaler as GradScaler
from model import TDColor
# from utils import print_network
from dataset import Dataset, tensor_lab2rgb
from torch.utils.data import DataLoader
import warnings
import random
import cv2
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F
from evaluate import psnr_ssim, calculate_cf, fid
from thop import profile


warnings.filterwarnings("ignore")

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '4321'

# ------------------------------------------参数设置------------------------------------------
# 生成器参数
encoder_name = 'convnext-l'
decoder_name = 'MultiScaleColorDecoder'
num_input_channels = 3
input_size = (256, 256)
nf = 512
last_norm = 'Spectral'
do_normalize = False
num_queries_l = 100
num_queries_ab = 50
num_scales = 3
dec_layers = 9
encoder_from_pretrain = False

data_path = './'

model_path = 'G:/paper_re/0_MFPTGAN_ab/checkpoints/GF7_1202/net_g_latest.pth'
save_path = 'G:/paper_re/0_MFPTGAN_ab/checkpoints/GF7_1202/results_latest'
label_path = 'G:/datasets/GF7/test_label'
eval_path = 'G:/paper_re/0_MFPTGAN_ab/checkpoints/GF7_1202/eval_latest_fwd.txt'

device = torch.cuda.current_device()

# ------------------------------------------模型定义------------------------------------------

generator = TDColor(encoder_name,
                decoder_name,
                num_input_channels,
                input_size,
                nf,
                last_norm,   # 这个参数代表什么
                do_normalize,
                num_queries_l,
                num_queries_ab,
                num_scales,
                dec_layers,
                encoder_from_pretrain=encoder_from_pretrain)

generator = generator.to(device)

generator.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')), # ['generator'],
            strict=False)

# num_params_g = print_network(generator)
# print('[Network generator] Total number of parameters : %.3f M' % (num_params_g))

generator.eval()

time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')

pra_input = [torch.randn(1,3,256,256).to("cuda")]
flops, params = profile(generator, inputs=pra_input)
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Fparams = ' + str(params/1000**2) + 'Million')

f = open(eval_path, 'w')
f.write(f'FLOPs = {flops/1000**3} G\nFparams = {params/1000**2} Million\n')

# ------------------------------------------数据设置------------------------------------------
dataset = Dataset(data_path, list(input_size), 'test')
dataset_size = len(dataset)
print('The number of testing images = %d' % dataset_size)

data_test = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=True)

if not os.path.exists(save_path):
    os.makedirs(save_path)

pbda = tqdm(enumerate(data_test), total=len(data_test), bar_format="{l_bar}{bar:10}{r_bar}")

psnr_total = 0
ssim_total = 0
cf_total = 0


for i, test_data in pbda:
    image_name = test_data['input_path'][0].split(os.sep)[-1]

    input_l = test_data['input_l'].to(device)
    input_ab = test_data['input_ab'].to(device)
    input = tensor_lab2rgb(torch.cat([input_l, input_ab], dim=1))

    label_l = test_data['label_l'].to(device)
    label_ab = test_data['label_ab'].to(device)
    label = tensor_lab2rgb(torch.cat([label_l, label_ab], dim=1))

    out_l, out_ab = generator(input)
    out = tensor_lab2rgb(torch.cat([out_l, out_ab], dim=1))
    out = out[0].detach().cpu().float().numpy().transpose(1, 2, 0)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # out = np.clip(out * 255.0, 0, 255).round().astype(np.uint8)
    save_dir = os.path.join(save_path, image_name)
    cv2.imwrite(save_dir, out)

    label = rearrange(label, 'bs c h w -> bs h w c')
    label = label[0].detach().cpu().float().numpy()
    label = (label * 255.0).clip(0, 255).astype(np.uint8)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

    psnr, ssim, _ = psnr_ssim(label, out)
    psnr_total += psnr
    ssim_total += ssim

    cf = calculate_cf(out)
    cf_total += cf

    f.write(f'{image_name}:\tpsnr: {psnr:.4f}\tssim: {ssim:.4f}\tcf: {cf}\n')
                    
psnr_mean = psnr_total / len(data_test)
ssim_mean = ssim_total / len(data_test)
cf_mean = cf_total / len(data_test)

fid_value = fid([label_path, save_path])

f.write(f'psnr_mean: {psnr_mean:.4f}\nssim_mean: {ssim_mean:.4f}\n'
            f'cf_mean: {cf_mean:.4f}\nfid: {fid_value:.4f}\n')

print(f'psnr_mean: {psnr_mean:.4f}\nssim_mean: {ssim_mean:.4f}\n'
          f'cf_mean: {cf_mean:.4f}\nfid: {fid_value:.4f}\n')
