import os
import datetime
from time import time
import torch
import numpy as np
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import DataLoader
from model import TDColor
from discriminator import DynamicUNetDiscriminator
from dataset import Dataset, tensor_lab2rgb, color_enhacne_blend
from losses import L1Loss, GANLoss, ssim, PerceptualLoss
# from vgg import Vgg16
from torch.utils.tensorboard.writer import SummaryWriter
import warnings
import yaml
from tqdm import tqdm
import shutil
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from lr_scheduler import MultiStepRestartLR
from einops import rearrange
from evaluate import psnr_ssim, calculate_cf, fid
import math
import cv2
from torchsummary import summary

warnings.filterwarnings("ignore")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '4321'

# 设置全局随机种子
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def train():

    start_time = time()

    device = torch.cuda.current_device()

    config = read_yaml('./config.yaml')
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
    shutil.copy('./config.yaml', './config/GF7_1120_model9.yaml')

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')

    writer=SummaryWriter(os.path.join(config['save_path'], 'logs'))

    # ------------------------------------------数据设置------------------------------------------
    train_set = Dataset(config['data_path'], config['input_size'], 'train')

    train_data_loader = DataLoader(
        dataset=train_set, num_workers=config['num_workers'], batch_size=config['batch_size'], 
        shuffle=True, pin_memory=True, drop_last=True)
    
    # dataset_val = Dataset(config['data_path'], config['input_size'], 'val')

    # data_val_loader = DataLoader(dataset=dataset_val, batch_size=1, num_workers=0, shuffle=False, pin_memory=True)

    # ------------------------------------------模型定义------------------------------------------
    generator = TDColor(encoder_name=config['encoder_name'],
                 decoder_name=config['decoder_name'],
                 num_input_channels=config['num_input_channels'],
                 input_size=tuple(config['input_size']),
                 nf=config['nf'],
                 last_norm=config['last_norm'],   # 这个参数代表什么
                 do_normalize=config['do_normalize'],
                 num_queries_l=config['num_queries_l'],
                 num_queries_ab=config['num_queries_ab'],
                 num_scales=config['num_scales'],
                 dec_layers=config['dec_layers'],
                 encoder_from_pretrain=config['encoder_from_pretrain'])
    generator = generator.to(device)
    # generator = DDP(generator, device_ids=[device], find_unused_parameters=True)

    discriminator = DynamicUNetDiscriminator(n_channels=config['n_channels'], nf=config['nf_d'])
    discriminator = discriminator.to(device)
    # discriminator = DDP(discriminator, device_ids=[device], find_unused_parameters=True)

    generator.train()
    discriminator.train()

    # ------------------------------------------优化设置------------------------------------------
    optimizer_g = torch.optim.AdamW(generator.parameters(), 
                                    config['lr_g'], 
                                    betas=tuple(config['betas_g']), 
                                    weight_decay=config['weight_decay_g'])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), 
                                   config['lr_d'], 
                                   betas=tuple(config['betas_d']), 
                                   weight_decay=config['weight_decay_d'])
    optimizers = [optimizer_g, optimizer_d]

    schedulers = []
    for optimizer in optimizers:
        schedulers.append(MultiStepRestartLR(optimizer, milestones=config['milestones'], gamma=config['gamma']))

    L1_loss = L1Loss(reduction=config['reduction']).to(device)
    Per_loss = PerceptualLoss(config['layer_weights'], 
                              config['vgg_type'], 
                              config['use_input_norm'], 
                              config['range_norm'], 
                              config['criterion']).to(device)
    # Per_loss = Vgg16().type(torch.cuda.FloatTensor).to(device)
    GAN_loss = GANLoss(config['gan_type'], config['real_label_val'], config['fake_label_val']).to(device)
    
    # ------------------------------------------正式训练------------------------------------------
    current_iters = config['start_epoch'] * len(train_data_loader)

    if config['resume_train']:
        assert os.path.exists(config['resume_g_path']), f"{config['resume_g_path']} files don't exist!"
        assert os.path.exists(config['resume_d_path']), f"{config['resume_d_path']} files don't exist!"
        state_dict_g = torch.load(config['resume_g_path'], map_location=torch.device('cpu'))
        state_dict_d = torch.load(config['resume_d_path'], map_location=torch.device('cpu'))
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_d['discriminator'])
        optimizer_g.load_state_dict(state_dict_g['optimizer_g'])
        optimizer_d.load_state_dict(state_dict_d['optimizer_d'])
        # 强制将优化器的参数tensor移到当前设备，确保参数组一致
        for param in optimizer_g.param_groups[0]['params']: 
            param.data = param.data.to(device)
        for param in optimizer_d.param_groups[0]['params']:
            param.data = param.data.to(device)
        print(f'Load net_g_{current_iters}.pth!\nLoad net_d_{current_iters}.pth!')
        with open(os.path.join(config['save_path'], 'logs', "epoch_loss" + time_str + ".txt"), 'a') as f:
            f.write(f'Load net_g_{current_iters}.pth!\nLoad net_d_{current_iters}.pth!\n')

    # epochs = math.ceil((config['iters'] - current_iters) / (len(train_data_loader)))
    epochs = config['n_epochs']

    print(f'Total epochs: {epochs}\n')
    with open(os.path.join(config['save_path'], 'logs', "epoch_loss" + time_str + ".txt"), 'a') as f:
        f.write(f'Total epochs: {epochs}\n')

    prev_division_Loss = 0
    Loss = 0

    for epoch in range(config['start_epoch'], epochs):
        pbda = tqdm(enumerate(train_data_loader), total=len(train_data_loader), bar_format="{l_bar}{bar:10}{r_bar}")
        for i, batch in pbda:
            current_iters += 1
            # if current_iters > config['iters']:
            #     break

            for scheduler in schedulers:
                scheduler.step()
            
            # 输入数据
            input_l = batch['input_l'].to(device)
            input_ab = batch['input_ab'].to(device)
            input = tensor_lab2rgb(torch.cat([input_l, input_ab], dim=1))

            # 标签数据
            label_l = batch['label_l'].to(device)
            label_ab = batch['label_ab'].to(device)
            label = tensor_lab2rgb(torch.cat([label_l, label_ab], dim=1))
            if config['color_enchance']:
                for i in range(label.shape[1]):
                    label[i] = color_enhacne_blend(label[i], factor=config['color_enhance_factor'])

            for p in discriminator.parameters():
                p.requires_grad = False
            optimizer_g.zero_grad()

            # 生成影像
            out_l, out_ab = generator(input)
            out = tensor_lab2rgb(torch.cat([out_l, out_ab], dim=1))

            # 生成器损失函数
            fake_g_pred = discriminator(out)
            l_pix_l = config['w_l'] * L1_loss(out_l, label_l)
            l_pix_ab = config['w_ab'] * L1_loss(out_ab, label_ab)
            l_pix = l_pix_l + l_pix_ab 
            
            l_per = Per_loss(out, label) * config['w_per']

            # tv损失
            # diff_i = torch.sum(torch.abs(out[:, :, :, 1:] - out[:, :, :, :-1]))
            # diff_j = torch.sum(torch.abs(out[:, :, 1:, :] - out[:, :, :-1, :]))
            # l_tv = config['w_tv'] * (diff_i + diff_j) / (256 * 256)

            # ssim损失
            # X = label
            # Y = out

            # l_ssim = 1 - ssim(X, Y, data_range=1, size_average=True)
            # l_ssim = config['w_ss'] * l_ssim

            l_gan = config['w_gan'] * GAN_loss(fake_g_pred, target_is_real=True, is_disc=False)

            l_g_total = l_pix + l_per + l_gan
            # l_g_total = l_pix + l_gan
            
            l_g_total.backward()
            optimizer_g.step()

            # 判别器损失函数
            for p in discriminator.parameters():
                p.requires_grad = True
            optimizer_d.zero_grad()
            
            real_d_pred = discriminator(label)
            fake_d_pred = discriminator(out.detach())
            l_d = (GAN_loss(real_d_pred, target_is_real=True, is_disc=True) + 
                  GAN_loss(fake_d_pred, target_is_real=False, is_disc=True))
            real_score = real_d_pred.detach().mean()
            fake_score = fake_d_pred.detach().mean()

            l_d.backward()
            optimizer_d.step()

            # 打印训练进度
            if current_iters % config['print_freq'] == 0:

                msg = (f"iters: {current_iters}, "
                       f"l_g: {l_g_total.data.cpu().numpy(): .5f}, "
                       f"l_pix: {l_pix.data.cpu().numpy(): .5f}, "
                       f"l_pix_l: {l_pix_l.data.cpu().numpy(): .5f}, "
                       f"l_pix_ab: {l_pix_ab.data.cpu().numpy(): .5f}, "
                       f"l_per: {l_per.data.cpu().numpy(): .5f}, "
                       f"l_gan: {l_gan.data.cpu().numpy(): .5f}, "
                    #    f"l_tv: {l_tv.data.cpu().numpy(): .5f},"
                    #    f"l_ssim: {l_ssim.data.cpu().numpy(): .5f},"
                       f"l_d: {l_d.data.cpu().numpy(): .5f}, "
                       f"real_score: {real_score.data.cpu().numpy(): .5f}, "
                       f"fake_score: {fake_score.data.cpu().numpy(): .5f}, "
                       f"lr: {optimizer_g.param_groups[0]['lr']: .7f}\n")
                print(msg)

                with open(os.path.join(config['save_path'], 'logs', "epoch_loss" + time_str + ".txt"), 'a') as f:
                    f.write(msg)

                writer.add_scalar("loss/l_g", l_g_total.data.cpu().numpy(), current_iters)
                writer.add_scalar("loss/l_pix", l_pix.data.cpu().numpy(), current_iters)
                writer.add_scalar("loss/l_pix_l", l_pix_l.data.cpu().numpy(), current_iters)
                writer.add_scalar("loss/l_pix_ab", l_pix_ab.data.cpu().numpy(), current_iters)
                writer.add_scalar("loss/l_per", l_per.data.cpu().numpy(), current_iters)
                # writer.add_scalar("loss/l_tv", l_tv.data.cpu().numpy(), current_iters)
                # writer.add_scalar("loss/l_ssim", l_ssim.data.cpu().numpy(), current_iters)
                writer.add_scalar("loss/l_gan", l_gan.data.cpu().numpy(), current_iters)
                writer.add_scalar("loss/l_d",l_d.data.cpu().numpy(), current_iters)
                # writer.add_scalar("score/real_score", real_score.data.cpu().numpy(), current_iters)
                # writer.add_scalar("score/fake_score", fake_score.data.cpu().numpy(), current_iters)
                writer.add_scalars('score', {'real_score': real_score.data.cpu().numpy(), 
                                             'fake_score': fake_score.data.cpu().numpy()}, current_iters)
            
        prev_division_Loss = Loss
        Loss = l_g_total

        if Loss < prev_division_Loss:
            save_g = f'net_g_best.pth'
            save_d = f'net_d_best.pth'
            if torch.cuda.is_available():
                torch.save(generator.state_dict(), os.path.join(config['save_path'], save_g))
                torch.save(discriminator.state_dict(), os.path.join(config['save_path'], save_d))
            else:
                torch.save(generator.cpu().state_dict(), os.path.join(config['save_path'], save_g))
                torch.save(optimizer_g.cpu().state_dict(), os.path.join(config['save_path'], save_g))
                torch.save(discriminator.cpu().state_dict(), os.path.join(config['save_path'], save_d))
                torch.save(optimizer_d.cpu().state_dict(), os.path.join(config['save_path'], save_d))

            with open(os.path.join(config['save_path'], 'best_epoch_step.txt'), 'a') as f:
                f.write(f'best_{epoch+1}_{current_iters}\n')

        # 保存模型
        if (epoch + 1) % config['save_every_epoch'] == 0 or epoch == (epochs - 1): #save_every_epoch
            if epoch == (epochs - 1):
                save_g = f'net_g_latest.pth'
                save_d = f'net_d_latest.pth'
                if torch.cuda.is_available():
                    torch.save(generator.state_dict(), os.path.join(config['save_path'], save_g))
                    torch.save(discriminator.state_dict(), os.path.join(config['save_path'], save_d))
                else:
                    torch.save(generator.cpu().state_dict(), os.path.join(config['save_path'], save_g))
                    torch.save(optimizer_g.cpu().state_dict(), os.path.join(config['save_path'], save_g))
                    torch.save(discriminator.cpu().state_dict(), os.path.join(config['save_path'], save_d))
                    torch.save(optimizer_d.cpu().state_dict(), os.path.join(config['save_path'], save_d))
            else:
                save_g = f'net_g_{epoch + 1}.pth'
                save_d = f'net_d_{epoch + 1}.pth'
                if torch.cuda.is_available():
                    torch.save({'generator': generator.state_dict(),
                                'optimizer_g': optimizer_g.state_dict()
                                }, os.path.join(config['save_path'], save_g))
                    torch.save({'discriminator': discriminator.state_dict(),
                                'optimizer_d': optimizer_d.state_dict(),
                                }, os.path.join(config['save_path'], save_d))
                else:
                    torch.save(generator.cpu().state_dict(), os.path.join(config['save_path'], save_g))
                    torch.save(optimizer_g.cpu().state_dict(), os.path.join(config['save_path'], save_g))
                    torch.save(discriminator.cpu().state_dict(), os.path.join(config['save_path'], save_d))
                    torch.save(optimizer_d.cpu().state_dict(), os.path.join(config['save_path'], save_d))


        print(f'Epoch: {epoch + 1} finished!')
        with open(os.path.join(config['save_path'], 'logs', "epoch_loss" + time_str + ".txt"), 'a') as f:
            f.write(f'Epoch: {epoch + 1} finished!\n')
    
    # end_time = time()
    # total_time = datetime.timedelta(end_time - start_time)
    # with open(os.path.join(config['save_path'], 'logs', "epoch_loss" + time_str + ".txt"), 'a') as f:
    #     f.write(f"Total_time: {total_time}.\n")

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"YAML解析错误: {e}")
            return None

if __name__ == "__main__":
    # ------------------------------------------参数设置------------------------------------------
    train()
