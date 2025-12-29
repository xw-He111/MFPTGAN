
import torch
import torch.nn.functional as F
from collections import Counter
from torch.optim.lr_scheduler import LRScheduler
import os
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
import math
from einops import rearrange
import cv2
import numpy as np
from skimage import color

    
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    return num_params / 1e6
    
def update_learning_rate(optimizers, schedulers):
    old_lr = optimizers[0].param_groups[0]['lr']
    for scheduler in schedulers:
        scheduler.step()
    
    lr = optimizers[0].param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))

def get_scheduler(optimizer, start_epoch=0, n_epoch=20, n_epoch_decay=80):
    def lambda_rule(epoch, start_epoch=start_epoch, n_epoch=n_epoch, n_epoch_decay=n_epoch_decay):
        lr_l = 1.0 - max(0, epoch + start_epoch - n_epoch) / float(n_epoch_decay + 1)
        return lr_l
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler

def feature_visualization(features, features_num=100, save_dir='features/'):
    # save_dir = 'features/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    exist_img_num = len(os.listdir(save_dir))
    
    blocks = torch.chunk(features, features.shape[1], dim=1)

    plt.figure()
    for i in range(features_num):
        torch.squeeze(blocks[i])
        feature = transforms.ToPILImage()(blocks[i].squeeze())
        ax = plt.subplot(int(math.sqrt(features_num)), features_num // int(math.sqrt(features_num)) + 1, i+1)
        # ax = plt.subplot(1, 1, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature)

    plt.savefig(save_dir + '{}_{}.png'.format(features_num, exist_img_num), dpi=300)

def feature_visualization(features, features_num=100, save_dir='features/'):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    exist_img_num = len(os.listdir(save_dir))
    
    blocks = torch.chunk(features, features.shape[1], dim=1)

    plt.figure()
    for i in range(features_num):
        torch.squeeze(blocks[i])
        feature = transforms.ToPILImage()(blocks[i].squeeze())
        ax = plt.subplot(int(math.sqrt(features_num)), features_num // int(math.sqrt(features_num)) + 1, i+1)
        # ax = plt.subplot(1, 1, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature)

    plt.savefig(save_dir + '{}_{}.png'.format(features_num, exist_img_num), dpi=300)


def feature_visualization_single(features, features_num=100, save_dir='./visualization/out_l/'):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # exist_img_num = len(os.listdir(save_dir))
    
    blocks = torch.chunk(features, features.shape[1], dim=1)

    plt.figure()
    for i in range(features_num):
        torch.squeeze(blocks[i])
        # feature = transforms.ToPILImage()(blocks[i].squeeze())

        feature = blocks[i].detach().cpu()
        feature = rearrange(feature, 'bs c h w -> bs h w c')
        feature = np.float32(feature[0, :, :, :])

        output_max = feature.max()
        output_min = feature.min()

        feature = (feature - output_min) / (output_max - output_min)

        feature = (feature * 255.0).round()

        feature = feature.astype(np.uint8)

        plt.imshow(feature)

        plt.axis("off")

        plt.savefig(save_dir + '{}.png'.format(i), dpi=100, bbox_inches="tight", pad_inches=0.0)

        plt.close()