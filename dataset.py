
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from skimage import color
from torchvision.transforms import functional as F
import cv2
import random
from abc import ABCMeta, abstractmethod
import time

class Dataset(Dataset):
    def __init__(self, data_path, in_size, phase):
        self.phase = phase
        if self.phase == 'train':
            self.dir_input = os.path.join(data_path, 'train_input.txt')  # get the image directory
            self.dir_label = os.path.join(data_path, 'train_label.txt')  # get the label directory
        if self.phase == 'val':
            self.dir_input = os.path.join(data_path, 'val_input.txt')  # get the image directory
            self.dir_label = os.path.join(data_path, 'val_label.txt')  # get the label directory
        if self.phase == 'test':
            self.dir_input = os.path.join(data_path, 'test_input.txt')  # get the image directory
            self.dir_label = os.path.join(data_path, 'test_label.txt')  # get the label directory

        self.input_paths = []
        self.label_paths = []

        with open(self.dir_input, 'r') as fin:
            self.input_paths.extend([line.strip() for line in fin])

        with open(self.dir_label, 'r') as fin2:
            self.label_paths.extend([line.strip() for line in fin2])

        self.in_size = in_size

        self.file_client = read_byte()

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        input_path = self.input_paths[index]
        label_path = self.label_paths[index]

        retry = 3
        while retry > 0:
            try:
                input_bytes = self.file_client.get(input_path)
                label_bytes = self.file_client.get(label_path)
            except Exception as e:
                # change anther file to read
                index = random.randint(0, self.__len__())
                input_path = self.input_paths[index]
                label_path = self.label_paths[index]
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        
        img_input = imfrombytes(input_bytes, float32=True)
        img_input = cv2.resize(img_input, self.in_size)

        img_label = imfrombytes(label_bytes, float32=True)
        img_label = cv2.resize(img_label, self.in_size)

        if self.phase == 'train':
            img_input, img_label = self.apply_data_augmentation(img_input, img_label)
            img_input = np.ascontiguousarray(img_input)
            img_label = np.ascontiguousarray(img_label)

        img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

        img_input_l, img_input_ab = rgb2lab(img_input)
        img_label_l, img_label_ab = rgb2lab(img_label)

        # numpy to tensor
        img_input_l, img_input_ab = img2tensor([img_input_l, img_input_ab], bgr2rgb=False, float32=True)
        img_label_l, img_label_ab = img2tensor([img_label_l, img_label_ab], bgr2rgb=False, float32=True)

        return_d = {
            'input_l': img_input_l,
            'input_ab': img_input_ab,
            'label_l': img_label_l,
            'label_ab': img_label_ab,
            'input_path': input_path,
            'label_path': label_path
        }
        return return_d
    
    def apply_data_augmentation(self, img_input, img_label):
        # 随机水平翻转
        if random.random() > 0.5:
            img_input = cv2.flip(img_input, 1)
            img_label = cv2.flip(img_label, 1)

        # 随机垂直翻转
        if random.random() > 0.5:
            img_input = cv2.flip(img_input, 0)
            img_label = cv2.flip(img_label, 0)

        # 随机旋转（90， 180， 270度)
        # rot_degree = random.choice([0, 1, 2, 3])
        # if rot_degree > 0:
        #     img_input = np.rot90(img_input, rot_degree)
        #     img_label = np.rot90(img_label, rot_degree)

        return img_input, img_label

class BaseStorageBackend(metaclass=ABCMeta):
    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass

class read_byte(BaseStorageBackend):
    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf
    
    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf
    
def imfrombytes(content, flag='color', float32=False):
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 
                    'grayscale': cv2.IMREAD_GRAYSCALE, 
                    'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255
    return img

def rgb2lab(img_rgb):
    img_lab = color.rgb2lab(img_rgb)
    img_l = img_lab[:, :, :1]
    img_ab = img_lab[:, :, 1:]

    return img_l, img_ab

def tensor_lab2rgb(labs, illuminant="D65", observer="2"):
    """
    Args:
        lab    : (B, C, H, W)
    Returns:
        tuple   : (C, H, W)
    """
    illuminants = \
        {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
               '10': (1.111420406956693, 1, 0.3519978321919493)},
         "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
                 '10': (0.9672062750333777, 1, 0.8142801513128616)},
         "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
                 '10': (0.9579665682254781, 1, 0.9092525159847462)},
         "D65": {'2': (0.95047, 1., 1.08883),  # This was: `lab_ref_white`
                 '10': (0.94809667673716, 1, 1.0730513595166162)},
         "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
                 '10': (0.9441713925645873, 1, 1.2064272211720228)},
         "E": {'2': (1.0, 1.0, 1.0),
               '10': (1.0, 1.0, 1.0)}}
    xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]])

    rgb_from_xyz = np.array([[3.240481340, -0.96925495, 0.055646640], [-1.53715152, 1.875990000, -0.20404134],
                             [-0.49853633, 0.041555930, 1.057311070]])
    labs = labs.to(torch.float32)
    B, C, H, W = labs.shape
    arrs = labs.permute((0, 2, 3, 1)).contiguous()  # (B, 3, H, W) -> (B, H, W, 3)
    L, a, b = arrs[:, :, :, 0:1], arrs[:, :, :, 1:2], arrs[:, :, :, 2:]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)
    invalid = z.data < 0
    z[invalid] = 0
    xyz = torch.cat([x, y, z], dim=3)
    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.) / 7.787
    xyz_ref_white = illuminants[illuminant][observer]
    for i in range(C):
        mask_xyz[:, :, :, i] = mask_xyz[:, :, :, i] * xyz_ref_white[i]

    rgb_trans = torch.mm(mask_xyz.view(-1, 3), torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(B, H, W, C)
    rgb_trans = rgb_trans.to(torch.float32)
    rgb = rgb_trans.permute((0, 3, 1, 2)).contiguous()
    mask = rgb.data > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92
    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb

def color_enhacne_blend(x, factor=1.2):
    x_g = Grayscale(3)(x)
    out = x_g * (1.0 - factor) + x * factor
    out[out < 0] = 0
    out[out > 1] = 1
    return out

class Grayscale(torch.nn.Module):
    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Grayscaled image.
        """
        return F.rgb_to_grayscale(img, num_output_channels=self.num_output_channels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"
    

def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img
    
    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)