import cv2
import random
import time
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import rgb2lab
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class LabDataset(data.Dataset):
    """
    Dataset used for Lab colorizaion
    """

    def __init__(self, opt):
        super(LabDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        label_file = self.opt['dataroot_label']
        assert label_file is not None
        if not isinstance(label_file, list):
            label_file = [label_file]
        self.label_paths = []
        for label_txt in label_file:
            with open(label_txt, 'r') as fin:
                self.label_paths.extend([line.strip() for line in fin])

        pan_file = self.opt['dataroot_pan']
        assert pan_file is not None
        if not isinstance(pan_file, list):
            pan_file = [pan_file]
        self.pan_paths = []
        for pan_txt in pan_file:
            with open(pan_txt, 'r') as fin2:
                self.pan_paths.extend([line.strip() for line in fin2])


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        label_path = self.label_paths[index]   # 彩色影像
        pan_path = self.pan_paths[index]   # 全色影像
        gt_size = self.opt['gt_size']
        # avoid errors caused by high latency in reading files  避免高延迟出现的错误
        retry = 3
        while retry > 0:
            try:
                label_bytes = self.file_client.get(label_path, 'label')
                pan_bytes = self.file_client.get(pan_path, 'pan')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                label_path = self.label_paths[index]
                pan_path = self.pan_paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_label = imfrombytes(label_bytes, float32=True)
        img_label = cv2.resize(img_label, (gt_size, gt_size))  # TODO: 直接resize是否是最佳方案？

        img_pan = imfrombytes(pan_bytes, float32=True)
        img_pan = cv2.resize(img_pan, (gt_size, gt_size))  # TODO: 直接resize是否是最佳方案？

        # ----------------------------- Get gray lq, to tentor ----------------------------- #
        # convert to gray
        img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)
        img_pan = cv2.cvtColor(img_pan, cv2.COLOR_BGR2RGB)

        img_l, img_ab = rgb2lab(img_label)

        img_l_pan, img_ab_pan = rgb2lab(img_pan)

        # numpy to tensor
        img_l, img_ab = img2tensor([img_l, img_ab], bgr2rgb=False, float32=True)

        img_l_pan, img_ab_pan = img2tensor([img_l_pan, img_ab_pan], bgr2rgb=False, float32=True)

        return_d = {
            'pan_l': img_l_pan,
            'pan_ab': img_ab_pan,
            'label_ab': img_ab,
            'label_l': img_l,
            'pan_path': pan_path,
            'label_path': label_path
        }
        return return_d

    def __len__(self):
        return len(self.pan_paths)
