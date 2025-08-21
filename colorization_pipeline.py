import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from basicsr.archs.tdcolor_arch import TDColor
import torch.nn.functional as F
import time


class ImageColorizationPipeline(object):

    def __init__(self, model_path, input_size=256, model_size='large'):
        
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.model = TDColor(
            encoder_name=self.encoder_name,
            decoder_name='MultiScaleColorDecoder',
            input_size=[self.input_size, self.input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img):
        self.height, self.width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_gray_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab, output_l = tuple(i.cpu() for i in self.model(tensor_gray_rgb))  # (1, 2, self.height, self.width)

        # resize ab -> concat original l -> rgb
        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_l_resize = F.interpolate(output_l, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((output_l_resize, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)    

        return output_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./net_g_latest.pth')
    parser.add_argument('--input', type=str, default='dot/', help='input test image folder or video path')
    parser.add_argument('--output', type=str, default='results_dot', help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=256, help='input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='ddcolor model size')
    args = parser.parse_args()

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size, model_size=args.model_size)

    start_time = time.time()
    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))
        image_out = colorizer.process(img)
        cv2.imwrite(os.path.join(args.output, name), image_out)

    end_time = time.time()
    used_time = end_time - start_time
    fps = 2000 / used_time
    print(f"fps: {fps}")
    with open('./fps.txt', 'a') as f:
        f.write(f"fps: {fps}\n")

if __name__ == '__main__':
    main()
