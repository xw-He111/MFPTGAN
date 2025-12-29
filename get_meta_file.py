from tqdm import tqdm
import argparse
import os
import random

def find_and_save_images(data_path, output_name, data_path_gray, output_name_gray, data_path_rgb_val, output_name_rgb_val, data_path_gray_val, output_name_gray_val):
    all_img_path = []
    all_img_path_gray = []
    all_img_path_rgb_val = []
    all_img_path_gray_val = []

    for root, dirs, files in os.walk(data_path):
        for ext in ['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG']:
            all_img_path += [os.path.join(root, file) for file in files if file.lower().endswith(f'.{ext}')]

    with open(output_name, 'w') as f:
        for path in tqdm(all_img_path):
            f.write(path + '\n')

    for root, dirs, files in os.walk(data_path_gray):
        for ext in ['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG']:
            all_img_path_gray += [os.path.join(root, file) for file in files if file.lower().endswith(f'.{ext}')]

    with open(output_name_gray, 'w') as f:
        for path in tqdm(all_img_path_gray):
            f.write(path + '\n')

    for root, dirs, files in os.walk(data_path_rgb_val):
        for ext in ['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG']:
            all_img_path_rgb_val += [os.path.join(root, file) for file in files if file.lower().endswith(f'.{ext}')]

    with open(output_name_rgb_val, 'w') as f:
        for path in tqdm(all_img_path_rgb_val):
            f.write(path + '\n')

    for root, dirs, files in os.walk(data_path_gray_val):
        for ext in ['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG']:
            all_img_path_gray_val += [os.path.join(root, file) for file in files if file.lower().endswith(f'.{ext}')]

    with open(output_name_gray_val, 'w') as f:
        for path in tqdm(all_img_path_gray_val):
            f.write(path + '\n')


if __name__ == '__main__':

    # val = random.sample(range(1, 2416), 30)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', type=str, required=False, default='train_label.txt', help='Path output file.')
    parser.add_argument('--data-path', type=str, required=False, default='G:/datasets/GF7/train_label', help='Path to dataset')
    parser.add_argument('--output-name-rgb-val', type=str, required=False, default='test_label.txt', help='Path output file.')
    parser.add_argument('--data-path-rgb-val', type=str, required=False, default='G:/datasets/GF7/test_label', help='Path to dataset')
    # parser.add_argument('--val-path', type=str, required=False, default='rgb_val.txt')
    # parser.add_argument('--val_num', default=val)
    parser.add_argument('--output-name-gray', type=str, required=False, default='train_input.txt', help='Path output file.')
    parser.add_argument('--data-path-gray', type=str, required=False, default='G:/datasets/GF7/train_input', help='Path to dataset')
    parser.add_argument('--output-name-gray-val', type=str, required=False, default='test_input.txt', help='Path output file.')
    parser.add_argument('--data-path-gray-val', type=str, required=False, default='G:/datasets/GF7/test_input', help='Path to dataset')

    args = parser.parse_args()

    print(f'Generating {args.output_name} from {args.data_path} ...')

    find_and_save_images(args.data_path, args.output_name, args.data_path_gray, args.output_name_gray, args.data_path_rgb_val, args.output_name_rgb_val,
                         args.data_path_gray_val, args.output_name_gray_val)

    print('Done.')
