import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# 需要安装以下库
# pip install lpips piq DISTS_pytorch

import lpips  # 用于计算 LPIPS
import piq    # 用于计算 PSNR、SSIM、MS-SSIM
from DISTS_pytorch import DISTS  # 用于计算 DISTS

def compute_metrics(input_dir, gt_dir):
    input_images = sorted(os.listdir(input_dir))
    gt_images = sorted(os.listdir(gt_dir))

    assert input_images == gt_images, "输入和 GT 文件夹中的图像名称不匹配"

    # 初始化指标
    total_psnr = 0
    total_ssim = 0
    total_ms_ssim = 0
    total_lpips = 0
    total_dists = 0  # 用于累积 DISTS

    # 初始化 LPIPS 和 DISTS 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    loss_fn_dists = DISTS().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for img_name in tqdm(input_images, desc="计算指标"):
        input_path = os.path.join(input_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)

        # 加载图像并转换为张量
        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        input_tensor = transform(input_img).unsqueeze(0).to(device)
        gt_tensor = transform(gt_img).unsqueeze(0).to(device)

        # 计算 PSNR
        psnr = piq.psnr(input_tensor, gt_tensor, data_range=1.0).item()
        total_psnr += psnr

        # 计算 SSIM
        ssim = piq.ssim(input_tensor, gt_tensor, data_range=1.0).item()
        total_ssim += ssim

        # 计算 MS-SSIM
        ms_ssim = piq.multi_scale_ssim(input_tensor, gt_tensor, data_range=1.0).item()
        total_ms_ssim += ms_ssim

        # 计算 LPIPS
        with torch.no_grad():
            lpips_value = loss_fn_lpips(input_tensor, gt_tensor).item()
        total_lpips += lpips_value

        # 计算 DISTS
        with torch.no_grad():
            dists_value = loss_fn_dists(input_tensor, gt_tensor).item()
        total_dists += dists_value

    num_images = len(input_images)
    print(f'平均 PSNR: {total_psnr / num_images:.4f}')
    print(f'平均 SSIM: {total_ssim / num_images:.4f}')
    print(f'平均 MS-SSIM: {total_ms_ssim / num_images:.4f}')
    print(f'平均 LPIPS: {total_lpips / num_images:.4f}')
    print(f'平均 DISTS: {total_dists / num_images:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算两个文件夹中对应图像的图像恢复评价指标')
    parser.add_argument('--input_dir', type=str,default='results' , help='')
    parser.add_argument('--gt_dir', type=str,default='/home/dongnan/SLF/data/image/kodak_512', help='')
    args = parser.parse_args()

    compute_metrics(args.input_dir, args.gt_dir)
