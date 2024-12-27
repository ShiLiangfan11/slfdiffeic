from typing import Sequence, Dict, Union
import time
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image
import ast
from typing import Dict, Union
import time
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from .ram.models.ram_lora import ram
from .ram import inference_ram as inference
import matplotlib.pyplot as plt
import os

# 在程序启动时，设置多进程启动方法为 'spawn'
mp.set_start_method('spawn', force=True)
class LICDataset(data.Dataset):
    def __init__(self, file_list: str, out_size: int, crop_type: str, use_hflip: bool, use_rot: bool, 
                 model_path='/home/dongnan/SLF/NVC/DiffEIC-main/weight/models/ram_swin_large_14m.pth',
                 condition_path='/home/dongnan/SLF/NVC/DiffEIC-main/weight/models/DAPE.pth', device='cuda'):
        """
        数据集类，包含图片加载、预处理和文本生成。
        """
        super(LICDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)  # 确保 load_file_list 已经定义
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        self.use_rot = use_rot

        # 初始化 ImageToTextGenerator
        self.device = device
        self.weight_dtype = torch.float16

        # 加载 RAM 模型
        self.model = ram(pretrained=model_path,
                         pretrained_condition=condition_path,
                         image_size=384,
                         vit='swin_l')
        self.model.eval()
        self.model.to(self.device, dtype=self.weight_dtype)

    @torch.no_grad()
    def generate_text_from_tensor(self, input_tensor, user_prompt='', positive_prompt=''):
        """
        生成文本描述。
        """
        # 将 tensor 移动到指定的设备和 dtype
        lq = input_tensor.to(self.device).type(self.weight_dtype)

        # 如果需要，调整图片大小到 384x384
        b, c, h, w = lq.shape
        if h != 384 or w != 384:
            lq = torch.nn.functional.interpolate(lq, size=(384, 384), mode='bilinear', align_corners=False)

        # 使用 RAM 模型生成描述
        res = inference(lq, self.model)

        # 构造最终的文本提示
        generated_texts = []
        for i in range(len(res)):
            validation_prompt = f"{res[i]}, {positive_prompt},"
            if user_prompt != '':
                validation_prompt = f"{user_prompt}, {validation_prompt}"
            generated_texts.append(validation_prompt)

        txt_list = generated_texts[0]
        txt_list = txt_list.rstrip(', ')
        txt_cleaned = txt_list.strip("[]").replace("'", "")
        return txt_cleaned

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        """
        获取指定索引的数据项，处理图像并生成相应的文本描述。
        """
        gt_path = self.paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"

        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)

        img_gt = (pil_img_gt / 255.0).astype(np.float32)
        transform = transforms.ToTensor()
        img_gt_tensor = transform(pil_img_gt)

        img_gt_tensor = img_gt_tensor.unsqueeze(0)

        # 使用生成器生成文本
        txt_i = self.generate_text_from_tensor(img_gt_tensor)

        # 数据增强
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=self.use_rot, return_status=False)

        target = (img_gt * 2 - 1).astype(np.float32)
        source = img_gt.astype(np.float32)

        return dict(jpg=target, txt=txt_i, hint=source)

    def __len__(self) -> int:
        return len(self.paths)