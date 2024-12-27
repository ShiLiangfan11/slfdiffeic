import torch
from .ram.models.ram_lora import ram
from .ram import inference_ram as inference

class TextGenerator:
    def __init__(self, 
                 model_path='/home/dongnan/SLF/NVC/DiffEIC-main/weight/models/ram_swin_large_14m.pth', 
                 condition_path='/home/dongnan/SLF/NVC/DiffEIC-main/weight/models/DAPE.pth', 
                 device='cuda', 
                 weight_dtype=torch.float16):
        """
        初始化 TextGenerator，加载 RAM 模型。

        参数：
        - model_path (str): RAM 模型的权重路径，默认路径为之前指定的路径。
        - condition_path (str): 条件路径，用于初始化 RAM 模型，默认路径为之前指定的路径。
        - device (str): 设备，默认为 'cuda'。
        - weight_dtype (torch.dtype): 权重数据类型，默认为 torch.float16。
        """
        self.device = device
        self.weight_dtype = weight_dtype

        # 加载 RAM 模型
        self.model = ram(pretrained=model_path,
                         pretrained_condition=condition_path,
                         image_size=384,
                         vit='swin_l')
        self.model.eval()
        self.model.to(self.device, dtype=self.weight_dtype)

    @torch.no_grad()
    def generate_text(self, input_tensor, user_prompt='', positive_prompt=''):
        """
        生成文本描述。

        参数：
        - input_tensor (torch.Tensor): 输入的图像张量 (B, C, H, W)。
        - user_prompt (str): 用户提供的额外提示。
        - positive_prompt (str): 正面提示，用于增强生成的文本。

        返回：
        - str: 生成的文本描述。
        """
        # 将张量移动到指定的设备和 dtype
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
