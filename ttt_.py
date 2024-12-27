import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from ldm.models.autoencoder import AutoencoderKL
import os

# 显示输入图像和重构图像
def show_images(inputs, reconstructions):
    # 将张量转换为 NumPy 数组，并通过Matplotlib显示图像
    def tensor_to_image(tensor):
        return tensor.cpu().numpy().transpose(0, 2, 3, 1)  # 转换为 (N, H, W, C)

    inputs_img = tensor_to_image(inputs)
    reconstructions_img = tensor_to_image(reconstructions)
    
    # 绘制输入和重构图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(inputs_img[0])
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(reconstructions_img[0])
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")

    plt.show()

# 这里是你给出的配置，可以根据实际需求调整
ddconfig = {
    "z_channels": 4,
    "double_z": True,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
    "resolution": 256
}

lossconfig = {
    "target": "torch.nn.Identity"  # 使用 Identity loss 作为占位符
}
scale_factor = 0.18215
# 初始化 AutoencoderKL 模型
model = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=8192, monitor="val/rec_loss")

# 加载预训练权重
checkpoint_path = "/home/dongnan/SLF/NVC/DiffEIC-main/weight/v2-1_512-ema-pruned.ckpt"  # 请替换为实际的 ckpt 文件路径
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# 从 checkpoint 中提取 state_dict 并加载到模型
state_dict = checkpoint["state_dict"]
first_stage_state_dict = {k: v for k, v in state_dict.items() if 'first_stage_model' in k}
# 打印所有模块名称

model.load_state_dict(first_stage_state_dict, strict=False)
print(f"Loaded pre-trained model from {checkpoint_path}")

# 假设你有一张图片 (替换成你的图片路径)
image_path = 'test/input/3.png'  # 请替换为实际图片路径
img = Image.open(image_path).convert("RGB")

# 对图像进行预处理，调整为模型输入的大小（256x256）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为 tensor
])

# 加载并归一化图像
img_tensor = transform(img).unsqueeze(0)  # 添加批次维度，变成 [1, 3, 256, 256]


# 将图像放到设备上 (如果有 GPU，放到 GPU 上)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
img_tensor = img_tensor.to(device)
# 测试：使用 `encode` 和 `decode` 进行推理
model.eval()  # 设置为评估模式
with torch.no_grad():  # 禁用梯度计算
    # 进行编码
    encoded = model.encode(img_tensor * 2 - 1).mode() * scale_factor
    z = 1. / scale_factor * encoded
    # 进行解码
    reconstructions = model.decode(z)  # 使用 decode 进行解码
    reconstructions = ((reconstructions+ 1) / 2).clamp(0, 1)
    # 打印输入、重构图像以及 posterior 相关的中间变量形状
    print(f"Input shape: {img_tensor.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Reconstruction shape: {reconstructions.shape}")

    # 显示输入图像和重构图像
    show_images(img_tensor, reconstructions)
