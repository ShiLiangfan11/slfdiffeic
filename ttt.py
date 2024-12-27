from model.slf_cm import DMCI
import torch

if __name__ == "__main__":
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bin_path = './output/1.bin'  # 输出路径，若为 None，跳过相关操作
    bin_path = None
    # 创建一个随机输入张量，大小为 (1, 4, 32, 32)
    x = torch.randn(1, 4, 32, 32).to(device)  # 将输入转到 GPU 或 CPU
    
    # 实例化模型并将模型转到 GPU 或 CPU
    model = DMCI().to(device)  # 将模型转到 GPU 或 CPU
    
    # 根据 bin_path 是否有效决定是否执行 update 和 encode 操作
    if bin_path:  # 如果 bin_path 非空
        model.update(force=True)  # 强制更新模型
        z = model.encode(x, 1, 0, bin_path)  # 使用 bin_path 进行编码
    else:
        z = model(x)  # 不使用 bin_path，直接编码

    # 打印结果（可以选择性打印或进行其他操作）
    if isinstance(z, dict) and "x_hat" in z:
        print("z.shape:", z["x_hat"].shape)  # 打印 'x_hat' 的形状
    else:
        print("Encoding result is not in expected format.")
