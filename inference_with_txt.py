from typing import List, Tuple, Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser, Namespace

import os
import struct
import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.ddim_sampler import DDIMSampler
from model.diffeic import DiffEIC
from utils.image import pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from dataset.text_generate import TextGenerator


@torch.no_grad()
def process(
    model: DiffEIC,
    imgs: List[np.ndarray],
    txt: str,
    sampler: str,
    steps: int,
    stream_path: str
) -> Tuple[List[np.ndarray], float]:
    """
    Apply DiffEIC model on a list of images.
    """
    n_samples = len(imgs)
    if sampler == "ddpm":
        sampler_obj = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler_obj = DDIMSampler(model)

    control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    height, width = control.size(-2), control.size(-1)

    # 调用 apply_condition_compress 仅压缩控制张量
    bpp = model.apply_condition_compress(control, stream_path, height, width)

    # 将文本保存到单独的文件
    txt_path = stream_path + "_text.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)

    # 解压控制张量
    c_latent = model.apply_condition_decompress(stream_path)

    # 从文本文件加载文本
    with open(txt_path, "r", encoding="utf-8") as f:
        decoded_txt = f.read()

    # 确保解码文本一致
    assert decoded_txt == txt, "Decoded text does not match the original text."

    cond = {
        "c_latent": [c_latent],
        "c_crossattn": [model.get_learned_conditioning(decoded_txt * n_samples)],
    }

    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

    if isinstance(sampler_obj, SpacedSampler):
        samples = sampler_obj.sample(
            steps,
            shape,
            cond,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            cond_fn=None,
            x_T=x_T,
        )
    else:
        sampler_obj: DDIMSampler
        samples, _ = sampler_obj.sample(
            S=steps,
            batch_size=shape[0],
            shape=shape[1:],
            conditioning=cond,
            unconditional_conditioning=None,
            x_T=x_T,
            eta=0,
        )

    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)

    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    preds = [x_samples[i] for i in range(n_samples)]
    return preds, bpp


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--ckpt", default="path to checkpoint file", type=str, help="full checkpoint path")
    parser.add_argument("--config", default="configs/model/diffeic.yaml", type=str, help="model config path")

    parser.add_argument("--input", type=str, default="path to input images")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", default=50, type=int)

    parser.add_argument("--output", type=str, default="results/")

    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    if args.device == "cpu":
        disable_xformers()

    model: DiffEIC = instantiate_from_config(OmegaConf.load(args.config))
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)

    generator_txt = TextGenerator()
    bpps = []
    assert os.path.isdir(args.input)

    print(f"sampling {args.steps} steps using {args.sampler} sampler")
    for file_path in list_image_files(args.input, follow_links=True):
        img = Image.open(file_path).convert("RGB")
        x = pad(np.array(img), scale=64)
        transform = transforms.ToTensor()
        img_gt_tensor = transform(x)
        img_gt_tensor = img_gt_tensor.unsqueeze(0)

        # Generate text
        txt_i = generator_txt.generate_text(img_gt_tensor)

        save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, "data")
        save_path = os.path.join(parent_path, f"{stem}.png")
        stream_path = os.path.join(stream_parent_path, f"{stem}")

        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)

        preds, bpp = process(
            model, [x], txt_i, steps=args.steps, sampler=args.sampler, stream_path=stream_path
        )
        pred = preds[0]

        bpps.append(bpp)

        # Remove padding
        pred = pred[: img.height, : img.width, :]

        Image.fromarray(pred).save(save_path)
        print(f"save to {save_path}, bpp {bpp}")

    avg_bpp = sum(bpps) / len(bpps)
    print(f"avg bpp: {avg_bpp}")


if __name__ == "__main__":
    main()
