#!/bin/bash

# 设置环境变量或激活虚拟环境（如果有的话）
# source /path/to/your/virtualenv/bin/activate

# 运行 Python 脚本
python3 inference_partition.py \
   --ckpt_sd weight/v2-1_512-ema-pruned.ckpt \
   --ckpt_lc ./weight/1_2_1/lc.ckpt \
   --config configs/model/diffeic.yaml \
   --input /home/dongnan/SLF/data/image/kodak_512 \
   --output /home/dongnan/SLF/data/image/kodak_512_part \
   --steps 50 \
   --device cuda
