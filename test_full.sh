#!/bin/bash

# 设置环境变量或激活虚拟环境（如果有的话）
# source /path/to/your/virtualenv/bin/activate

# 运行 Python 脚本
python3 inference_with_txt.py \
   --ckpt logs/1_2_3/lightning_logs/version_9/checkpoints/step=24999.ckpt \
   --config configs/model/diffeic.yaml \
   --input /home/dongnan/SLF/data/image/kodak_512\
   --output /home/dongnan/SLF/data/image/kodak_512_full \
   --steps 10 \
   --device cuda
