#!/bin/bash
#SBATCH --job-name=train_vae
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# 激活新环境（路径为绝对路径）
source /scratch/gz2199/calli_env/bin/activate

# 进入工作目录
cd /scratch/gz2199/Content-Style-Disentangled-Representation-Learning

# 启动训练脚本
python main_train2.py --config configs/config_gan2.yaml
