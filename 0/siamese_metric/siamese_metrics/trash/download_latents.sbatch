#!/bin/bash
#SBATCH --job-name=download_latents
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1   # 即使不用 GPU，也需要指定，否则无法提交
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=24:00:00

module load git-lfs

cd /scratch/gz2199/

if [ ! -d "Font-Latent-Full-PT" ]; then
    git clone https://huggingface.co/datasets/YuanhengLi/Font-Latent-Full-PT
fi

cd Font-Latent-Full-PT

git lfs install
git lfs pull
