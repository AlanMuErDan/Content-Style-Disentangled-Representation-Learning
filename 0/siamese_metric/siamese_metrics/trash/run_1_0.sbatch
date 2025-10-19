#!/bin/bash
#SBATCH --job-name=font_training
#SBATCH --output=logs/font_training_%j.out
#SBATCH --error=logs/font_training_%j.err
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:1   # 即使不用 GPU，也需要指定，否则无法提交
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00

# 创建日志目录
mkdir -p logs

echo "🚀 开始字体训练任务..."
echo "📅 开始时间: $(date)"
echo "🖥️ 节点信息: $(hostname)"
echo "💾 内存信息:"
free -h
echo "🎯 GPU信息:"
nvidia-smi

# 激活conda环境
echo "🔧 激活conda环境..."
conda activate /scratch/gz2199/calli_env

# 切换到工作目录
echo "📁 切换到工作目录..."
cd /scratch/gz2199/Content-Style-Disentangled-Representation-Learning

# 显示数据文件信息
echo "📊 数据文件信息:"
ls -lh Font-Latent-Full-PT/font_latents_v2.pt

# 运行训练脚本
echo "🎯 开始运行字体训练脚本..."
python /scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/1.py

# 显示完成信息
echo "✅ 训练完成!"
echo "📅 结束时间: $(date)"
echo "💾 最终内存使用:"
free -h