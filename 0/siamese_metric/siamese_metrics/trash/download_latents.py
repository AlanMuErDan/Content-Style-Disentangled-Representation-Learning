#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download latents script - converted from SBATCH
Interactive version for real-time monitoring
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并实时显示输出"""
    print(f"\n{'='*50}")
    print(f"📋 {description}")
    print(f"🔧 Command: {cmd}")
    print(f"{'='*50}")
    
    try:
        # 使用subprocess.run实时显示输出
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时打印输出
        for line in result.stdout.splitlines():
            print(f"📤 {line}")
            
        print(f"✅ {description} - 完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误: {description} 失败")
        print(f"💥 错误信息: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"📤 输出: {e.stdout}")
        return False

def main():
    print("🚀 开始下载字体潜在编码数据...")
    
    # 设置工作目录
    work_dir = Path("/scratch/gz2199/")
    os.chdir(work_dir)
    print(f"📁 工作目录: {work_dir}")
    
    # 检查目录是否存在
    repo_dir = work_dir / "Font-Latent-Full-PT"
    
    if not repo_dir.exists():
        print("📥 仓库不存在，开始克隆...")
        success = run_command(
            "git clone https://huggingface.co/datasets/YuanhengLi/Font-Latent-Full-PT",
            "克隆 HuggingFace 数据集仓库"
        )
        if not success:
            print("❌ 克隆失败，退出")
            sys.exit(1)
    else:
        print(f"✅ 仓库已存在: {repo_dir}")
    
    # 进入仓库目录
    os.chdir(repo_dir)
    print(f"📂 切换到: {repo_dir}")
    
    # 安装和配置 git-lfs
    print("\n🔧 配置 Git LFS...")
    run_command("git lfs install", "安装 Git LFS")
    
    # 拉取 LFS 文件
    print("\n📦 下载大文件...")
    success = run_command("git lfs pull", "拉取 LFS 大文件")
    
    if success:
        print("\n🎉 所有操作完成!")
        
        # 显示下载的文件
        print("\n📋 检查下载的文件:")
        try:
            for file in repo_dir.iterdir():
                if file.is_file():
                    size = file.stat().st_size / (1024*1024)  # MB
                    print(f"📄 {file.name}: {size:.2f} MB")
        except Exception as e:
            print(f"⚠️ 无法列出文件: {e}")
            
        # 检查是否有 .pt 文件
        pt_files = list(repo_dir.glob("*.pt"))
        if pt_files:
            print(f"\n✅ 找到 {len(pt_files)} 个 .pt 文件:")
            for pt_file in pt_files:
                size = pt_file.stat().st_size / (1024*1024)  # MB
                print(f"🔥 {pt_file.name}: {size:.2f} MB")
        else:
            print("\n⚠️ 没有找到 .pt 文件")
            
    else:
        print("\n❌ 下载过程中出现错误")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)