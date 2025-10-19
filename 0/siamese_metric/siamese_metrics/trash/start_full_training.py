#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动全数据集训练的便捷脚本
"""

if __name__ == "__main__":
    from full_training import full_scale_training
    
    print("🚀 启动全数据集训练...")
    print("⚡ 这将使用完整的数据集和优化的超参数")
    print("⏰ 预计训练时间: 2-4小时")
    print("💾 模型将自动保存为 *_full.pth 文件")
    print("=" * 60)
    
    # 启动全规模训练
    try:
        content_model, style_model, accessor, decoder = full_scale_training(encoder_type="enhanced")
        print("\n🎉 训练成功完成！")
        print("📁 检查保存的模型文件:")
        print("   - content_siamese_model_full.pth")
        print("   - style_siamese_model_full.pth")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        print("💡 请检查数据路径和GPU内存")