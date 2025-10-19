#!/usr/bin/env python3
"""
简单的图片对比测试脚本
直接指定两张图片进行对比
"""

import sys
import os
sys.path.append('/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0')

from image_similarity_checker import ImageSimilarityChecker

def test_specific_images():
    """测试指定的图片对"""
    print("🧪 简单图片对比测试")
    print("=" * 50)
    
    # 使用你的10k模型路径
    content_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/10kcontent_siamese_model_full.pth"
    style_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/10kstyle_siamese_model_full.pth"
    
    # 如果10k模型不存在，尝试使用调试模型
    if not os.path.exists(content_model_path):
        content_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/debug_content_siamese_model.pth"
        style_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/debug_style_siamese_model.pth"
        print("⚠️  使用调试模型进行测试")
    
    # 检查模型文件
    if not os.path.exists(content_model_path) or not os.path.exists(style_model_path):
        print("❌ 模型文件不存在，请先训练模型")
        print(f"需要的文件:")
        print(f"  - {content_model_path}")
        print(f"  - {style_model_path}")
        return
    
    try:
        # 初始化检查器
        print("📦 初始化相似度检查器...")
        checker = ImageSimilarityChecker(
            content_model_path=content_model_path,
            style_model_path=style_model_path
        )
        
        # 定义测试图片路径 - 你可以修改这些路径
        test_cases = [
            {
                "name": "相同内容不同风格",
                "img1": "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/data/train/CELingDHJW_Da/CELingDHJW_Da+八.png",
                "img2": "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/data/train/FZCaiYTJ/FZCaiYTJ+八.png",
                "expected": "内容相似，风格不同"
            },
            {
                "name": "不同内容相同风格", 
                "img1": "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/data/train/CELingDHJW_Da/CELingDHJW_Da+八.png",
                "img2": "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/data/train/CELingDHJW_Da/CELingDHJW_Da+不.png",
                "expected": "内容不同，风格相似"
            },
            {
                "name": "完全不同",
                "img1": "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/data/train/CELingDHJW_Da/CELingDHJW_Da+八.png",
                "img2": "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/data/train/FZCaiYTJ/FZCaiYTJ+不.png",
                "expected": "内容和风格都不同"
            }
        ]
        
        # 运行测试
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"测试 {i}: {test['name']}")
            print(f"{'='*60}")
            print(f"图片1: {os.path.basename(test['img1'])}")
            print(f"图片2: {os.path.basename(test['img2'])}")
            print(f"期望: {test['expected']}")
            
            # 检查文件是否存在
            if not os.path.exists(test['img1']):
                print(f"❌ 图片1不存在: {test['img1']}")
                continue
            if not os.path.exists(test['img2']):
                print(f"❌ 图片2不存在: {test['img2']}")
                continue
            
            # 执行比较
            result = checker.comprehensive_check(test['img1'], test['img2'])
            
            # 打印结果
            print(f"\n📊 比较结果:")
            print("-" * 30)
            print(f"📝 内容相似度: {result['content']['score']:.4f} {'✅' if result['content']['similar'] else '❌'}")
            print(f"🎨 风格相似度: {result['style']['score']:.4f} {'✅' if result['style']['similar'] else '❌'}")
            print(f"🎯 综合相似度: {result['overall']['score']:.4f}")
            
            # 分析结果
            analyze_result(result, test['name'])
        
        print(f"\n{'='*60}")
        print("✅ 所有测试完成！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def analyze_result(result, test_name):
    """分析测试结果"""
    content_score = result['content']['score']
    style_score = result['style']['score']
    
    print(f"\n💡 结果分析:")
    
    if "相同内容不同风格" in test_name:
        if content_score > 0.6 and style_score < 0.6:
            print("✅ 符合期望：成功识别相同内容和不同风格")
        elif content_score > 0.6:
            print("⚠️  部分符合：识别了相同内容，但风格区分不够明显")
        else:
            print("❌ 不符合期望：未能识别相同内容")
    
    elif "不同内容相同风格" in test_name:
        if content_score < 0.6 and style_score > 0.6:
            print("✅ 符合期望：成功识别不同内容和相同风格")
        elif style_score > 0.6:
            print("⚠️  部分符合：识别了相同风格，但内容区分不够明显")
        else:
            print("❌ 不符合期望：未能识别相同风格")
    
    elif "完全不同" in test_name:
        if content_score < 0.6 and style_score < 0.6:
            print("✅ 符合期望：成功识别内容和风格都不同")
        else:
            print("⚠️  部分符合：部分维度识别出差异")

def custom_test():
    """自定义测试 - 你可以在这里输入自己的图片路径"""
    print("\n🎯 自定义测试")
    print("=" * 50)
    
    # 在这里输入你想测试的图片路径
    img1_path = input("请输入第一张图片的完整路径: ").strip()
    img2_path = input("请输入第二张图片的完整路径: ").strip()
    
    if not img1_path or not img2_path:
        print("❌ 路径不能为空")
        return
    
    if not os.path.exists(img1_path):
        print(f"❌ 图片1不存在: {img1_path}")
        return
    
    if not os.path.exists(img2_path):
        print(f"❌ 图片2不存在: {img2_path}")
        return
    
    try:
        # 使用模型路径
        content_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/10kcontent_siamese_model_full.pth"
        style_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/10kstyle_siamese_model_full.pth"
        
        if not os.path.exists(content_model_path):
            content_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/debug_content_siamese_model.pth"
            style_model_path = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/debug_style_siamese_model.pth"
        
        checker = ImageSimilarityChecker(
            content_model_path=content_model_path,
            style_model_path=style_model_path
        )
        
        print(f"\n🔍 比较图片:")
        print(f"图片1: {img1_path}")
        print(f"图片2: {img2_path}")
        
        result = checker.comprehensive_check(img1_path, img2_path)
        
        print(f"\n📊 比较结果:")
        print("-" * 30)
        print(f"📝 内容相似度: {result['content']['score']:.4f} {'✅ 相似' if result['content']['similar'] else '❌ 不相似'}")
        print(f"🎨 风格相似度: {result['style']['score']:.4f} {'✅ 相似' if result['style']['similar'] else '❌ 不相似'}")
        print(f"🎯 综合判断: {'✅ 相似' if result['overall']['similar'] else '❌ 不相似'}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    # 运行预定义测试
    test_specific_images()
    
    # 询问是否进行自定义测试
    print(f"\n" + "="*60)
    user_input = input("是否进行自定义测试? (y/n): ").strip().lower()
    if user_input in ['y', 'yes', '是']:
        custom_test()