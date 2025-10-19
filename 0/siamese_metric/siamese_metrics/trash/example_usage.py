#!/usr/bin/env python3
"""
图片相似度检查器使用示例
演示如何使用ImageSimilarityChecker类
"""

from image_similarity_checker import ImageSimilarityChecker
from PIL import Image
import numpy as np

def example_usage():
    """使用示例"""
    print("🔍 图片相似度检查器使用示例")
    print("=" * 50)
    
    try:
        # 初始化检查器
        print("📦 初始化相似度检查器...")
        checker = ImageSimilarityChecker()
        
        # 示例1: 使用文件路径
        print("\n📁 示例1: 使用文件路径比较")
        print("注意: 请确保图片文件存在")
        # image1_path = "path/to/your/image1.jpg"
        # image2_path = "path/to/your/image2.jpg"
        # 
        # result = checker.comprehensive_check(image1_path, image2_path)
        # print_result(result, image1_path, image2_path)
        
        # 示例2: 创建测试图像
        print("\n🎨 示例2: 使用生成的测试图像")
        
        # 创建两个相似的测试图像
        similar_img1 = create_test_image(pattern="horizontal_lines")
        similar_img2 = create_test_image(pattern="horizontal_lines", noise=0.1)
        
        print("检查相似图像...")
        result1 = checker.comprehensive_check(similar_img1, similar_img2)
        print_result(result1, "相似图像1", "相似图像2")
        
        # 创建两个不同的测试图像
        diff_img1 = create_test_image(pattern="horizontal_lines")
        diff_img2 = create_test_image(pattern="vertical_lines")
        
        print("\n检查不同图像...")
        result2 = checker.comprehensive_check(diff_img1, diff_img2)
        print_result(result2, "水平线图像", "垂直线图像")
        
        # 示例3: 单独检查内容或风格
        print("\n📝 示例3: 单独检查内容相似度")
        content_similar, content_score = checker.check_content_similarity(
            similar_img1, similar_img2, threshold=0.7
        )
        print(f"内容相似度: {content_score:.4f}")
        print(f"是否相似: {'✅ 是' if content_similar else '❌ 否'}")
        
        print("\n🎨 示例4: 单独检查风格相似度")
        style_similar, style_score = checker.check_style_similarity(
            similar_img1, similar_img2, threshold=0.7
        )
        print(f"风格相似度: {style_score:.4f}")
        print(f"是否相似: {'✅ 是' if style_similar else '❌ 否'}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("💡 提示: 请确保已训练好模型文件存在")

def create_test_image(size=(64, 64), pattern="horizontal_lines", noise=0.0):
    """创建测试图像"""
    img = np.zeros(size, dtype=np.uint8)
    
    if pattern == "horizontal_lines":
        # 水平线条
        for i in range(0, size[0], 8):
            img[i:i+4, :] = 255
    elif pattern == "vertical_lines":
        # 垂直线条
        for i in range(0, size[1], 8):
            img[:, i:i+4] = 255
    elif pattern == "checkerboard":
        # 棋盘格
        for i in range(0, size[0], 16):
            for j in range(0, size[1], 16):
                if (i // 16 + j // 16) % 2 == 0:
                    img[i:i+16, j:j+16] = 255
    
    # 添加噪声
    if noise > 0:
        noise_array = np.random.normal(0, noise * 255, size)
        img = np.clip(img.astype(float) + noise_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img)

def print_result(result, name1, name2):
    """打印比较结果"""
    print(f"\n📊 比较结果: {name1} vs {name2}")
    print("-" * 40)
    print(f"📝 内容相似度: {result['content']['score']:.4f} {'✅' if result['content']['similar'] else '❌'}")
    print(f"🎨 风格相似度: {result['style']['score']:.4f} {'✅' if result['style']['similar'] else '❌'}")
    print(f"🎯 综合判断: {result['overall']['score']:.4f} {'✅ 相似' if result['overall']['similar'] else '❌ 不相似'}")

def batch_check_example():
    """批量检查示例"""
    print("\n📦 批量检查示例")
    print("=" * 50)
    
    try:
        checker = ImageSimilarityChecker()
        
        # 创建多个测试图像
        images = {
            "horizontal1": create_test_image("horizontal_lines"),
            "horizontal2": create_test_image("horizontal_lines", noise=0.05),
            "vertical1": create_test_image("vertical_lines"),
            "checkerboard1": create_test_image("checkerboard"),
        }
        
        # 批量比较
        comparisons = [
            ("horizontal1", "horizontal2"),
            ("horizontal1", "vertical1"),
            ("horizontal1", "checkerboard1"),
            ("vertical1", "checkerboard1"),
        ]
        
        print("🔍 批量相似度检查结果:")
        print("图像对\t\t内容相似度\t风格相似度\t综合判断")
        print("-" * 60)
        
        for name1, name2 in comparisons:
            result = checker.comprehensive_check(images[name1], images[name2])
            content_score = result['content']['score']
            style_score = result['style']['score']
            overall = "✅" if result['overall']['similar'] else "❌"
            
            print(f"{name1}-{name2}\t{content_score:.3f}\t\t{style_score:.3f}\t\t{overall}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    # 运行基本示例
    example_usage()
    
    # 运行批量检查示例
    batch_check_example()
    
    print("\n💡 使用提示:")
    print("1. 确保模型文件存在 (10kcontent_siamese_model_full.pth, 10kstyle_siamese_model_full.pth)")
    print("2. 或者使用调试模型 (debug_content_siamese_model.pth, debug_style_siamese_model.pth)")
    print("3. 可以通过命令行使用: python image_similarity_checker.py image1.jpg image2.jpg")
    print("4. 支持各种图像格式和numpy数组输入")