#!/usr/bin/env python3

import os
import sys

# Add the directory to Python path
current_dir = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0"
sys.path.insert(0, current_dir)

def test_comparison():
    """Test image comparison with sample images"""
    
    # Import after adding path
    from compare_images import ImageComparator
    
    # Model paths
    content_model_path = os.path.join(current_dir, "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/ckpt/content_final_20251015_112547.pth")
    style_model_path = os.path.join(current_dir, "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/ckpt/style_final_20251015_183825.pth")
    
    # Check if models exist
    if not os.path.exists(content_model_path):
        print(f"Content model not found: {content_model_path}")
        return
    
    if not os.path.exists(style_model_path):
        print(f"Style model not found: {style_model_path}")
        return
    
    # Sample image paths
    data_dir = "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/data/train"
    
    # Test case
    img1 = os.path.join(data_dir, "CELingDHJW_Da", "CELingDHJW_Da+艾.png")
    #img2 = os.path.join(data_dir, "FZCaiYTJ_Xian", "FZCaiYTJ_Xian+艾.png")
    #img2 = os.path.join(data_dir, "CELingDHJW_Da", "CELingDHJW_Da+挨.png")
    img2 = os.path.join(data_dir, "FZCaiYTJ_Xian", "FZCaiYTJ_Xian+挨.png")


    if os.path.exists(img1) and os.path.exists(img2):
        print("Testing same content, different style:")
        print(f"Image 1: {img1}")
        print(f"Image 2: {img2}")
        
        try:
            comparator = ImageComparator(content_model_path, style_model_path)
            result = comparator.compare_images(img1, img2)
            
            print(f"Content similarity: {result['content_score']:.4f} ({'Similar' if result['content_similar'] else 'Different'})")
            print(f"Style similarity: {result['style_score']:.4f} ({'Similar' if result['style_similar'] else 'Different'})")
            print()
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Sample images not found")

def main():
    """Main function for command line usage"""
    if len(sys.argv) == 3:
        # Command line usage
        from compare_images import ImageComparator
        
        image1_path = sys.argv[1]
        image2_path = sys.argv[2]
        
        content_model_path = os.path.join(current_dir, "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/ckpt/content_final_20251015_112547.pth")
        style_model_path = os.path.join(current_dir, "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/0/1/0/ckpt/style_final_20251015_183825.pth")
        
        try:
            comparator = ImageComparator(content_model_path, style_model_path)
            result = comparator.compare_images(image1_path, image2_path)
            
            print(f"Image 1: {image1_path}")
            print(f"Image 2: {image2_path}")
            print(f"Content similarity: {result['content_score']:.4f} ({'Similar' if result['content_similar'] else 'Different'})")
            print(f"Style similarity: {result['style_score']:.4f} ({'Similar' if result['style_similar'] else 'Different'})")
            print(f"Overall similarity: {result['overall_score']:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Test with sample images
        test_comparison()

if __name__ == "__main__":
    main()