import numpy as np
from PIL import Image
import cv2
from utils.smart_diagnosis import SmartDiagnosisSystem

def test_lesion_annotation():
    # 创建一个测试图像（或加载一个示例医学图像）
    # 这里我们创建一个简单的测试图像
    img_size = (400, 400)
    test_image = Image.new('RGB', img_size, color='white')
    
    # 创建SmartDiagnosisSystem实例
    diagnosis_system = SmartDiagnosisSystem()
    
    # 模拟一些病变区域
    test_lesions = [
        {
            'bbox': [50, 50, 100, 100],  # x, y, width, height
            'area': 10000,
            'centroid': [100, 100]
        },
        {
            'bbox': [200, 200, 80, 60],
            'area': 4800,
            'centroid': [240, 230]
        }
    ]
    
    # 模拟置信度分数
    confidence_scores = [0.85, 0.92]
    
    # 绘制病变标注
    annotated_image = diagnosis_system.draw_lesions(test_image, test_lesions, confidence_scores)
    
    # 保存结果
    annotated_image.save('test_annotated_image.png')
    print("已保存标注后的图像到: test_annotated_image.png")

if __name__ == "__main__":
    test_lesion_annotation() 