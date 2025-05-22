import cv2
import numpy as np
from typing import Dict, Any, Tuple

class ImageProcessor:
    """图像处理核心类"""
    
    def __init__(self):
        self.current_image = None
        self.processing_history = []
        
    def load_image(self, image_path: str) -> Dict[str, Any]:
        """加载图像"""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                return {'success': False, 'error': '无法加载图像'}
            
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            return {
                'success': True,
                'image': self.current_image,
                'shape': self.current_image.shape
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def adjust_brightness_contrast(self, brightness: float = 0, contrast: float = 1) -> Dict[str, Any]:
        """调整亮度和对比度"""
        try:
            if self.current_image is None:
                return {'success': False, 'error': '没有加载图像'}
            
            adjusted = cv2.convertScaleAbs(self.current_image, alpha=contrast, beta=brightness)
            self.processing_history.append({
                'operation': 'adjust_brightness_contrast',
                'params': {'brightness': brightness, 'contrast': contrast}
            })
            return {'success': True, 'image': adjusted}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def apply_filter(self, filter_type: str, kernel_size: int = 3) -> Dict[str, Any]:
        """应用图像滤波"""
        try:
            if self.current_image is None:
                return {'success': False, 'error': '没有加载图像'}
            
            if filter_type == 'gaussian':
                filtered = cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)
            elif filter_type == 'median':
                filtered = cv2.medianBlur(self.current_image, kernel_size)
            else:
                return {'success': False, 'error': '不支持的滤波类型'}
            
            self.processing_history.append({
                'operation': 'apply_filter',
                'params': {'filter_type': filter_type, 'kernel_size': kernel_size}
            })
            return {'success': True, 'image': filtered}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def enhance_details(self, strength: float = 1.0) -> Dict[str, Any]:
        """增强图像细节"""
        try:
            if self.current_image is None:
                return {'success': False, 'error': '没有加载图像'}
            
            # 使用USM锐化
            gaussian = cv2.GaussianBlur(self.current_image, (0, 0), 3.0)
            unsharp_image = cv2.addWeighted(self.current_image, 1.0 + strength, 
                                          gaussian, -strength, 0)
            
            self.processing_history.append({
                'operation': 'enhance_details',
                'params': {'strength': strength}
            })
            return {'success': True, 'image': unsharp_image}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_processing_history(self) -> list:
        """获取处理历史"""
        return self.processing_history
    
    def reset_image(self) -> Dict[str, Any]:
        """重置图像处理"""
        try:
            if self.current_image is None:
                return {'success': False, 'error': '没有加载图像'}
            
            self.processing_history = []
            return {'success': True, 'message': '图像已重置'}
        except Exception as e:
            return {'success': False, 'error': str(e)} 