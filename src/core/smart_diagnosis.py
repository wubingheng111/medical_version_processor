import numpy as np
import cv2
from typing import Dict, Any, List
import zhipuai
import os
from PIL import Image
import io

class SmartDiagnosisSystem:
    """智能诊断系统类"""
    
    def __init__(self):
        self.api_key = os.getenv("ZHIPUAI_API_KEY")
        self.detection_history = []
    
    def initialize_vlm(self, api_key: str = None):
        """初始化智谱AI接口"""
        try:
            if api_key:
                self.api_key = api_key
            
            if not self.api_key:
                return {'success': False, 'error': '未设置API Key'}
            
            zhipuai.api_key = self.api_key
            return {'success': True, 'message': '智谱AI接口初始化成功'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def detect_lesions(self, image: np.ndarray) -> Dict[str, Any]:
        """检测病变区域"""
        try:
            # 图像预处理
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 使用自适应阈值进行分割
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 形态学操作
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 找到轮廓
            contours, _ = cv2.findContours(
                opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 分析每个轮廓
            lesions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 过滤小区域
                    x, y, w, h = cv2.boundingRect(contour)
                    lesions.append({
                        'bbox': [x, y, w, h],
                        'area': area,
                        'contour': contour.tolist()
                    })
            
            result = {
                'success': True,
                'lesions': lesions,
                'count': len(lesions)
            }
            
            self.detection_history.append({
                'operation': 'detect_lesions',
                'result': result
            })
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_with_vlm(self, image: np.ndarray, prompt: str) -> Dict[str, Any]:
        """使用智谱AI的VLM模型分析图像"""
        try:
            if not self.api_key:
                return {'success': False, 'error': '未初始化智谱AI接口'}
            
            # 将numpy数组转换为PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # 将图像转换为bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # 调用智谱AI的VLM接口
            response = zhipuai.model_api.invoke(
                model="glm-4v",
                prompt=prompt,
                file_list=[{
                    "type": "image",
                    "content": img_byte_arr
                }]
            )
            
            if response.get('code') == 200:
                result = {
                    'success': True,
                    'analysis': response.get('data', {}).get('text', ''),
                    'prompt': prompt
                }
            else:
                result = {
                    'success': False,
                    'error': response.get('msg', '调用智谱AI接口失败')
                }
            
            self.detection_history.append({
                'operation': 'analyze_with_vlm',
                'params': {'prompt': prompt},
                'result': result
            })
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_detection_history(self) -> List[Dict]:
        """获取检测历史"""
        return self.detection_history
    
    def analyze_lesion_characteristics(self, image: np.ndarray, lesion: Dict) -> Dict[str, Any]:
        """分析病变特征"""
        try:
            x, y, w, h = lesion['bbox']
            roi = image[y:y+h, x:x+w]
            
            # 计算颜色特征
            mean_color = np.mean(roi, axis=(0,1)).tolist()
            std_color = np.std(roi, axis=(0,1)).tolist()
            
            # 计算纹理特征
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            glcm = self._calculate_glcm(gray_roi)
            texture_features = self._extract_texture_features(glcm)
            
            characteristics = {
                'color': {
                    'mean': mean_color,
                    'std': std_color
                },
                'texture': texture_features,
                'shape': {
                    'area': lesion['area'],
                    'aspect_ratio': w/h if h > 0 else 0
                }
            }
            
            return {'success': True, 'characteristics': characteristics}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_glcm(self, image: np.ndarray) -> np.ndarray:
        """计算灰度共生矩阵"""
        glcm = np.zeros((256, 256))
        h, w = image.shape
        for i in range(h-1):
            for j in range(w-1):
                glcm[image[i,j], image[i,j+1]] += 1
        return glcm / glcm.sum()
    
    def _extract_texture_features(self, glcm: np.ndarray) -> Dict[str, float]:
        """提取纹理特征"""
        contrast = np.sum(np.square(np.arange(256)[:,None] - np.arange(256)) * glcm)
        correlation = np.sum((np.arange(256)[:,None] - np.mean(glcm)) * 
                           (np.arange(256) - np.mean(glcm)) * glcm) / (np.std(glcm) ** 2)
        energy = np.sum(np.square(glcm))
        homogeneity = np.sum(glcm / (1 + np.square(np.arange(256)[:,None] - np.arange(256))))
        
        return {
            'contrast': float(contrast),
            'correlation': float(correlation),
            'energy': float(energy),
            'homogeneity': float(homogeneity)
        } 