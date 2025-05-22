import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import json
import os
import base64
from io import BytesIO
from utils.vlm_analyzer import VLMAnalyzer
import re

class SmartDiagnosisSystem:
    """智能医学影像诊断系统
    
    主要职责：
    1. 病变检测和分析
    2. 器官分割
    3. 医学指标测量
    """
    
    def __init__(self):
        self.vlm_analyzer = VLMAnalyzer()
        self.detection_models = {}
        self.current_case = None
        
    def analyze_image(self, image, analysis_type):
        """分析医学图像
        
        Args:
            image: PIL Image对象
            analysis_type: 分析类型，可选值：
                - "lesion_detection": 病变检测
                - "organ_segmentation": 器官分割
                - "measurement": 关键指标测量
        
        Returns:
            dict: 分析结果
        """
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_type,
                'findings': [],
                'measurements': {},
                'confidence_score': 0.0
            }
            
            # 执行图像分析
            if analysis_type == "lesion_detection":
                results.update(self._detect_lesions(image))
            elif analysis_type == "organ_segmentation":
                results.update(self._segment_organs(image))
            elif analysis_type == "measurement":
                results.update(self._perform_measurements(image))
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _detect_lesions(self, image):
        """病变检测
        
        使用多种方法进行病变检测：
        1. 传统图像处理方法
        2. VLM模型分析（作为辅助验证）
        """
        results = {
            'lesions': [],
            'confidence_scores': [],
            'detection_method': None
        }
        
        try:
            # 1. 使用传统方法检测病变区域
            traditional_results = self._traditional_detection(image)
            if traditional_results['lesions']:
                results.update(traditional_results)
                results['detection_method'] = 'traditional'
                
                # 2. 使用VLM进行验证和调整
                try:
                    vlm_result = self._validate_with_vlm(image, traditional_results)
                    if vlm_result['success']:
                        # 根据VLM分析调整置信度
                        for i, lesion in enumerate(results['lesions']):
                            adjusted_confidence = self._adjust_confidence_with_vlm(
                                lesion,
                                results['confidence_scores'][i],
                                vlm_result['analysis']
                            )
                            results['confidence_scores'][i] = adjusted_confidence
                        results['detection_method'] = 'hybrid'
                except Exception as e:
                    print(f"VLM验证失败: {str(e)}")
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _traditional_detection(self, image):
        """使用传统图像处理方法进行检测"""
        results = {
            'lesions': [],
            'confidence_scores': []
        }
        
        try:
            # 图像预处理
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Otsu's阈值法
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
            # 轮廓检测
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 计算图像统计特征
            mean_intensity = np.mean(gray)
            image_area = gray.shape[0] * gray.shape[1]
            min_area = int(image_area * 0.001)  # 最小面积为图像面积的0.1%
            max_area = int(image_area * 0.3)    # 最大面积为图像面积的30%
            
            # 分析每个候选区域
            for contour in contours:
                try:
                    area = cv2.contourArea(contour)
                    if min_area < area < max_area:
                        # 计算特征
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w)/h if h > 0 else 0
                        
                        # 计算置信度
                        confidence = self._calculate_confidence(
                            area=area,
                            circularity=circularity,
                            aspect_ratio=aspect_ratio,
                            intensity_diff=abs(cv2.mean(gray, mask=cv2.drawContours(
                                np.zeros_like(gray), [contour], 0, 255, -1
                            ))[0] - mean_intensity)
                        )
                        
                        if confidence > 0.4:  # 置信度阈值
                            results['lesions'].append({
                                'bbox': [x, y, w, h],
                                'area': area,
                                'centroid': [x + w//2, y + h//2],
                                'circularity': circularity,
                                'aspect_ratio': aspect_ratio
                            })
                            results['confidence_scores'].append(confidence)
                            
                except Exception as e:
                    print(f"处理轮廓时出错: {str(e)}")
                    continue
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _validate_with_vlm(self, image, detection_results):
        """使用VLM验证检测结果"""
        try:
            # 将图像转换为base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 构建提示词
            prompt = """请仔细分析这张医学图像，验证检测到的病变区域。
            已检测到的区域：
            """
            for i, lesion in enumerate(detection_results['lesions']):
                prompt += f"\n区域{i+1}: 位置({lesion['centroid']}), 面积{lesion['area']:.1f}"
            
            prompt += """\n请评估：
            1. 这些区域是否确实为病变
            2. 是否存在漏检的病变区域
            3. 每个区域的特征描述和严重程度
            
            请以结构化的方式输出分析结果。"""
            
            return self.vlm_analyzer.analyze_with_prompt(img_str, prompt)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_confidence(self, area, circularity, aspect_ratio, intensity_diff):
        """计算病变区域的置信度"""
        try:
            # 面积得分
            area_score = min(area / 1000.0, 1.0) if area > 0 else 0.0
            
            # 圆形度得分（病变通常较为规则）
            circularity_score = max(0.0, min(circularity, 1.0))
            
            # 长宽比得分（排除过于细长的区域）
            aspect_ratio_score = 1.0 / (1.0 + abs(aspect_ratio - 1.0))
            
            # 强度差异得分
            intensity_score = min(intensity_diff / 50.0, 1.0)
            
            # 综合计算置信度
            confidence = (area_score + circularity_score + aspect_ratio_score + intensity_score) / 4
            return min(max(confidence, 0.0), 0.99)
            
        except Exception as e:
            print(f"置信度计算错误: {str(e)}")
            return 0.3
    
    def draw_lesions(self, image, lesions, confidence_scores):
        """在图像上绘制病变标注"""
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 创建RGB图像副本
            if len(img_array.shape) == 2:
                annotated_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                else:
                    annotated_img = img_array.copy()
            
            # 为每个病变区域添加标注
            for lesion, confidence in zip(lesions, confidence_scores):
                try:
                    x, y, w, h = lesion['bbox']
                    
                    # 设置颜色（红色到绿色的渐变）
                    color = (
                        int(255 * (1 - confidence)),  # R
                        int(255 * confidence),        # G
                        0                            # B
                    )
                    
                    # 绘制半透明填充
                    overlay = annotated_img.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                    cv2.addWeighted(overlay, 0.3, annotated_img, 0.7, 0, annotated_img)
                    
                    # 绘制边框
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
                    
                    # 添加置信度标签
                    label = f"{confidence:.1%}"
                    cv2.putText(annotated_img, label, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 绘制中心点
                    center_x, center_y = lesion['centroid']
                    cv2.circle(annotated_img, (int(center_x), int(center_y)), 3, color, -1)
                    
                except Exception as e:
                    print(f"绘制单个病变标注时出错: {str(e)}")
                    continue
            
            # 转换回PIL Image
            return Image.fromarray(annotated_img)
            
        except Exception as e:
            print(f"绘制病变标注时出错: {str(e)}")
            return image
    
    def _segment_organs(self, image):
        """器官分割"""
        # TODO: 实现器官分割算法
        return {
            'segments': [],
            'organ_types': []
        }
    
    def _perform_measurements(self, image):
        """进行关键指标测量"""
        # TODO: 实现测量算法
        return {
            'dimensions': {},
            'densities': {},
            'ratios': {}
        }

    def load_models(self, model_configs):
        """加载诊断模型"""
        for model_name, config in model_configs.items():
            try:
                # TODO: 实现模型加载逻辑
                self.detection_models[model_name] = None
            except Exception as e:
                print(f"模型 {model_name} 加载失败: {str(e)}")
    
    def track_progress(self, case_id, previous_results):
        """跟踪病变进展"""
        if not self.current_case or self.current_case['id'] != case_id:
            return {'error': '案例ID不匹配'}
            
        try:
            progress_analysis = {
                'case_id': case_id,
                'timestamp': datetime.now().isoformat(),
                'changes': [],
                'trend': None
            }
            
            # TODO: 实现进展分析算法
            
            return progress_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_report(self, analysis_results, template_type='standard'):
        """生成分析报告"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'template_type': template_type,
                'findings': [],
                'measurements': {},
                'recommendations': []
            }
            
            # 处理分析结果
            if 'lesions' in analysis_results:
                for i, lesion in enumerate(analysis_results['lesions']):
                    finding = {
                        'id': f'L{i+1}',
                        'type': 'lesion',
                        'location': f'位于 ({lesion["centroid"][0]}, {lesion["centroid"][1]})',
                        'size': f'{lesion["area"]:.1f} 像素',
                        'confidence': f'{analysis_results["confidence_scores"][i]:.2%}'
                    }
                    report['findings'].append(finding)
            
            # 添加建议
            if report['findings']:
                report['recommendations'].append({
                    'type': 'follow_up',
                    'description': '建议进行定期复查，跟踪病变变化。'
                })
            
            return report
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_analysis(self, case_id, analysis_results, save_path):
        """保存分析结果"""
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            filename = f'analysis_{case_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            filepath = os.path.join(save_path, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            
            return {'success': True, 'filepath': filepath}
            
        except Exception as e:
            return {'error': str(e)} 