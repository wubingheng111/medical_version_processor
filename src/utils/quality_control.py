import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import os

class QualityMetrics:
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """计算图像对比度"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.std(image)
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """计算图像亮度"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.mean(image)
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """计算图像清晰度"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)
    
    @staticmethod
    def calculate_noise_level(image: np.ndarray) -> float:
        """计算图像噪声水平"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # 使用高斯滤波作为参考
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        noise = image.astype(np.float32) - denoised
        return np.std(noise)
    
    @staticmethod
    def calculate_snr(image: np.ndarray) -> float:
        """计算信噪比"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        mean_signal = np.mean(image)
        noise_std = np.std(image)
        return 20 * np.log10(mean_signal / noise_std) if noise_std > 0 else 0

    @staticmethod
    def calculate_dynamic_range(image: np.ndarray) -> float:
        """计算动态范围"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.max(image) - np.min(image)

    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """计算图像熵（信息量）"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    @staticmethod
    def calculate_uniformity(image: np.ndarray) -> float:
        """计算图像均匀性"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        return np.sum(hist * hist)

    @staticmethod
    def calculate_edge_density(image: np.ndarray) -> float:
        """计算边缘密度"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(image, 100, 200)
        return np.sum(edges > 0) / (image.shape[0] * image.shape[1])

class QualityControl:
    def __init__(self):
        self.quality_thresholds = {
            'contrast': {'min': 30, 'max': 200},
            'brightness': {'min': 40, 'max': 220},
            'sharpness': {'min': 100, 'max': 1000},
            'noise_level': {'min': 0, 'max': 50},
            'snr': {'min': 20, 'max': float('inf')},
            'dynamic_range': {'min': 100, 'max': 250},
            'entropy': {'min': 6, 'max': 8},
            'uniformity': {'min': 0.1, 'max': 0.5},
            'edge_density': {'min': 0.05, 'max': 0.3}
        }
        self.quality_history = []
        self.optimization_strategies = {
            'contrast_enhancement': self._optimize_contrast,
            'noise_reduction': self._optimize_noise_reduction,
            'sharpening': self._optimize_sharpening,
            'brightness_adjustment': self._optimize_brightness
        }
    
    def assess_image_quality(self, image: Image.Image) -> Dict:
        """评估图像质量"""
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 计算各项指标
            metrics = {
                'contrast': QualityMetrics.calculate_contrast(img_array),
                'brightness': QualityMetrics.calculate_brightness(img_array),
                'sharpness': QualityMetrics.calculate_sharpness(img_array),
                'noise_level': QualityMetrics.calculate_noise_level(img_array),
                'snr': QualityMetrics.calculate_snr(img_array),
                'dynamic_range': QualityMetrics.calculate_dynamic_range(img_array),
                'entropy': QualityMetrics.calculate_entropy(img_array),
                'uniformity': QualityMetrics.calculate_uniformity(img_array),
                'edge_density': QualityMetrics.calculate_edge_density(img_array)
            }
            
            # 评估各项指标
            assessment = {}
            overall_score = 0
            weights = {
                'contrast': 0.15,
                'brightness': 0.1,
                'sharpness': 0.15,
                'noise_level': 0.15,
                'snr': 0.15,
                'dynamic_range': 0.1,
                'entropy': 0.1,
                'uniformity': 0.05,
                'edge_density': 0.05
            }
            
            for metric, value in metrics.items():
                thresholds = self.quality_thresholds[metric]
                if value < thresholds['min']:
                    status = 'low'
                    score = value / thresholds['min']
                elif value > thresholds['max']:
                    status = 'high'
                    score = thresholds['max'] / value
                else:
                    status = 'good'
                    score = 1.0
                
                assessment[metric] = {
                    'status': status,
                    'value': value,
                    'score': score,
                    'recommendation': self._generate_metric_recommendation(metric, status, value)
                }
                
                overall_score += score * weights[metric]
            
            # 记录评估历史
            quality_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'assessment': assessment,
                'overall_score': overall_score
            }
            self.quality_history.append(quality_record)
            
            return {
                'success': True,
                'metrics': metrics,
                'assessment': assessment,
                'overall_score': overall_score,
                'quality_grade': self._get_quality_grade(overall_score)
            }
            
        except Exception as e:
            return {'error': str(e)}

    def _generate_metric_recommendation(self, metric: str, status: str, value: float) -> str:
        """生成具体的指标改进建议"""
        recommendations = {
            'contrast': {
                'low': "建议增加对比度，可以尝试直方图均衡化或局部对比度增强",
                'high': "建议降低对比度，可以尝试gamma校正或对比度限制"
            },
            'brightness': {
                'low': "建议提高亮度，可以尝试线性变换或亮度增强",
                'high': "建议降低亮度，可以尝试线性变换或亮度衰减"
            },
            'sharpness': {
                'low': "建议增加锐化程度，可以尝试USM锐化或拉普拉斯锐化",
                'high': "建议降低锐化程度，可以尝试适当的平滑处理"
            },
            'noise_level': {
                'high': "建议进行降噪处理，可以尝试高斯滤波或双边滤波"
            },
            'snr': {
                'low': "建议改善信噪比，可以尝试降噪或信号增强"
            },
            'dynamic_range': {
                'low': "建议扩展动态范围，可以尝试直方图拉伸",
                'high': "建议压缩动态范围，可以尝试对数变换"
            },
            'entropy': {
                'low': "建议增加图像细节，可以尝试局部增强",
                'high': "建议适当降低复杂度，可以尝试平滑处理"
            },
            'uniformity': {
                'low': "建议增加图像均匀性，可以尝试局部均衡化",
                'high': "建议增加图像变化，可以尝试局部增强"
            },
            'edge_density': {
                'low': "建议增强边缘特征，可以尝试边缘增强",
                'high': "建议减少边缘噪声，可以尝试边缘平滑"
            }
        }
        
        if status in recommendations.get(metric, {}):
            return recommendations[metric][status]
        return None

    def _get_quality_grade(self, score: float) -> str:
        """根据总分获取质量等级"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        else:
            return 'D'

    def _optimize_contrast(self, image: np.ndarray, metrics: Dict) -> Dict:
        """优化对比度参数"""
        current_contrast = metrics['contrast']
        target_contrast = (self.quality_thresholds['contrast']['min'] + 
                         self.quality_thresholds['contrast']['max']) / 2
        
        if current_contrast < target_contrast:
            return {
                'method': 'clahe',
                'clip_limit': min(3.0, target_contrast / current_contrast),
                'tile_grid_size': (8, 8)
            }
        else:
            return {
                'method': 'gamma',
                'gamma': current_contrast / target_contrast
            }

    def _optimize_noise_reduction(self, image: np.ndarray, metrics: Dict) -> Dict:
        """优化降噪参数"""
        noise_level = metrics['noise_level']
        if noise_level > self.quality_thresholds['noise_level']['max']:
            if noise_level > 100:
                return {
                    'method': 'nlm',
                    'h': noise_level / 10,
                    'search_window': 21,
                    'block_size': 7
                }
            else:
                return {
                    'method': 'bilateral',
                    'sigma_color': noise_level / 5,
                    'sigma_space': 15
                }
        return {
            'method': 'gaussian',
            'kernel_size': 3,
            'sigma': noise_level / 10
        }

    def _optimize_sharpening(self, image: np.ndarray, metrics: Dict) -> Dict:
        """优化锐化参数"""
        current_sharpness = metrics['sharpness']
        target_sharpness = self.quality_thresholds['sharpness']['min']
        
        if current_sharpness < target_sharpness:
            return {
                'method': 'usm',
                'amount': min(2.0, target_sharpness / current_sharpness),
                'radius': 1,
                'threshold': 3
            }
        return {
            'method': 'none'
        }

    def _optimize_brightness(self, image: np.ndarray, metrics: Dict) -> Dict:
        """优化亮度参数"""
        current_brightness = metrics['brightness']
        target_brightness = (self.quality_thresholds['brightness']['min'] + 
                           self.quality_thresholds['brightness']['max']) / 2
        
        return {
            'method': 'linear',
            'alpha': 1.0,
            'beta': target_brightness - current_brightness
        }

    def suggest_improvements(self, image: Image.Image) -> Dict:
        """提供全面的改进建议"""
        try:
            quality_assessment = self.assess_image_quality(image)
            if 'error' in quality_assessment:
                return quality_assessment
            
            improvements = []
            priorities = []
            
            # 分析每个指标并生成改进建议
            for metric, assessment in quality_assessment['assessment'].items():
                if assessment['status'] != 'good':
                    improvement = {
                        'metric': metric,
                        'current_value': assessment['value'],
                        'target_range': [
                            self.quality_thresholds[metric]['min'],
                            self.quality_thresholds[metric]['max']
                        ],
                        'recommendation': assessment['recommendation'],
                        'priority': self._calculate_priority(metric, assessment)
                    }
                    improvements.append(improvement)
                    priorities.append(improvement['priority'])
            
            # 根据优先级排序建议
            sorted_improvements = [x for _, x in sorted(zip(priorities, improvements), 
                                                      reverse=True)]
            
            # 生成处理流程建议
            processing_pipeline = self._generate_processing_pipeline(sorted_improvements)
            
            return {
                'success': True,
                'current_quality_score': quality_assessment['overall_score'],
                'quality_grade': quality_assessment['quality_grade'],
                'improvements': sorted_improvements,
                'processing_pipeline': processing_pipeline
            }
            
        except Exception as e:
            return {'error': str(e)}

    def _calculate_priority(self, metric: str, assessment: Dict) -> float:
        """计算改进优先级"""
        # 基础权重
        base_weights = {
            'noise_level': 0.9,
            'snr': 0.85,
            'contrast': 0.8,
            'sharpness': 0.75,
            'brightness': 0.7,
            'dynamic_range': 0.65,
            'entropy': 0.6,
            'uniformity': 0.5,
            'edge_density': 0.45
        }
        
        # 计算偏离程度
        if assessment['status'] == 'low':
            deviation = 1 - (assessment['value'] / self.quality_thresholds[metric]['min'])
        else:  # high
            deviation = (assessment['value'] / self.quality_thresholds[metric]['max']) - 1
        
        # 根据偏离程度和基础权重计算优先级
        return base_weights[metric] * (1 + deviation)

    def _generate_processing_pipeline(self, improvements: List[Dict]) -> List[Dict]:
        """生成处理流程建议"""
        pipeline = []
        processed_metrics = set()
        
        # 定义处理步骤的依赖关系
        dependencies = {
            'sharpness': ['noise_level'],  # 降噪后再锐化
            'contrast': ['brightness'],     # 调整亮度后再调整对比度
            'edge_density': ['noise_level', 'sharpness']  # 降噪和锐化后再处理边缘
        }
        
        # 按优先级处理每个需要改进的指标
        for improvement in improvements:
            metric = improvement['metric']
            if metric in processed_metrics:
                continue
            
            # 检查依赖是否已处理
            if metric in dependencies:
                for dep in dependencies[metric]:
                    if dep not in processed_metrics:
                        # 查找依赖项的改进建议
                        dep_improvement = next(
                            (imp for imp in improvements if imp['metric'] == dep), 
                            None
                        )
                        if dep_improvement:
                            pipeline.append({
                                'step': len(pipeline) + 1,
                                'operation': self._get_operation_for_metric(dep),
                                'reason': f"处理{metric}前需要先处理{dep}",
                                'parameters': self._get_default_parameters(dep)
                            })
                            processed_metrics.add(dep)
            
            # 添加当前指标的处理步骤
            pipeline.append({
                'step': len(pipeline) + 1,
                'operation': self._get_operation_for_metric(metric),
                'reason': improvement['recommendation'],
                'parameters': self._get_default_parameters(metric)
            })
            processed_metrics.add(metric)
        
        return pipeline

    def _get_operation_for_metric(self, metric: str) -> str:
        """获取指标对应的处理操作"""
        operations = {
            'noise_level': 'denoise',
            'snr': 'denoise',
            'contrast': 'contrast_adjustment',
            'sharpness': 'sharpen',
            'brightness': 'contrast_adjustment',
            'dynamic_range': 'histogram_equalization',
            'entropy': 'contrast_adjustment',
            'uniformity': 'histogram_equalization',
            'edge_density': 'edge_detection'
        }
        return operations.get(metric, 'auto_enhance')

    def _get_default_parameters(self, metric: str) -> Dict:
        """获取处理操作的默认参数"""
        parameters = {
            'noise_level': {
                'method': 'bilateral'
            },
            'contrast': {
                'contrast': 1.2,
                'brightness': 0
            },
            'sharpness': {
                'amount': 1.5,
                'radius': 1,
                'threshold': 3
            },
            'brightness': {
                'contrast': 1.0,
                'brightness': 10
            },
            'dynamic_range': {},  # 直方图均衡化不需要参数
            'entropy': {
                'contrast': 1.3,
                'brightness': 0
            },
            'uniformity': {},  # 直方图均衡化不需要参数
            'edge_density': {
                'method': 'canny'
            }
        }
        return parameters.get(metric, {})

    def validate_processing_result(self, original_image: Image.Image, 
                                processed_image: Image.Image,
                                process_type: str) -> Dict:
        """验证处理结果"""
        try:
            # 评估原始图像和处理后图像的质量
            original_quality = self.assess_image_quality(original_image)
            processed_quality = self.assess_image_quality(processed_image)
            
            if 'error' in original_quality or 'error' in processed_quality:
                return {'error': '质量评估失败'}
            
            # 计算质量变化
            changes = {}
            for metric in original_quality['metrics']:
                original_value = original_quality['metrics'][metric]
                processed_value = processed_quality['metrics'][metric]
                
                change = {
                    'before': original_value,
                    'after': processed_value,
                    'difference': processed_value - original_value,
                    'percentage': ((processed_value - original_value) / original_value * 100 
                                 if original_value != 0 else float('inf'))
                }
                changes[metric] = change
            
            # 根据处理类型评估结果
            validation_result = self._evaluate_changes(changes, process_type)
            
            return {
                'success': True,
                'changes': changes,
                'validation': validation_result
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _evaluate_changes(self, changes: Dict, process_type: str) -> Dict:
        """评估变化的有效性"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'suggestions': []
        }
        
        # 根据处理类型设置期望的变化
        expectations = {
            'contrast_enhancement': {
                'contrast': {'direction': 'increase', 'min_change': 10},
                'brightness': {'direction': 'stable', 'max_change': 20}
            },
            'noise_reduction': {
                'noise_level': {'direction': 'decrease', 'min_change': -5},
                'snr': {'direction': 'increase', 'min_change': 2}
            },
            'sharpening': {
                'sharpness': {'direction': 'increase', 'min_change': 50},
                'noise_level': {'direction': 'stable', 'max_change': 10}
            }
        }
        
        if process_type in expectations:
            expected = expectations[process_type]
            for metric, expectation in expected.items():
                change = changes[metric]
                
                if expectation['direction'] == 'increase':
                    if change['percentage'] < expectation['min_change']:
                        validation['warnings'].append(
                            f"{metric}增加不足 (期望>{expectation['min_change']}%, 实际{change['percentage']:.1f}%)"
                        )
                        validation['is_valid'] = False
                
                elif expectation['direction'] == 'decrease':
                    if change['percentage'] > -expectation['min_change']:
                        validation['warnings'].append(
                            f"{metric}减少不足 (期望<{-expectation['min_change']}%, 实际{change['percentage']:.1f}%)"
                        )
                        validation['is_valid'] = False
                
                elif expectation['direction'] == 'stable':
                    if abs(change['percentage']) > expectation['max_change']:
                        validation['warnings'].append(
                            f"{metric}变化过大 (期望<{expectation['max_change']}%, 实际{abs(change['percentage']):.1f}%)"
                        )
                        validation['is_valid'] = False
        
        # 添加改进建议
        if not validation['is_valid']:
            validation['suggestions'].extend(self._generate_suggestions(changes, process_type))
        
        return validation
    
    def _generate_suggestions(self, changes: Dict, process_type: str) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if process_type == 'contrast_enhancement':
            if changes['contrast']['percentage'] < 10:
                suggestions.append("建议增加对比度增强的强度")
            if abs(changes['brightness']['percentage']) > 20:
                suggestions.append("建议减小亮度调整的幅度")
        
        elif process_type == 'noise_reduction':
            if changes['noise_level']['percentage'] > -5:
                suggestions.append("建议增加降噪强度")
            if changes['snr']['percentage'] < 2:
                suggestions.append("建议尝试其他降噪算法")
        
        elif process_type == 'sharpening':
            if changes['sharpness']['percentage'] < 50:
                suggestions.append("建议增加锐化强度")
            if changes['noise_level']['percentage'] > 10:
                suggestions.append("建议在锐化前先进行降噪处理")
        
        return suggestions
    
    def optimize_parameters(self, image: Image.Image, process_type: str, 
                          current_params: Dict) -> Dict:
        """优化处理参数"""
        try:
            # 评估当前图像质量
            current_quality = self.assess_image_quality(image)
            if 'error' in current_quality:
                return {'error': '质量评估失败'}
            
            # 根据处理类型和当前质量生成优化建议
            optimization = self._generate_parameter_optimization(
                current_quality['metrics'],
                process_type,
                current_params
            )
            
            return {
                'success': True,
                'current_quality': current_quality,
                'optimized_params': optimization['params'],
                'expected_improvements': optimization['improvements']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_parameter_optimization(self, metrics: Dict, process_type: str, 
                                      current_params: Dict) -> Dict:
        """生成参数优化建议"""
        optimization = {
            'params': current_params.copy(),
            'improvements': []
        }
        
        if process_type == 'contrast_enhancement':
            if metrics['contrast'] < self.quality_thresholds['contrast']['min']:
                optimization['params']['contrast_factor'] = min(
                    current_params.get('contrast_factor', 1.0) * 1.2,
                    2.0
                )
                optimization['improvements'].append("预期对比度提升20%")
        
        elif process_type == 'noise_reduction':
            if metrics['noise_level'] > self.quality_thresholds['noise_level']['max']:
                optimization['params']['strength'] = min(
                    current_params.get('strength', 0.5) + 0.2,
                    1.0
                )
                optimization['improvements'].append("预期噪声降低30%")
        
        elif process_type == 'sharpening':
            if metrics['sharpness'] < self.quality_thresholds['sharpness']['min']:
                optimization['params']['radius'] = max(
                    current_params.get('radius', 1.0) - 0.2,
                    0.5
                )
                optimization['params']['amount'] = min(
                    current_params.get('amount', 1.0) * 1.3,
                    2.0
                )
                optimization['improvements'].append("预期清晰度提升30%")
        
        return optimization
    
    def export_quality_report(self, export_path: str) -> Dict:
        """导出质量报告"""
        try:
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'quality_thresholds': self.quality_thresholds,
                'quality_history': self.quality_history
            }
            
            # 添加统计信息
            stats = self._calculate_quality_statistics()
            report['statistics'] = stats
            
            # 保存报告
            filename = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(export_path, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return {'success': True, 'filepath': filepath}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_quality_statistics(self) -> Dict:
        """计算质量统计信息"""
        if not self.quality_history:
            return {}
        
        stats = {'metrics': {}}
        
        # 获取所有指标
        metrics = self.quality_history[0]['metrics'].keys()
        
        for metric in metrics:
            values = [h['metrics'][metric] for h in self.quality_history]
            stats['metrics'][metric] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # 计算质量评估结果的分布
        assessment_stats = {'total': len(self.quality_history)}
        for status in ['good', 'low', 'high']:
            count = sum(
                1 for h in self.quality_history
                for m in h['assessment'].values()
                if m['status'] == status
            )
            assessment_stats[status] = count
        
        stats['assessment_distribution'] = assessment_stats
        
        return stats 