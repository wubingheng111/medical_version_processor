import json
from datetime import datetime
import os
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from collections import Counter

class WorkflowStep:
    """工作流步骤"""
    def __init__(self, step_type: str, parameters: Dict = None, 
                 conditions: Dict = None, next_steps: List[str] = None):
        self.step_type = step_type
        self.parameters = parameters or {}
        self.conditions = conditions or {}
        self.next_steps = next_steps or []
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            'step_type': self.step_type,
            'parameters': self.parameters,
            'conditions': self.conditions,
            'next_steps': self.next_steps,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowStep':
        step = cls(
            step_type=data['step_type'],
            parameters=data['parameters'],
            conditions=data['conditions'],
            next_steps=data['next_steps']
        )
        step.created_at = data['created_at']
        return step

class Workflow:
    """工作流定义"""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps = {}
        self.start_step = None
        self.created_at = datetime.now().isoformat()
    
    def add_step(self, step_id: str, step: WorkflowStep) -> None:
        self.steps[step_id] = step
        if not self.start_step:
            self.start_step = step_id
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'steps': {k: v.to_dict() for k, v in self.steps.items()},
            'start_step': self.start_step,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Workflow':
        workflow = cls(data['name'], data['description'])
        workflow.created_at = data['created_at']
        workflow.start_step = data['start_step']
        
        for step_id, step_data in data['steps'].items():
            workflow.steps[step_id] = WorkflowStep.from_dict(step_data)
        
        return workflow

class SmartWorkflowSystem:
    """智能工作流系统
    
    主要职责：
    1. 工作流定义和管理
    2. 工作流执行和监控
    3. 参数优化和调整
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.workflows = {}
        self.parameter_models = {}
        self.execution_history = []
        self.load_workflows()
        self.load_parameter_models()
        self._init_default_workflows()
    
    def _init_default_workflows(self):
        """初始化默认工作流模板"""
        # 1. 病变检测工作流
        lesion_detection_workflow = {
            'name': 'lesion_detection_workflow',
            'description': '病变检测标准工作流',
            'steps': [
                {
                    'type': 'image_preprocessing',
                    'parameters': {
                        'operations': ['grayscale', 'histogram_equalization'],
                        'quality_threshold': 0.7
                    },
                    'conditions': {
                        'quality_check': {'threshold': 0.7, 'next_step': 'enhancement'}
                    },
                    'next_steps': ['enhancement']
                },
                {
                    'type': 'image_enhancement',
                    'parameters': {
                        'contrast': 1.2,
                        'brightness': 10,
                        'clahe_clip_limit': 2.0,
                        'clahe_grid_size': (8, 8)
                    },
                    'next_steps': ['segmentation']
                },
                {
                    'type': 'image_segmentation',
                    'parameters': {
                        'method': 'adaptive_threshold',
                        'block_size': 11,
                        'c_value': 2
                    },
                    'next_steps': ['lesion_detection']
                },
                {
                    'type': 'lesion_detection',
                    'parameters': {
                        'confidence_threshold': 0.4,
                        'min_area_ratio': 0.001,
                        'max_area_ratio': 0.3
                    },
                    'next_steps': ['quality_assessment']
                },
                {
                    'type': 'quality_assessment',
                    'parameters': {
                        'metrics': ['contrast', 'brightness', 'sharpness', 'noise_level', 'snr']
                    },
                    'conditions': {
                        'quality_check': {'threshold': 0.8, 'next_step': 'enhancement'}
                    }
                }
            ]
        }
        
        # 2. 器官分割工作流
        organ_segmentation_workflow = {
            'name': 'organ_segmentation_workflow',
            'description': '器官分割标准工作流',
            'steps': [
                {
                    'type': 'image_preprocessing',
                    'parameters': {
                        'operations': ['grayscale', 'normalize'],
                        'quality_threshold': 0.7
                    },
                    'next_steps': ['enhancement']
                },
                {
                    'type': 'image_enhancement',
                    'parameters': {
                        'contrast': 1.1,
                        'brightness': 0,
                        'gamma': 1.0
                    },
                    'next_steps': ['segmentation']
                },
                {
                    'type': 'image_segmentation',
                    'parameters': {
                        'method': 'watershed',
                        'marker_threshold': 0.3
                    },
                    'next_steps': ['organ_detection']
                },
                {
                    'type': 'organ_detection',
                    'parameters': {
                        'target_organs': ['liver', 'kidney', 'spleen'],
                        'confidence_threshold': 0.5
                    },
                    'next_steps': ['measurement']
                },
                {
                    'type': 'measurement',
                    'parameters': {
                        'metrics': ['volume', 'density', 'shape_index']
                    }
                }
            ]
        }
        
        # 3. 图像质量优化工作流
        quality_optimization_workflow = {
            'name': 'quality_optimization_workflow',
            'description': '图像质量优化工作流',
            'steps': [
                {
                    'type': 'quality_assessment',
                    'parameters': {
                        'metrics': ['contrast', 'brightness', 'sharpness', 'noise_level', 'snr']
                    },
                    'conditions': {
                        'contrast_check': {'threshold': 0.7, 'next_step': 'contrast_enhancement'},
                        'brightness_check': {'threshold': 0.7, 'next_step': 'brightness_adjustment'},
                        'sharpness_check': {'threshold': 0.7, 'next_step': 'sharpness_enhancement'},
                        'noise_check': {'threshold': 0.3, 'next_step': 'noise_reduction'}
                    },
                    'next_steps': ['enhancement_selection']
                },
                {
                    'type': 'enhancement_selection',
                    'parameters': {
                        'methods': {
                            'contrast_enhancement': {
                                'clahe_clip_limit': 2.0,
                                'clahe_grid_size': (8, 8)
                            },
                            'brightness_adjustment': {
                                'gamma': 1.0,
                                'brightness_offset': 10
                            },
                            'sharpness_enhancement': {
                                'kernel_size': 3,
                                'sigma': 1.0
                            },
                            'noise_reduction': {
                                'method': 'non_local_means',
                                'h': 10,
                                'template_window_size': 7,
                                'search_window_size': 21
                            }
                        }
                    },
                    'next_steps': ['final_assessment']
                },
                {
                    'type': 'final_assessment',
                    'parameters': {
                        'metrics': ['contrast', 'brightness', 'sharpness', 'noise_level', 'snr'],
                        'min_improvement': 0.2
                    }
                }
            ]
        }
        
        # 创建工作流
        for workflow_data in [lesion_detection_workflow, organ_segmentation_workflow, 
                            quality_optimization_workflow]:
            self.create_workflow(
                name=workflow_data['name'],
                description=workflow_data['description'],
                steps=workflow_data['steps']
            )
    
    def load_workflows(self) -> None:
        """加载所有工作流"""
        try:
            workflow_path = os.path.join(self.storage_path, 'workflows')
            if not os.path.exists(workflow_path):
                os.makedirs(workflow_path)
                return
            
            for filename in os.listdir(workflow_path):
                if filename.endswith('.json'):
                    with open(os.path.join(workflow_path, filename), 'r', encoding='utf-8') as f:
                        workflow_data = json.load(f)
                        workflow = Workflow.from_dict(workflow_data)
                        self.workflows[workflow.name] = workflow
                        
        except Exception as e:
            print(f"加载工作流失败: {str(e)}")
    
    def load_parameter_models(self) -> None:
        """加载参数预测模型"""
        try:
            model_path = os.path.join(self.storage_path, 'models')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                return
            
            for filename in os.listdir(model_path):
                if filename.endswith('.joblib'):
                    model_name = os.path.splitext(filename)[0]
                    model = joblib.load(os.path.join(model_path, filename))
                    self.parameter_models[model_name] = model
                    
        except Exception as e:
            print(f"加载参数模型失败: {str(e)}")
    
    def create_workflow(self, name: str, description: str, steps: List[Dict]) -> Dict:
        """创建新工作流"""
        try:
            if name in self.workflows:
                return {'error': '工作流名称已存在'}
            
            workflow = Workflow(name, description)
            
            for i, step_data in enumerate(steps):
                step_id = f"step_{i+1}"
                step = WorkflowStep(
                    step_type=step_data['type'],
                    parameters=step_data.get('parameters', {}),
                    conditions=step_data.get('conditions', {}),
                    next_steps=step_data.get('next_steps', [])
                )
                workflow.add_step(step_id, step)
            
            # 保存工作流
            result = self.save_workflow(workflow)
            if 'error' in result:
                return result
            
            self.workflows[name] = workflow
            return {'success': True, 'workflow': workflow.to_dict()}
            
        except Exception as e:
            return {'error': str(e)}
    
    def execute_workflow(self, workflow_name: str, input_image: Image.Image,
                        custom_params: Dict = None) -> Dict:
        """执行工作流"""
        try:
            if workflow_name not in self.workflows:
                return {'error': '工作流不存在'}
            
            workflow = self.workflows[workflow_name]
            if not workflow.start_step:
                return {'error': '工作流没有起始步骤'}
            
            execution_result = {
                'workflow_name': workflow_name,
                'started_at': datetime.now().isoformat(),
                'steps_executed': [],
                'final_image': None,
                'intermediate_results': {}
            }
            
            current_image = input_image
            current_step_id = workflow.start_step
            
            while current_step_id:
                step = workflow.steps[current_step_id]
                
                # 合并自定义参数
                step_params = step.parameters.copy()
                if custom_params and current_step_id in custom_params:
                    step_params.update(custom_params[current_step_id])
                
                # 预测最优参数
                predicted_params = self._predict_parameters(
                    step.step_type,
                    current_image,
                    step_params
                )
                if predicted_params:
                    step_params.update(predicted_params)
                
                # 执行处理步骤
                result = self._execute_step(step.step_type, current_image, step_params)
                
                if 'error' in result:
                    return {'error': f"步骤 {current_step_id} 执行失败: {result['error']}"}
                
                execution_result['steps_executed'].append({
                    'step_id': current_step_id,
                    'step_type': step.step_type,
                    'parameters': step_params,
                    'executed_at': datetime.now().isoformat()
                })
                
                execution_result['intermediate_results'][current_step_id] = result
                current_image = result['processed_image']
                
                # 确定下一步
                current_step_id = self._determine_next_step(step, result)
            
            execution_result['final_image'] = current_image
            execution_result['completed_at'] = datetime.now().isoformat()
            
            # 记录执行历史
            self.execution_history.append(execution_result)
            
            return {'success': True, 'execution_result': execution_result}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _predict_parameters(self, step_type: str, image: Image.Image,
                          base_params: Dict) -> Optional[Dict]:
        """预测处理参数"""
        try:
            if step_type not in self.parameter_models:
                return None
            
            # 提取图像特征
            features = self._extract_image_features(image)
            
            # 使用模型预测参数
            model = self.parameter_models[step_type]
            predictions = model.predict([features])[0]
            
            # 将预测结果转换为参数字典
            param_names = model.feature_names_out_
            predicted_params = {
                name: value for name, value in zip(param_names, predictions)
            }
            
            return predicted_params
            
        except Exception as e:
            print(f"参数预测失败: {str(e)}")
            return None
    
    def _extract_image_features(self, image: Image.Image) -> np.ndarray:
        """提取图像特征"""
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 计算基本统计特征
            features = [
                np.mean(img_array),  # 平均亮度
                np.std(img_array),   # 对比度
                np.median(img_array), # 中位数
                np.percentile(img_array, 10),  # 10%分位数
                np.percentile(img_array, 90),  # 90%分位数
            ]
            
            # 计算纹理特征
            glcm = self._calculate_glcm(img_array)
            features.extend([
                np.mean(glcm),  # 平均值
                np.std(glcm),   # 标准差
                np.sum(glcm * np.log(glcm + 1e-6))  # 熵
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            return np.zeros(8)  # 返回默认特征向量
    
    def _calculate_glcm(self, image: np.ndarray, distance: int = 1,
                       angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
        """计算灰度共生矩阵"""
        try:
            glcm = np.zeros((256, 256))
            height, width = image.shape
            
            for angle in angles:
                dx = int(distance * np.cos(angle))
                dy = int(distance * np.sin(angle))
                
                for i in range(height):
                    for j in range(width):
                        if 0 <= i + dy < height and 0 <= j + dx < width:
                            i1 = image[i, j]
                            i2 = image[i + dy, j + dx]
                            glcm[i1, i2] += 1
            
            # 归一化
            glcm = glcm / len(angles)
            glcm = glcm / np.sum(glcm)
            
            return glcm
            
        except Exception as e:
            print(f"GLCM计算失败: {str(e)}")
            return np.zeros((256, 256))
    
    def _execute_step(self, step_type: str, image: Image.Image,
                     parameters: Dict) -> Dict:
        """执行处理步骤"""
        try:
            # 根据步骤类型调用相应的处理函数
            if step_type == "lesion_detection":
                from utils.smart_diagnosis import SmartDiagnosisSystem
                system = SmartDiagnosisSystem()
                result = system.analyze_image(image, "lesion_detection")
                if 'error' not in result:
                    return {
                        'success': True,
                        'processed_image': result.get('annotated_image', image),
                        'metrics': {
                            'quality_score': 0.8,
                            'processing_time': 0.1
                        }
                    }
                else:
                    return {'error': result['error']}
            
            elif step_type == "image_enhancement":
                from utils.image_processor import ImageProcessor
                processed = ImageProcessor.auto_enhance(image)
                return {
                    'success': True,
                    'processed_image': processed,
                    'metrics': {
                        'quality_score': 0.8,
                        'processing_time': 0.1
                    }
                }
            
            else:
                return {'error': f'不支持的步骤类型: {step_type}'}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _determine_next_step(self, current_step: WorkflowStep,
                           step_result: Dict) -> Optional[str]:
        """确定下一步骤"""
        try:
            if not current_step.next_steps:
                return None
            
            # 检查条件
            for condition in current_step.conditions:
                if condition['type'] == 'quality_threshold':
                    if step_result['metrics']['quality_score'] < condition['threshold']:
                        return condition['next_step']
                elif condition['type'] == 'processing_time':
                    if step_result['metrics']['processing_time'] > condition['threshold']:
                        return condition['next_step']
            
            # 如果没有满足的条件，返回默认下一步
            return current_step.next_steps[0] if current_step.next_steps else None
            
        except Exception as e:
            print(f"确定下一步失败: {str(e)}")
            return None
    
    def save_workflow(self, workflow: Workflow) -> Dict:
        """保存工作流"""
        try:
            workflow_path = os.path.join(self.storage_path, 'workflows')
            if not os.path.exists(workflow_path):
                os.makedirs(workflow_path)
            
            filename = f"{workflow.name.replace(' ', '_')}.json"
            filepath = os.path.join(workflow_path, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(workflow.to_dict(), f, ensure_ascii=False, indent=2)
            
            return {'success': True, 'filepath': filepath}
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_execution_history(self) -> Dict:
        """分析执行历史"""
        try:
            if not self.execution_history:
                return {'error': '没有执行历史记录'}
            
            analysis = {
                'total_executions': len(self.execution_history),
                'average_steps': 0,
                'step_statistics': defaultdict(lambda: {
                    'count': 0,
                    'average_time': 0,
                    'success_rate': 0
                }),
                'common_patterns': [],
                'performance_trends': {}
            }
            
            # 计算统计信息
            total_steps = 0
            for execution in self.execution_history:
                total_steps += len(execution['steps_executed'])
                
                for step in execution['steps_executed']:
                    stats = analysis['step_statistics'][step['step_type']]
                    stats['count'] += 1
                    
                    # 计算执行时间
                    if 'executed_at' in step:
                        executed_at = datetime.fromisoformat(step['executed_at'])
                        if 'started_at' in execution:
                            started_at = datetime.fromisoformat(execution['started_at'])
                            execution_time = (executed_at - started_at).total_seconds()
                            stats['average_time'] = (stats['average_time'] * (stats['count'] - 1) +
                                                   execution_time) / stats['count']
            
            analysis['average_steps'] = total_steps / len(self.execution_history)
            
            # 识别常见模式
            step_sequences = [
                tuple(step['step_type'] for step in execution['steps_executed'])
                for execution in self.execution_history
            ]
            
            pattern_counter = Counter(step_sequences)
            analysis['common_patterns'] = [
                {
                    'sequence': list(pattern),
                    'frequency': count / len(self.execution_history)
                }
                for pattern, count in pattern_counter.most_common(5)
            ]
            
            # 分析性能趋势
            execution_times = []
            for execution in self.execution_history:
                if 'started_at' in execution and 'completed_at' in execution:
                    start = datetime.fromisoformat(execution['started_at'])
                    end = datetime.fromisoformat(execution['completed_at'])
                    execution_times.append((end - start).total_seconds())
            
            if execution_times:
                analysis['performance_trends'] = {
                    'average_execution_time': sum(execution_times) / len(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times)
                }
            
            return {'success': True, 'analysis': analysis}
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_recommended_workflow(self, image_type: str, analysis_goal: str) -> Dict:
        """获取推荐的工作流
        
        Args:
            image_type: 图像类型，如 'ct', 'mri', 'xray' 等
            analysis_goal: 分析目标，如 'lesion_detection', 'organ_segmentation' 等
        
        Returns:
            Dict: 推荐的工作流配置
        """
        try:
            if analysis_goal == 'lesion_detection':
                workflow = self.workflows.get('lesion_detection_workflow')
                if workflow:
                    # 根据图像类型调整参数
                    if image_type == 'ct':
                        workflow.steps['enhancement'].parameters.update({
                            'window_center': 40,
                            'window_width': 400
                        })
                    elif image_type == 'mri':
                        workflow.steps['preprocessing'].parameters['operations'].append('bias_correction')
                    
                    return {'success': True, 'workflow': workflow.to_dict()}
            
            elif analysis_goal == 'organ_segmentation':
                workflow = self.workflows.get('organ_segmentation_workflow')
                if workflow:
                    # 根据图像类型调整参数
                    if image_type == 'ct':
                        workflow.steps['segmentation'].parameters.update({
                            'hounsfield_threshold': True,
                            'organ_specific_ranges': {
                                'liver': [-40, 180],
                                'kidney': [20, 400],
                                'spleen': [40, 200]
                            }
                        })
                    
                    return {'success': True, 'workflow': workflow.to_dict()}
            
            return {'error': '未找到匹配的工作流'}
            
        except Exception as e:
            return {'error': str(e)}
    
    def optimize_workflow(self, workflow_name: str, execution_history: List[Dict]) -> Dict:
        """根据执行历史优化工作流参数"""
        try:
            if workflow_name not in self.workflows:
                return {'error': '工作流不存在'}
            
            workflow = self.workflows[workflow_name]
            optimized_params = {}
            
            # 分析执行历史
            for execution in execution_history:
                for step in execution['steps_executed']:
                    step_type = step['step_type']
                    if step_type not in optimized_params:
                        optimized_params[step_type] = {
                            'params': [],
                            'quality_scores': []
                        }
                    
                    # 收集参数和质量得分
                    optimized_params[step_type]['params'].append(step['parameters'])
                    if 'metrics' in step and 'quality_score' in step['metrics']:
                        optimized_params[step_type]['quality_scores'].append(
                            step['metrics']['quality_score']
                        )
            
            # 优化每个步骤的参数
            for step_type, data in optimized_params.items():
                if data['params'] and data['quality_scores']:
                    # 找到得分最高的参数组合
                    best_idx = np.argmax(data['quality_scores'])
                    best_params = data['params'][best_idx]
                    
                    # 更新工作流参数
                    for step in workflow.steps.values():
                        if step.step_type == step_type:
                            step.parameters.update(best_params)
            
            # 保存优化后的工作流
            self.save_workflow(workflow)
            
            return {
                'success': True,
                'workflow': workflow.to_dict(),
                'optimizations': optimized_params
            }
            
        except Exception as e:
            return {'error': str(e)} 