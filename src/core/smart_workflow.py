from typing import Dict, Any, List, Callable
import json
import os
from datetime import datetime

class SmartWorkflowSystem:
    """智能工作流系统类"""
    
    def __init__(self):
        self.workflows = {}
        self.current_workflow = None
        self.execution_history = []
        self.workflow_templates = {
            'lesion_detection': {
                'name': '病变检测工作流',
                'steps': [
                    {
                        'name': '图像预处理',
                        'function': 'preprocess_image',
                        'params': {'brightness': 0, 'contrast': 1}
                    },
                    {
                        'name': '病变检测',
                        'function': 'detect_lesions',
                        'params': {}
                    },
                    {
                        'name': 'VLM分析',
                        'function': 'analyze_with_vlm',
                        'params': {'prompt': '检测图像中的异常区域'}
                    }
                ]
            },
            'organ_segmentation': {
                'name': '器官分割工作流',
                'steps': [
                    {
                        'name': '图像增强',
                        'function': 'enhance_image',
                        'params': {'strength': 1.0}
                    },
                    {
                        'name': '器官分割',
                        'function': 'segment_organs',
                        'params': {}
                    }
                ]
            },
            'quality_optimization': {
                'name': '图像质量优化工作流',
                'steps': [
                    {
                        'name': '质量评估',
                        'function': 'assess_quality',
                        'params': {}
                    },
                    {
                        'name': '自动优化',
                        'function': 'auto_optimize',
                        'params': {}
                    }
                ]
            }
        }
    
    def create_workflow(self, name: str, template_id: str = None) -> Dict[str, Any]:
        """创建新的工作流"""
        try:
            if template_id and template_id not in self.workflow_templates:
                return {'success': False, 'error': '模板不存在'}
            
            workflow = {
                'id': f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'name': name,
                'created_at': datetime.now().isoformat(),
                'status': 'created',
                'steps': []
            }
            
            if template_id:
                workflow['template_id'] = template_id
                workflow['steps'] = self.workflow_templates[template_id]['steps']
            
            self.workflows[workflow['id']] = workflow
            return {'success': True, 'workflow': workflow}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def add_step(self, workflow_id: str, step: Dict) -> Dict[str, Any]:
        """添加工作流步骤"""
        try:
            if workflow_id not in self.workflows:
                return {'success': False, 'error': '工作流不存在'}
            
            required_fields = ['name', 'function', 'params']
            if not all(field in step for field in required_fields):
                return {'success': False, 'error': '步骤信息不完整'}
            
            self.workflows[workflow_id]['steps'].append(step)
            return {'success': True, 'step': step}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def execute_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流"""
        try:
            if workflow_id not in self.workflows:
                return {'success': False, 'error': '工作流不存在'}
            
            workflow = self.workflows[workflow_id]
            results = []
            
            for step in workflow['steps']:
                if step['function'] not in context:
                    results.append({
                        'step': step['name'],
                        'success': False,
                        'error': f"函数 {step['function']} 不存在"
                    })
                    continue
                
                try:
                    func = context[step['function']]
                    result = func(**step['params'])
                    results.append({
                        'step': step['name'],
                        'success': True,
                        'result': result
                    })
                except Exception as e:
                    results.append({
                        'step': step['name'],
                        'success': False,
                        'error': str(e)
                    })
            
            execution_record = {
                'workflow_id': workflow_id,
                'executed_at': datetime.now().isoformat(),
                'results': results
            }
            
            self.execution_history.append(execution_record)
            return {'success': True, 'results': results}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_workflow_templates(self) -> Dict[str, Any]:
        """获取工作流模板列表"""
        return {'success': True, 'templates': self.workflow_templates}
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        try:
            if workflow_id not in self.workflows:
                return {'success': False, 'error': '工作流不存在'}
            
            workflow = self.workflows[workflow_id]
            executions = [
                record for record in self.execution_history 
                if record['workflow_id'] == workflow_id
            ]
            
            return {
                'success': True,
                'workflow': workflow,
                'executions': executions
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_workflow(self, workflow_id: str, filepath: str) -> Dict[str, Any]:
        """保存工作流配置"""
        try:
            if workflow_id not in self.workflows:
                return {'success': False, 'error': '工作流不存在'}
            
            workflow = self.workflows[workflow_id]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(workflow, f, ensure_ascii=False, indent=2)
            
            return {'success': True, 'message': '工作流配置已保存'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def load_workflow(self, filepath: str) -> Dict[str, Any]:
        """加载工作流配置"""
        try:
            if not os.path.exists(filepath):
                return {'success': False, 'error': '文件不存在'}
            
            with open(filepath, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            
            if 'id' not in workflow:
                workflow['id'] = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.workflows[workflow['id']] = workflow
            return {'success': True, 'workflow': workflow}
            
        except Exception as e:
            return {'success': False, 'error': str(e)} 