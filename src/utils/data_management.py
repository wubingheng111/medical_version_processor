import json
from datetime import datetime
import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from PIL import Image
import sqlite3
import shutil

class CaseDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    def initialize_database(self):
        """初始化数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # 创建病例表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT,
                    metadata TEXT
                )
            ''')
            
            # 创建图像表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    case_id TEXT,
                    file_path TEXT,
                    image_type TEXT,
                    created_at TEXT,
                    metadata TEXT,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id)
                )
            ''')
            
            # 创建处理记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_records (
                    record_id TEXT PRIMARY KEY,
                    image_id TEXT,
                    process_type TEXT,
                    parameters TEXT,
                    created_at TEXT,
                    result_path TEXT,
                    FOREIGN KEY (image_id) REFERENCES images (image_id)
                )
            ''')
            
            self.conn.commit()
            
        except Exception as e:
            print(f"数据库初始化失败: {str(e)}")
    
    def add_case(self, case_data: Dict) -> Dict:
        """添加新病例"""
        try:
            cursor = self.conn.cursor()
            
            case_id = case_data.get('case_id', str(uuid.uuid4()))
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO cases (case_id, patient_id, created_at, updated_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                case_id,
                case_data.get('patient_id'),
                now,
                now,
                'active',
                json.dumps(case_data.get('metadata', {}))
            ))
            
            self.conn.commit()
            return {'success': True, 'case_id': case_id}
            
        except Exception as e:
            return {'error': str(e)}
    
    def add_image(self, case_id: str, image_path: str, image_type: str, 
                 metadata: Dict) -> Dict:
        """添加图像到病例"""
        try:
            cursor = self.conn.cursor()
            
            image_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # 复制图像到存储目录
            storage_path = os.path.join('data', 'images', case_id)
            os.makedirs(storage_path, exist_ok=True)
            
            new_image_path = os.path.join(storage_path, f"{image_id}{os.path.splitext(image_path)[1]}")
            shutil.copy2(image_path, new_image_path)
            
            cursor.execute('''
                INSERT INTO images (image_id, case_id, file_path, image_type, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                image_id,
                case_id,
                new_image_path,
                image_type,
                now,
                json.dumps(metadata)
            ))
            
            self.conn.commit()
            return {'success': True, 'image_id': image_id}
            
        except Exception as e:
            return {'error': str(e)}
    
    def add_processing_record(self, image_id: str, process_type: str,
                            parameters: Dict, result_path: str) -> Dict:
        """添加处理记录"""
        try:
            cursor = self.conn.cursor()
            
            record_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO processing_records 
                (record_id, image_id, process_type, parameters, created_at, result_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                record_id,
                image_id,
                process_type,
                json.dumps(parameters),
                now,
                result_path
            ))
            
            self.conn.commit()
            return {'success': True, 'record_id': record_id}
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_case_history(self, case_id: str) -> Dict:
        """获取病例历史"""
        try:
            cursor = self.conn.cursor()
            
            # 获取病例信息
            cursor.execute('SELECT * FROM cases WHERE case_id = ?', (case_id,))
            case_data = cursor.fetchone()
            
            if not case_data:
                return {'error': '病例不存在'}
            
            # 获取相关图像
            cursor.execute('SELECT * FROM images WHERE case_id = ?', (case_id,))
            images = cursor.fetchall()
            
            # 获取处理记录
            processing_records = []
            for image in images:
                cursor.execute('''
                    SELECT * FROM processing_records WHERE image_id = ?
                ''', (image[0],))
                records = cursor.fetchall()
                processing_records.extend(records)
            
            return {
                'success': True,
                'case_data': case_data,
                'images': images,
                'processing_records': processing_records
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def export_case_data(self, case_id: str, export_path: str) -> Dict:
        """导出病例数据"""
        try:
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            
            case_history = self.get_case_history(case_id)
            if 'error' in case_history:
                return case_history
            
            # 导出数据到JSON文件
            export_data = {
                'case_info': case_history['case_data'],
                'images': [],
                'processing_records': []
            }
            
            # 复制图像文件
            images_dir = os.path.join(export_path, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            for image in case_history['images']:
                image_id = image[0]
                original_path = image[2]
                new_path = os.path.join(images_dir, f"{image_id}{os.path.splitext(original_path)[1]}")
                
                shutil.copy2(original_path, new_path)
                
                image_data = {
                    'image_id': image_id,
                    'type': image[3],
                    'metadata': json.loads(image[5]),
                    'exported_path': new_path
                }
                export_data['images'].append(image_data)
            
            # 添加处理记录
            for record in case_history['processing_records']:
                record_data = {
                    'record_id': record[0],
                    'image_id': record[1],
                    'process_type': record[2],
                    'parameters': json.loads(record[3]),
                    'created_at': record[4],
                    'result_path': record[5]
                }
                export_data['processing_records'].append(record_data)
            
            # 保存导出数据
            export_file = os.path.join(export_path, f"case_{case_id}_export.json")
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return {'success': True, 'export_path': export_path}
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_statistics(self, start_date: str = None, end_date: str = None) -> Dict:
        """生成统计报告"""
        try:
            cursor = self.conn.cursor()
            
            # 构建查询条件
            date_condition = ""
            params = []
            if start_date:
                date_condition += " AND created_at >= ?"
                params.append(start_date)
            if end_date:
                date_condition += " AND created_at <= ?"
                params.append(end_date)
            
            # 统计病例数量
            cursor.execute(f'''
                SELECT COUNT(*) FROM cases 
                WHERE 1=1 {date_condition}
            ''', params)
            total_cases = cursor.fetchone()[0]
            
            # 统计图像数量
            cursor.execute(f'''
                SELECT COUNT(*), image_type 
                FROM images 
                WHERE 1=1 {date_condition}
                GROUP BY image_type
            ''', params)
            image_stats = cursor.fetchall()
            
            # 统计处理记录
            cursor.execute(f'''
                SELECT COUNT(*), process_type 
                FROM processing_records 
                WHERE 1=1 {date_condition}
                GROUP BY process_type
            ''', params)
            processing_stats = cursor.fetchall()
            
            # 生成统计报告
            stats = {
                'total_cases': total_cases,
                'image_statistics': {
                    'total': sum(count for count, _ in image_stats),
                    'by_type': {type_: count for count, type_ in image_stats}
                },
                'processing_statistics': {
                    'total': sum(count for count, _ in processing_stats),
                    'by_type': {type_: count for count, type_ in processing_stats}
                },
                'generated_at': datetime.now().isoformat()
            }
            
            return {'success': True, 'statistics': stats}
            
        except Exception as e:
            return {'error': str(e)}
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

class ProcessingTemplate:
    def __init__(self, name: str, description: str, steps: List[Dict]):
        self.name = name
        self.description = description
        self.steps = steps
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'name': self.name,
            'description': self.description,
            'steps': self.steps,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessingTemplate':
        """从字典创建模板"""
        return cls(
            name=data['name'],
            description=data['description'],
            steps=data['steps']
        )

class TemplateManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.templates = {}
        self.load_templates()
    
    def load_templates(self):
        """加载所有模板"""
        try:
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path)
                return
            
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    with open(os.path.join(self.storage_path, filename), 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        template = ProcessingTemplate.from_dict(template_data)
                        self.templates[template.name] = template
                        
        except Exception as e:
            print(f"加载模板失败: {str(e)}")
    
    def save_template(self, template: ProcessingTemplate) -> Dict:
        """保存处理模板"""
        try:
            filename = f"{template.name.replace(' ', '_')}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(template.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.templates[template.name] = template
            return {'success': True, 'filepath': filepath}
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_template(self, name: str) -> Optional[ProcessingTemplate]:
        """获取模板"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self.templates.keys())
    
    def delete_template(self, name: str) -> Dict:
        """删除模板"""
        try:
            if name not in self.templates:
                return {'error': '模板不存在'}
            
            filename = f"{name.replace(' ', '_')}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            del self.templates[name]
            return {'success': True}
            
        except Exception as e:
            return {'error': str(e)} 