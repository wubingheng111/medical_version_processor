import streamlit as st
import os
from PIL import Image
import numpy as np
from utils.image_processor import ImageProcessor
from utils.vlm_analyzer import VLMAnalyzer
from utils.report_generator import ReportGenerator
import json
from datetime import datetime
import time
import asyncio
import concurrent.futures
from threading import Thread
from docx import Document
from docx.shared import Inches
import sys
import matplotlib.pyplot as plt
import cv2
from utils.smart_diagnosis import SmartDiagnosisSystem
from utils.data_management import CaseDatabase
from utils.quality_control import QualityControl
from utils.smart_workflow import SmartWorkflowSystem

# 设置页面配置
st.set_page_config(
    page_title="🏥 MedVision Pro - 智能医学影像分析平台",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    /* 页面背景 */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d3748 100%);
        background-image: 
            linear-gradient(135deg, rgba(26, 26, 26, 0.97) 0%, rgba(45, 55, 72, 0.97) 100%),
            url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%234299e1' fill-opacity='0.06'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(26, 32, 44, 0.95) 0%, rgba(45, 55, 72, 0.95) 100%);
        backdrop-filter: blur(10px);
    }

    /* 主标题样式 */
    .main-title {
        background: linear-gradient(120deg, rgba(44, 82, 130, 0.92), rgba(26, 54, 93, 0.92));
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
    }

    .main-title h1 {
        color: #fff;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        background: linear-gradient(120deg, #ffffff, #90cdf4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-title p {
        color: #e2e8f0;
        margin-top: 0.5rem;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* 卡片容器样式 */
    .stCard {
        background: linear-gradient(145deg, rgba(45, 55, 72, 0.8), rgba(26, 32, 44, 0.8));
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    /* 按钮样式 */
    .stButton>button {
        background: linear-gradient(45deg, #3182ce, #2c5282);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #4299e1, #3182ce);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* 输入框样式 */
    .stTextInput>div>div>input {
        background: rgba(45, 55, 72, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 8px;
    }

    /* 选择框样式 */
    .stSelectbox>div>div {
        background: rgba(45, 55, 72, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 8px;
    }

    /* 滑块样式 */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #3182ce, #2c5282);
    }

    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-title, .stCard {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

def load_presets():
    """加载预设参数"""
    try:
        with open('presets.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_preset(name, params):
    """保存预设参数"""
    presets = load_presets()
    presets[name] = {
        'params': params,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('presets.json', 'w', encoding='utf-8') as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

# 在main()函数前添加新的功能类
class ProcessingTemplate:
    def __init__(self, name, steps, description):
        self.name = name
        self.steps = steps
        self.description = description
        self.created_at = datetime.now()

class FavoriteManager:
    def __init__(self):
        self.favorites_file = "favorites.json"
        self.favorites = self.load_favorites()
    
    def load_favorites(self):
        try:
            with open(self.favorites_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_favorites(self):
        with open(self.favorites_file, 'w', encoding='utf-8') as f:
            json.dump(self.favorites, f, ensure_ascii=False, indent=2)
    
    def add_favorite(self, name, processing_steps):
        self.favorites[name] = {
            'steps': processing_steps,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save_favorites()
    
    def remove_favorite(self, name):
        if name in self.favorites:
            del self.favorites[name]
            self.save_favorites()

def process_single_image(file, operation_type, current_params):
    """处理单个图像的函数"""
    try:
        # 读取图像并转换为PIL Image对象
        with Image.open(file) as original:
            # 创建副本以避免文件关闭问题
            image_copy = original.copy()
            
            # 确保图像模式正确
            if image_copy.mode not in ['RGB', 'L']:
                image_copy = image_copy.convert('RGB')
            
            if operation_type == "统一处理":
                try:
                    processed = apply_current_processing(image_copy)
                    if not isinstance(processed, Image.Image):
                        processed = Image.fromarray(np.array(processed))
                    return {
                        'success': True,
                        'file_name': file.name,
                        'original': image_copy,
                        'processed': processed,
                        'type': '统一处理'
                    }
                except Exception as e:
                    print(f"统一处理失败: {str(e)}")
                    image_copy.close()
                    return {
                        'success': False,
                        'file_name': file.name,
                        'error': f"统一处理失败: {str(e)}"
                    }
            elif operation_type == "自动优化":
                try:
                    print("开始执行自动优化...")
                    result = ImageProcessor.auto_optimize_image(image_copy)
                    
                    if not result['success']:
                        raise ValueError(result.get('error', '自动优化失败'))
                    
                    processed = result['optimized_image']
                    quality_assessment = result['quality_assessment']
                    
                    # 确保图像是PIL Image对象
                    if not isinstance(processed, Image.Image):
                        processed = Image.fromarray(np.array(processed))
                    
                    # 显示质量评估结果
                    st.markdown("### 质量评估结果")
                    display_quality_assessment(quality_assessment)
                    
                    return {
                        'success': True,
                        'file_name': file.name,
                        'original': image_copy,
                        'processed': processed,
                        'type': '自动优化',
                        'quality_assessment': quality_assessment
                    }
                except Exception as e:
                    print(f"自动优化过程中发生错误: {str(e)}")
                    if 'image_copy' in locals():
                        image_copy.close()
                    return {
                        'success': False,
                        'file_name': file.name,
                        'error': f"自动优化失败: {str(e)}"
                    }
    except Exception as e:
        print(f"图像处理失败: {str(e)}")
        return {
            'success': False,
            'file_name': file.name,
            'error': f"图像处理失败: {str(e)}"
        }

def batch_process_images(files, operation_type, current_params):
    """批量处理图像的函数"""
    results = []
    total_files = len(files)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(
                process_single_image, 
                file, 
                operation_type, 
                current_params
            ): file for file in files
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            completed += 1
            progress = completed / total_files
            
            result = future.result()
            results.append(result)
    
    return results

def save_image_safely(image, save_path, file_name):
    """安全地保存图像文件"""
    try:
        # 使用绝对路径
        abs_save_path = os.path.abspath(save_path)
        print(f"保存目录: {abs_save_path}")
        
        # 确保目录存在
        os.makedirs(abs_save_path, exist_ok=True)
        
        # 构建完整路径
        full_path = os.path.join(abs_save_path, file_name)
        print(f"完整保存路径: {full_path}")
        
        # 验证图像对象
        if not isinstance(image, Image.Image):
            print("转换图像对象为PIL Image")
            image = Image.fromarray(np.array(image))
        
        # 确保图像模式正确
        if image.mode not in ['RGB', 'L']:
            print(f"转换图像模式从 {image.mode} 到 RGB")
            image = image.convert('RGB')
        
        # 创建图像副本
        image_copy = image.copy()
        
        # 保存图像
        image_copy.save(full_path, format='PNG', optimize=True)
        image_copy.close()
        
        # 验证文件是否成功保存
        if os.path.exists(full_path):
            try:
                # 验证保存的文件是否可以打开
                with Image.open(full_path) as verify_img:
                    verify_img.verify()
                return True, full_path
            except Exception as e:
                error_msg = f"文件保存验证失败: {str(e)}"
                print(error_msg)
                if os.path.exists(full_path):
                    os.remove(full_path)
                return False, error_msg
        else:
            return False, "文件未能成功保存"
            
    except Exception as e:
        import traceback
        error_msg = f"保存失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return False, error_msg

def display_batch_results(results_container, processed_results):
    """显示批处理结果"""
    with results_container:
        st.subheader("批处理结果")
        
        # 创建输出目录
        output_base = os.path.join(os.getcwd(), "output")
        os.makedirs(os.path.join(output_base, "originals"), exist_ok=True)
        os.makedirs(os.path.join(output_base, "processed"), exist_ok=True)
        
        # 显示总体统计
        total = len(processed_results)
        successful = sum(1 for r in processed_results if r.get('success', False))
        failed = total - successful
        
        # 使用列布局显示统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总计处理", f"{total} 个文件")
        with col2:
            st.metric("成功", f"{successful} 个")
        with col3:
            st.metric("失败", f"{failed} 个")
        
        # 显示详细结果
        for idx, result in enumerate(processed_results):
            with st.expander(f"文件 {idx + 1}: {result.get('file_name', 'unknown')}", expanded=False):
                if not result.get('success', False):
                    st.error(f"处理失败: {result.get('error', '未知错误')}")
                    continue
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("原始图像")
                    if 'original' in result and result['original'] is not None:
                        try:
                            # 验证图像对象
                            img = result['original']
                            if not isinstance(img, Image.Image):
                                st.error(f"无效的图像对象类型: {type(img)}")
                                continue
                            
                            # 显示图像
                            st.image(img, use_container_width=True)
                            
                            # 保存原始图像
                            file_name = f"original_{os.path.splitext(result['file_name'])[0]}.png"
                            success, message = save_image_safely(
                                img,
                                os.path.join(output_base, "originals"),
                                file_name
                            )
                            if success:
                                st.success(f"原始图像已保存")
                            else:
                                st.error(f"保存失败: {message}")
                            
                        except Exception as e:
                            st.error(f"图像显示错误: {str(e)}")
                
                with col2:
                    if result['type'] in ['统一处理', '自动优化']:
                        st.write(f"处理后图像 ({result['type']})")
                        if 'processed' in result and result['processed'] is not None:
                        
                                # 验证图像对象
                                img = result['processed']
                                if not isinstance(img, Image.Image):
                                    st.error(f"无效的处理后图像对象类型: {type(img)}")
                                    continue
                                
                                # 显示图像
                                st.image(img, use_container_width=True)
                                
                                # 保存处理后图像
                                file_name = f"processed_{os.path.splitext(result['file_name'])[0]}.png"
                                success, message = save_image_safely(
                                    img,
                                    os.path.join(output_base, "processed"),
                                    file_name
                                )
                                if success:
                                    st.success(f"处理后图像已保存")
                                else:
                                    st.error(f"保存失败: {message}")
                                
                                # 显示质量评估结果
                                if result['type'] == '自动优化' and 'quality_assessment' in result:
                                    st.markdown("### 质量评估结果")
                                    display_quality_assessment(result['quality_assessment'])
                                else:
                                    st.info("未获取到质量评估结果")
                    
                    elif result['type'] == '批量分析':
                        if 'analysis' in result:
                            st.write("分析结果")
                            st.write(result['analysis'])

def export_results(processed_results, export_format, export_dir):
    """导出处理结果"""
    try:
        # 确保导出目录是绝对路径
        export_dir = os.path.abspath(export_dir)
        
        # 创建导出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(export_dir, f"批处理结果_{timestamp}")
        os.makedirs(export_path, exist_ok=True)
        
        successful_exports = []
        failed_exports = []
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(processed_results)
        
        for i, result in enumerate(processed_results):
            try:
                if not result.get('success', False):
                    failed_exports.append((result.get('file_name', 'unknown'), "处理失败"))
                    continue
                
                # 更新进度
                progress = (i + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"正在导出: {result.get('file_name', f'文件 {i+1}')} ({i+1}/{total_files})")
                
                # 为每个文件创建单独的目录
                file_name = result.get('file_name', f'result_{i}')
                safe_file_name = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_', '.'))
                file_dir = os.path.join(export_path, os.path.splitext(safe_file_name)[0])
                os.makedirs(file_dir, exist_ok=True)
                
                # 保存原始图像
                if 'original' in result and result['original'] is not None:
                    try:
                        original_path = os.path.join(file_dir, f"original.{export_format.lower()}")
                        # 确保图像对象是有效的
                        img = result['original']
                        if not img.is_loaded:
                            img.load()
                        img.save(original_path)
                    except Exception as e:
                        st.warning(f"保存原始图像失败: {str(e)}")
                
                # 保存处理后的图像或分析结果
                if result['type'] in ['统一处理', '自动优化']:
                    if 'processed' in result and result['processed'] is not None:
                        try:
                            processed_path = os.path.join(file_dir, f"processed.{export_format.lower()}")
                            # 确保图像对象是有效的
                            img = result['processed']
                            if not img.is_loaded:
                                img.load()
                            img.save(processed_path)
                        except Exception as e:
                            st.warning(f"保存处理后图像失败: {str(e)}")
                            continue
                elif result['type'] == '批量分析':
                    if 'analysis' in result:
                        analysis_path = os.path.join(file_dir, 'analysis.txt')
                        try:
                            with open(analysis_path, 'w', encoding='utf-8') as f:
                                if isinstance(result['analysis'], dict):
                                    if result['analysis'].get('success'):
                                        f.write(result['analysis']['analysis'])
                                    else:
                                        f.write(f"分析失败: {result['analysis'].get('error', '未知错误')}")
                                else:
                                    f.write(str(result['analysis']))
                        except Exception as e:
                            st.warning(f"保存分析结果失败: {str(e)}")
                            continue
                
                # 保存处理信息
                info = {
                    'filename': result.get('file_name', 'unknown'),
                    'type': result.get('type', 'unknown'),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 添加图像质量指标
                if result['type'] in ['统一处理', '自动优化'] and 'processed' in result:
                    try:
                        quality_metrics = ImageProcessor.measure_quality(result['processed'])
                        info['quality_metrics'] = quality_metrics
                    except Exception as e:
                        st.warning(f"获取质量指标失败: {str(e)}")
                
                # 保存信息文件
                info_path = os.path.join(file_dir, 'info.json')
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
                
                successful_exports.append(file_name)
                
            except Exception as e:
                failed_exports.append((result.get('file_name', f'result_{i}'), str(e)))
                st.error(f"导出失败: {str(e)}")
                continue
        
        # 清除进度显示
        progress_bar.empty()
        status_text.empty()
        
        # 创建导出报告
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'export_path': export_path,
            'total_files': total_files,
            'successful_exports': successful_exports,
            'failed_exports': len(failed_exports),
            'failed_details': failed_exports
        }
        
        # 保存导出报告
        report_path = os.path.join(export_path, 'export_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return export_path, report
        
    except Exception as e:
        st.error(f"导出过程发生错误: {str(e)}")
        return None, None

def main():
    # 初始化VLM分析器
    vlm_analyzer = VLMAnalyzer()
    
    # 初始化智能诊断系统
    smart_diagnosis = SmartDiagnosisSystem()
    
    # 初始化数据管理系统
    case_db = CaseDatabase('data/medical.db')
    
    # 初始化质量控制系统
    quality_control = QualityControl()
    
    # 初始化智能工作流系统
    workflow_system = SmartWorkflowSystem('data')
    
    # 将系统实例存储在会话状态中
    if 'systems' not in st.session_state:
        st.session_state.systems = {
            'vlm_analyzer': vlm_analyzer,
            'smart_diagnosis': smart_diagnosis,
            'case_db': case_db,
            'quality_control': quality_control,
            'workflow': workflow_system
        }
    
    # 初始化会话状态
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'redo_history' not in st.session_state:
        st.session_state.redo_history = []
    if 'current_params' not in st.session_state:
        st.session_state.current_params = {}
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    
    # 初始化收藏夹管理器
    if 'favorite_manager' not in st.session_state:
        st.session_state.favorite_manager = FavoriteManager()
    
    # 添加批处理模式开关
    batch_mode = st.sidebar.checkbox("批处理模式", value=False)
    
    if batch_mode:
        handle_batch_mode(vlm_analyzer)
    else:
        handle_single_mode(vlm_analyzer)

def handle_batch_mode(vlm_analyzer):
    """处理批处理模式"""
    st.title("批量处理模式")
    
    # 创建两列布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader("选择多个医学图像文件", type=["jpg", "png", "dcm", "nii.gz"], accept_multiple_files=True)
        if uploaded_files:
            st.write(f"已选择 {len(uploaded_files)} 个文件")
    
    with col2:
        # 批处理预设管理
        st.subheader("批处理预设")
        if 'batch_presets' not in st.session_state:
            st.session_state.batch_presets = load_batch_presets()
        
        preset_action = st.radio("预设操作", ["使用预设", "保存新预设"])
        
        if preset_action == "使用预设":
            if st.session_state.batch_presets:
                selected_preset = st.selectbox(
                    "选择预设",
                    list(st.session_state.batch_presets.keys())
                )
                if selected_preset:
                    preset_params = st.session_state.batch_presets[selected_preset]
                    st.write("预设参数：")
                    st.json(preset_params)
                    if st.button("应用预设"):
                        st.session_state.current_params = preset_params['params']
                        st.success("已应用预设参数")
        else:
            new_preset_name = st.text_input("预设名称")
            if st.button("保存当前参数为预设") and new_preset_name:
                save_batch_preset(new_preset_name, st.session_state.current_params)
                st.success(f"已保存预设: {new_preset_name}")
                st.session_state.batch_presets = load_batch_presets()
    
    # 批处理选项
    st.subheader("处理选项")
    batch_operation = st.selectbox("选择批处理操作", ["统一处理", "自动优化"])
    
    # 参数优化设置
    with st.expander("参数优化设置", expanded=False):
        st.write("自动参数优化将分析您的图像特征，推荐最佳处理参数")
        
        if batch_operation == "统一处理":
            # 统一处理的参数优化选项
            col1, col2 = st.columns(2)
            with col1:
                optimize_contrast = st.checkbox("优化对比度", value=True)
                optimize_brightness = st.checkbox("优化亮度", value=True)
            with col2:
                optimize_sharpness = st.checkbox("优化锐度", value=True)
                optimize_noise = st.checkbox("优化降噪", value=True)
            
            if st.button("分析最佳参数"):
                if uploaded_files:
                    with st.spinner("正在分析最佳参数..."):
                        # 使用第一张图片进行参数优化
                        sample_image = Image.open(uploaded_files[0])
                        optimal_params = optimize_processing_params(
                            sample_image,
                            {
                                'contrast': optimize_contrast,
                                'brightness': optimize_brightness,
                                'sharpness': optimize_sharpness,
                                'noise': optimize_noise
                            }
                        )
                        st.session_state.current_params = optimal_params
                        st.success("已找到最佳参数组合！")
                        st.json(optimal_params)
                else:
                    st.warning("请先上传图像文件")
        
        elif batch_operation == "自动优化":
            st.info("自动优化模式将对每张图像进行自适应处理，无需手动设置参数")
            
            # 自动优化的质量控制
            quality_threshold = st.slider(
                "质量提升阈值 (越高要求越严格)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1
            )
            st.session_state.quality_threshold = quality_threshold
    
    # 处理进度显示
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # 创建结果容器
    results_container = st.container()
    
    if uploaded_files and st.button("开始批处理", key="start_batch"):
        progress_text.text("正在准备处理...")
        
        try:
            # 执行批处理
            processed_results = batch_process_images(
                uploaded_files,
                batch_operation,
                st.session_state.get('current_params', {})
            )
            
            # 显示结果
            display_batch_results(results_container, processed_results)
            
        except Exception as e:
            st.error(f"批处理过程中发生错误: {str(e)}")
        finally:
            progress_text.empty()
            progress_bar.empty()

def optimize_processing_params(image, optimization_options):
    """优化处理参数
    
    Args:
        image: 输入图像
        optimization_options: 优化选项字典
    
    Returns:
        optimal_params: 优化后的参数字典
    """
    optimal_params = {}
    
    try:
        # 计算原始图像的质量指标
        original_metrics = ImageProcessor.measure_quality(image)
        
        if optimization_options.get('contrast', False):
            # 对比度优化
            best_contrast = 1.0
            best_contrast_score = 0
            
            for contrast in np.arange(0.8, 1.6, 0.1):
                test_img = ImageProcessor.adjust_contrast_brightness(image, contrast=contrast)
                metrics = ImageProcessor.measure_quality(test_img)
                if metrics['contrast'] > best_contrast_score:
                    best_contrast_score = metrics['contrast']
                    best_contrast = contrast
            
            optimal_params['contrast'] = best_contrast
        
        if optimization_options.get('brightness', False):
            # 亮度优化
            best_brightness = 0
            best_brightness_score = float('inf')
            
            for brightness in range(-50, 51, 10):
                test_img = ImageProcessor.adjust_contrast_brightness(image, brightness=brightness)
                metrics = ImageProcessor.measure_quality(test_img)
                score = abs(128 - np.mean(np.array(test_img)))  # 理想亮度应接近128
                if score < best_brightness_score:
                    best_brightness_score = score
                    best_brightness = brightness
            
            optimal_params['brightness'] = best_brightness
        
        if optimization_options.get('noise', False):
            # 降噪参数优化
            noise_level = original_metrics['noise_level']
            if noise_level > 30:
                optimal_params['denoise'] = {
                    'method': 'gaussian',
                    'sigma': min(noise_level / 10, 2.0)
                }
            elif noise_level > 10:
                optimal_params['denoise'] = {
                    'method': 'bilateral',
                    'sigma_color': 0.1,
                    'sigma_spatial': 15
                }
        
        if optimization_options.get('sharpness', False):
            # 锐度优化
            if original_metrics['sharpness'] < 100:
                optimal_params['sharpen'] = {
                    'amount': min(2.0, 100 / original_metrics['sharpness']),
                    'radius': 1,
                    'threshold': 3
                }
    
    except Exception as e:
        print(f"参数优化过程中发生错误: {str(e)}")
        optimal_params = {}
    
    return optimal_params

def load_batch_presets():
    """加载批处理预设"""
    try:
        with open('batch_presets.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_batch_preset(name, params):
    """保存批处理预设"""
    presets = load_batch_presets()
    presets[name] = {
        'params': params,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('batch_presets.json', 'w', encoding='utf-8') as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

def handle_single_mode(vlm_analyzer):
    """处理单图像模式"""
    # 侧边栏
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: white;'>🏥 MedVision Pro</h2>
                <p style='color: #a8b9cc;'>智能医学影像分析平台</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 自动分析选项
        st.markdown("---")
        auto_analyze = st.checkbox("🔄 处理后自动分析", value=False)
        st.session_state.auto_analyze = auto_analyze
        
        # 功能选择
        st.markdown("### 🛠️ 功能选择")
        operation = st.selectbox(
            "选择处理功能",
            ["图像预处理", "图像增强", "图像分割", "高级处理", "智能诊断",  "质量控制"],
            format_func=lambda x: {
                "图像预处理": "⚙️ 图像预处理",
                "图像增强": "✨ 图像增强",
                "图像分割": "✂️ 图像分割",
                "高级处理": "🔧 高级处理",
                "智能诊断": "🏥 智能诊断",
                "质量控制": "📊 质量控制"
            }[x]
        )
        
        # 撤销/重做按钮
        st.markdown("### 📝 操作历史")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ 撤销", disabled=len(st.session_state.processing_history) == 0):
                if len(st.session_state.processing_history) > 0:
                    last_operation = st.session_state.processing_history.pop()
                    st.session_state.redo_history.append(last_operation)
                    if len(st.session_state.processing_history) > 0:
                        st.session_state.processed_image = st.session_state.processing_history[-1]['image']
                        st.session_state.current_params = st.session_state.processing_history[-1]['params']
                    else:
                        st.session_state.processed_image = None
                        st.session_state.current_params = {}
                    st.session_state.last_analysis = None
                    st.rerun()
        
        with col2:
            if st.button("↪️ 重做", disabled=len(st.session_state.redo_history) == 0):
                if len(st.session_state.redo_history) > 0:
                    last_redo = st.session_state.redo_history.pop()
                    st.session_state.processing_history.append(last_redo)
                    st.session_state.processed_image = last_redo['image']
                    st.session_state.current_params = last_redo['params']
                    st.session_state.last_analysis = None
                    st.rerun()
                    
        # 工作流管理
        st.markdown("### 🔄 工作流")
        if st.button("保存当前工作流"):
            workflow_name = st.text_input("工作流名称")
            if workflow_name:
                workflow = st.session_state.systems['workflow'].create_workflow(
                    name=workflow_name,
                    description="用户自定义工作流",
                    steps=[{
                        'type': op['operation'],
                        'parameters': op['params']
                    } for op in st.session_state.processing_history]
                )
                if 'success' in workflow:
                    st.success("工作流保存成功！")
                else:
                    st.error("工作流保存失败：" + workflow.get('error', '未知错误'))
    
    # 主界面
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #4299e1; margin-bottom: 0.5rem; font-size: 2.5rem;'>🏥 MedVision Pro</h1>
            <p style='color: #a0aec0; font-size: 1.2rem; margin: 0;'>基于视觉语言模型的智能医学影像处理与分析系统</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 文件上传区域
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: #2d3748; border-radius: 10px; border: 2px dashed #4a5568; margin: 20px 0;'>
            <h3 style='color: #90cdf4; margin-bottom: 10px;'>📁 选择医学图像文件</h3>
            <p style='color: #718096; font-size: 0.9em;'>支持格式: JPG, PNG, DICOM, NIfTI</p>
            <p style='color: #718096; font-size: 0.9em;'>拖拽文件到此处或点击选择</p>
        </div>
    """, unsafe_allow_html=True)

    # 自定义上传组件的样式
    st.markdown("""
        <style>
        /* 上传区域样式 */
        .uploadedFile {
            background-color: transparent !important;
            color: #e0e0e0 !important;
            border: 2px dashed #4a5568 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        
        /* 文件上传按钮样式 */
        .stFileUploader > div > div > button {
            background: linear-gradient(45deg, #2c5282, #1a365d) !important;
            color: #e0e0e0 !important;
            border: 1px solid #4a5568 !important;
            border-radius: 5px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        /* 文件上传按钮hover效果 */
        .stFileUploader > div > div > button:hover {
            background: linear-gradient(45deg, #3182ce, #2c5282) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* 文件上传区域hover效果 */
        .stFileUploader > div:hover {
            background-color: #3d4a5f !important;
            border-color: #90cdf4 !important;
        }
        
        /* 移除默认的白色背景 */
        .stFileUploader > div {
            background-color: transparent !important;
            border: none !important;
        }
        
        .stFileUploader > div > div {
            background-color: transparent !important;
        }
        
        /* 上传进度条样式 */
        .stProgress > div > div {
            background-color: #4299e1 !important;
        }
        
        /* 文件名显示样式 */
        .uploadedFileName {
            color: #e0e0e0 !important;
        }
        
        /* 拖放提示文本样式 */
        .stFileUploader > div::before {
            content: "拖放文件到此处" !important;
            color: #718096 !important;
            font-size: 0.9em !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            pointer-events: none !important;
            opacity: 0.7 !important;
        }
        
        /* 上传区域激活状态样式 */
        .stFileUploader > div.drag-active {
            border-color: #90cdf4 !important;
            background-color: rgba(66, 153, 225, 0.1) !important;
        }
        
        /* 上传错误状态样式 */
        .stFileUploader > div.has-error {
            border-color: #fc8181 !important;
        }
        
        /* 上传成功状态样式 */
        .stFileUploader > div.is-success {
            border-color: #48bb78 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 使用自定义样式的文件上传组件
    uploaded_file = st.file_uploader("", type=["jpg", "png", "dcm", "nii.gz"])
    
    if uploaded_file is not None:
        try:
            # 读取并显示原始图像
            image = Image.open(uploaded_file)
            image_copy = image.copy()
            image.close()
            st.session_state.original_image = image_copy
            
            # 创建两列布局用于显示图像
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class='image-container'>
                        <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📷 原始图像</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.image(image_copy, use_container_width=True)
            
            with col2:
                st.markdown("""
                    <div class='image-container'>
                        <h3 style='color: #2c3e50; margin-bottom: 1rem;'>🔄 处理结果</h3>
                    </div>
                """, unsafe_allow_html=True)
                if st.session_state.processed_image is not None:
                    st.image(st.session_state.processed_image, use_container_width=True)
            
            # AI分析部分（居中布局）
            if st.session_state.processed_image is not None:
                st.markdown("""
                    <div class='analysis-section'>
                        <h2 class='analysis-header'>🤖 AI智能分析</h2>
                        <p style='color: #666; text-align: center;'>使用先进的视觉语言模型分析图像处理效果</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # 创建三列布局用于按钮
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("🔍 快速分析", key="ai_analysis_btn", help="对比处理前后的图像变化"):
                        with st.spinner("正在进行智能分析..."):
                            try:
                                # 验证图像是否存在且有效
                                if st.session_state.original_image is None:
                                    st.error("❌ 原始图像不可用，请重新上传图像。")
                                    return
                                    
                                if st.session_state.processed_image is None:
                                    st.error("❌ 处理后的图像不可用，请先进行图像处理。")
                                    return
                                    
                                if not st.session_state.processing_history:
                                    st.error("❌ 未找到处理历史，请先进行图像处理。")
                                    return
                                    
                                # 确保图像是PIL.Image对象
                                original_image = st.session_state.original_image
                                processed_image = st.session_state.processed_image
                                
                                if not isinstance(original_image, Image.Image):
                                    original_image = Image.fromarray(np.array(original_image))
                                if not isinstance(processed_image, Image.Image):
                                    processed_image = Image.fromarray(np.array(processed_image))
                                
                                # 获取最后一次处理操作
                                last_operation = st.session_state.processing_history[-1]['operation']
                                
                                # 调用VLM分析器
                                result = vlm_analyzer.analyze_image_changes(
                                    original_image,
                                    processed_image,
                                    last_operation
                                )
                                
                                if result['success']:
                                    st.session_state.last_analysis = result['analysis']
                                    
                                    # 使用新的卡片样式显示分析结果
                                    st.markdown("""
                                        <div class='analysis-card'>
                                            <h3>📊 分析结果</h3>
                                            <div class='analysis-content'>
                                    """, unsafe_allow_html=True)
                                    
                                    # 流式输出分析结果
                                    analysis_placeholder = st.empty()
                                    full_text = result['analysis']
                                    for i in range(len(full_text) + 1):
                                        analysis_placeholder.markdown(full_text[:i])
                                        if i < len(full_text):
                                            time.sleep(0.01)
                                    
                                    st.markdown("</div></div>", unsafe_allow_html=True)
                                else:
                                    st.error(f"❌ 分析失败: {result['error']}")
                                    st.info("💡 请尝试重新处理图像或联系技术支持。")
                            except Exception as e:
                                st.error(f"❌ 分析过程出错: {str(e)}")
                                st.info("💡 请检查图像格式是否正确，或尝试重新上传。")
                
                with col2:
                    if st.button("💡 优化建议", key="get_suggestions_btn", help="获取图像处理优化建议"):
                        with st.spinner("正在生成优化建议..."):
                            try:
                                # 确保图像是PIL.Image对象
                                processed_image = st.session_state.processed_image
                                if not isinstance(processed_image, Image.Image):
                                    processed_image = Image.fromarray(np.array(processed_image))
                                    
                                result = vlm_analyzer.suggest_improvements(
                                    processed_image,
                                    st.session_state.processing_history[-1]['operation']
                                )
                                
                                if result['success']:
                                    st.markdown("""
                                        <div class='suggestion-card'>
                                            <h3>🎯 优化建议</h3>
                                            <div class='suggestion-content'>
                                    """, unsafe_allow_html=True)
                                    
                                    # 流式输出优化建议
                                    suggestions_placeholder = st.empty()
                                    full_text = result['suggestions']
                                    for i in range(len(full_text) + 1):
                                        suggestions_placeholder.markdown(full_text[:i])
                                        if i < len(full_text):
                                            time.sleep(0.01)
                                    
                                    st.markdown("</div></div>", unsafe_allow_html=True)
                                else:
                                    st.error(f"❌ 生成建议失败: {result['error']}")
                                    st.info("💡 请尝试使用不同的处理参数。")
                            except Exception as e:
                                st.error(f"❌ 生成建议时出错: {str(e)}")
                                st.info("💡 请检查图像格式是否正确，或尝试重新处理。")
                
                with col3:
                    if st.button("📑 生成报告", key="generate_report_btn", help="生成完整分析报告"):
                        with st.spinner("正在生成分析报告..."):
                            try:
                                report_generator = ReportGenerator()
                                report_path = report_generator.save_analysis_report(
                                    st.session_state.original_image,
                                    st.session_state.processed_image,
                                    st.session_state.last_analysis,
                                    st.session_state.processing_history[-1]['operation'],
                                    st.session_state.current_params
                                )
                                st.success(f"✅ 报告已生成: {report_path}")
                                
                                # 添加下载按钮
                                with open(report_path, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ 下载报告",
                                        data=f,
                                        file_name="分析报告.pdf",
                                        mime="application/pdf"
                                    )
                            except Exception as e:
                                st.error(f"❌ 生成报告失败: {str(e)}")
                                st.info("💡 请确保已完成图像分析。")
            
            # 对比滑块
            if st.session_state.processed_image is not None:
                st.subheader("对比视图")
                comparison_value = st.slider("拖动滑块查看对比", 0, 100, 50)
                
                try:
                    # 确保两个图像具有相同的尺寸和通道数
                    orig_arr = np.array(st.session_state.original_image)
                    proc_arr = np.array(st.session_state.processed_image)
                    
                    # 确保两个图像都是3通道的
                    if len(orig_arr.shape) == 2:  # 如果原始图像是灰度图
                        orig_arr = cv2.cvtColor(orig_arr, cv2.COLOR_GRAY2RGB)
                    elif len(orig_arr.shape) == 3 and orig_arr.shape[2] == 4:  # 如果是RGBA图像
                        orig_arr = cv2.cvtColor(orig_arr, cv2.COLOR_RGBA2RGB)
                        
                    if len(proc_arr.shape) == 2:  # 如果处理后的图像是灰度图
                        proc_arr = cv2.cvtColor(proc_arr, cv2.COLOR_GRAY2RGB)
                    elif len(proc_arr.shape) == 3 and proc_arr.shape[2] == 4:  # 如果是RGBA图像
                        proc_arr = cv2.cvtColor(proc_arr, cv2.COLOR_RGBA2RGB)
                    
                    # 确保两个图像具有相同的维度
                    if orig_arr.shape != proc_arr.shape:
                        # 调整processed_image的大小以匹配original_image
                        processed_pil = st.session_state.processed_image.resize(
                            st.session_state.original_image.size, 
                            Image.Resampling.LANCZOS
                        )
                        proc_arr = np.array(processed_pil)
                        if len(proc_arr.shape) == 2:
                            proc_arr = cv2.cvtColor(proc_arr, cv2.COLOR_GRAY2RGB)
                    
                    # 确保是浮点数进行计算
                    orig_arr = orig_arr.astype(float)
                    proc_arr = proc_arr.astype(float)
                    
                    # 根据滑块值混合图像
                    blend = (orig_arr * (100 - comparison_value) + proc_arr * comparison_value) / 100
                    blend = np.clip(blend, 0, 255)  # 确保值在有效范围内
                    
                    st.image(blend.astype(np.uint8), caption=f"对比视图 ({comparison_value}%处理结果)", use_container_width=True)
                except Exception as e:
                    st.error(f"图像混合失败：{str(e)}")
                    st.info("请检查图像格式是否兼容，或尝试重新处理图像。")
            
            # 根据选择的功能显示不同的处理选项
            if operation == "图像预处理":
                preprocessing_options()
            elif operation == "图像增强":
                enhancement_options()
            elif operation == "图像分割":
                segmentation_options()
            elif operation == "高级处理":
                advanced_options()
            elif operation == "智能诊断":
                smart_diagnosis_options()
            elif operation == "远程协作":
                collaboration_options()
            elif operation == "质量控制":
                quality_control_options()
            
            # 处理图像后的自动分析
            if st.session_state.get('auto_analyze', False) and st.session_state.processed_image is not None:
                with st.spinner("正在自动分析处理结果..."):
                    try:
                        # 验证图像是否存在且有效
                        if st.session_state.original_image is None:
                            st.error("原始图像不可用，自动分析已跳过。")
                            return
                        
                        if st.session_state.processed_image is None:
                            st.error("处理后的图像不可用，自动分析已跳过。")
                            return
                        
                        if not st.session_state.processing_history:
                            st.error("未找到处理历史，自动分析已跳过。")
                            return
                        
                        # 确保图像是PIL.Image对象
                        original_image = st.session_state.original_image
                        processed_image = st.session_state.processed_image
                        
                        if not isinstance(original_image, Image.Image):
                            original_image = Image.fromarray(np.array(original_image))
                            st.session_state.original_image = original_image
                        
                        if not isinstance(processed_image, Image.Image):
                            processed_image = Image.fromarray(np.array(processed_image))
                            st.session_state.processed_image = processed_image
                        
                        # 获取最后一次处理操作
                        last_operation = st.session_state.processing_history[-1]['operation']
                        
                        # 调用VLM分析器
                        result = vlm_analyzer.analyze_image_changes(
                            original_image,
                            processed_image,
                            last_operation
                        )
                        
                        if result['success']:
                            st.session_state.last_analysis = result['analysis']
                            with st.expander("📊 自动分析结果", expanded=True):
                                st.markdown(st.session_state.last_analysis)
                        else:
                            st.error(f"自动分析失败: {result['error']}")
                            st.info("请尝试手动分析或调整处理参数。")
                    except Exception as e:
                        st.error(f"自动分析过程出错: {str(e)}")
                        st.info("请检查图像格式是否正确，或尝试重新处理。")
            
        except Exception as e:
            st.error(f"错误：{str(e)}")
    else:
        st.info("请上传一张医学图像")
    
    # 显示处理历史
    if st.session_state.processing_history:
        st.subheader("处理历史")
        for i, step in enumerate(st.session_state.processing_history):
            with st.expander(f"步骤 {i+1}: {step['operation']}", expanded=False):
                st.write(f"参数: {step['params']}")
                if st.button(f"恢复到此步骤", key=f"restore_{i}"):
                    st.session_state.processed_image = step['image']
                    st.session_state.current_params = step['params']
                    # 清除此步骤之后的历史
                    st.session_state.processing_history = st.session_state.processing_history[:i+1]
                    st.session_state.redo_history = []
                    st.session_state.last_analysis = None  # 清除上一次的分析结果
                    st.rerun()

def add_to_history(operation, params=None):
    """添加操作到处理历史"""
    try:
        # 验证处理后的图像
        if st.session_state.processed_image is None:
            raise ValueError("处理后的图像不可用")
        
        # 确保图像是PIL.Image对象
        processed_image = st.session_state.processed_image
        if not isinstance(processed_image, Image.Image):
            if isinstance(processed_image, np.ndarray):
                # 如果是numpy数组，转换为PIL Image
                if len(processed_image.shape) == 2:  # 灰度图
                    processed_image = Image.fromarray(processed_image, 'L')
                elif len(processed_image.shape) == 3:
                    if processed_image.shape[2] == 4:  # RGBA
                        processed_image = Image.fromarray(processed_image, 'RGBA')
                    else:  # RGB
                        processed_image = Image.fromarray(processed_image, 'RGB')
            else:
                raise ValueError("无法将图像转换为PIL Image格式")
        
        # 创建图像副本以避免引用问题
        processed_image_copy = processed_image.copy()
        
        # 添加到历史记录
        st.session_state.processing_history.append({
            'operation': operation,
            'params': params or {},
            'image': processed_image_copy
        })
        
        # 更新当前状态
        st.session_state.processed_image = processed_image_copy
        st.session_state.current_params = params or {}
        st.session_state.last_analysis = None  # 清除上一次的分析结果
        
        # 清除重做历史
        st.session_state.redo_history = []
        
    except Exception as e:
        st.error(f"添加处理历史失败: {str(e)}")
        st.info("请检查图像处理是否成功完成。")

def preprocessing_options():
    """图像预处理选项"""
    st.subheader("图像预处理")
    
    # 加载预设
    presets = load_presets()
    if presets:
        selected_preset = st.selectbox(
            "选择预设参数",
            ["不使用预设"] + list(presets.keys())
        )
        if selected_preset != "不使用预设":
            st.write(f"使用预设: {selected_preset}")
            st.write(f"创建时间: {presets[selected_preset]['created_at']}")
            params = presets[selected_preset]['params']
    
    # 添加自定义CSS确保对齐
    st.markdown("""
        <style>
        .processing-container {
            display: flex;
            align-items: stretch;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .processing-box {
            flex: 1;
            background-color: #2d3748;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #4a5568;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 创建两列布局用于灰度转换和直方图均衡化
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("灰度转换"):
            try:
                if st.session_state.original_image is not None:
                    st.write("正在处理图像...")
                    original_copy = st.session_state.original_image.copy()
                    processed = ImageProcessor.to_grayscale(original_copy)
                    if not isinstance(processed, Image.Image):
                        if isinstance(processed, np.ndarray):
                            processed = Image.fromarray(processed, 'L')
                        else:
                            raise ValueError("灰度转换失败：无法创建有效的图像对象")
                    processed_copy = processed.copy()
                    st.session_state.processed_image = processed_copy
                    add_to_history("灰度转换", {})
                    
                    if st.session_state.get('auto_analyze', False):
                        with st.spinner("正在分析处理结果..."):
                            try:
                                result = vlm_analyzer.analyze_image_changes(
                                    original_copy,
                                    processed_copy,
                                    "灰度转换"
                                )
                                if result['success']:
                                    st.session_state.last_analysis = result['analysis']
                                    st.markdown("### AI分析结果")
                                    st.markdown(result['analysis'])
                                else:
                                    st.error(f"分析失败: {result['error']}")
                            except Exception as e:
                                st.error(f"分析过程出错: {str(e)}")
                    st.success("图像处理完成！")
                    st.rerun()
                else:
                    st.error("请先上传图像")
            except Exception as e:
                st.error(f"灰度转换失败: {str(e)}")
    
    with col2:
        if st.button("直方图均衡化"):
            if st.session_state.original_image is not None:
                try:
                    processed = ImageProcessor.histogram_equalization(st.session_state.original_image)
                    if not isinstance(processed, Image.Image):
                        if isinstance(processed, np.ndarray):
                            processed = Image.fromarray(processed, 'RGB')
                    st.session_state.processed_image = processed.copy()
                    add_to_history("直方图均衡化")
                    st.rerun()
                except Exception as e:
                    st.error(f"直方图均衡化失败: {str(e)}")
            else:
                st.error("请先上传图像")
    
    # 使用container和自定义CSS来确保对齐
    st.markdown('<div class="processing-container">', unsafe_allow_html=True)
    
    # 左侧二值化处理
    st.markdown('<div class="processing-box">', unsafe_allow_html=True)
    threshold = st.slider("二值化阈值", 0, 255, 127)
    if st.button("二值化处理", key="binary_threshold"):
        if st.session_state.original_image is not None:
            try:
                processed = ImageProcessor.binary_threshold(st.session_state.original_image, threshold)
                if not isinstance(processed, Image.Image):
                    if isinstance(processed, np.ndarray):
                        processed = Image.fromarray(processed, 'RGB')
                st.session_state.processed_image = processed.copy()
                add_to_history("二值化处理", {'threshold': threshold})
                st.rerun()
            except Exception as e:
                st.error(f"二值化处理失败: {str(e)}")
        else:
            st.error("请先上传图像")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 右侧噪声消除
    st.markdown('<div class="processing-box">', unsafe_allow_html=True)
    method = st.selectbox("选择滤波方法", ["gaussian", "median", "bilateral"])
    if st.button("噪声消除", key="denoise"):
        if st.session_state.original_image is not None:
            try:
                processed = ImageProcessor.denoise(st.session_state.original_image, method=method)
                if not isinstance(processed, Image.Image):
                    if isinstance(processed, np.ndarray):
                        processed = Image.fromarray(processed, 'RGB')
                st.session_state.processed_image = processed.copy()
                add_to_history("噪声消除", {'method': method})
                st.rerun()
            except Exception as e:
                st.error(f"噪声消除失败: {str(e)}")
        else:
            st.error("请先上传图像")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 保存预设
    st.subheader("保存当前参数为预设")
    preset_name = st.text_input("预设名称")
    if st.button("保存预设") and preset_name and st.session_state.current_params:
        save_preset(preset_name, st.session_state.current_params)
        st.success(f"已保存预设: {preset_name}")

def enhancement_options():
    """图像增强选项"""
    st.subheader("图像增强")
    
    if st.session_state.original_image is not None:
        contrast = st.slider("对比度调整", 0.1, 3.0, 1.0)
        brightness = st.slider("亮度调整", -100, 100, 0)
        
        if st.button("应用增强"):
            processed = ImageProcessor.adjust_contrast_brightness(
                st.session_state.original_image,
                contrast=contrast,
                brightness=brightness
            )
            st.session_state.processed_image = processed
            add_to_history("应用图像增强", {'contrast': contrast, 'brightness': brightness})
            st.rerun()

def segmentation_options():
    """图像分割选项"""
    st.subheader("图像分割")
    
    if st.session_state.original_image is not None:
        method = st.selectbox(
            "选择分割方法",
            ["边缘检测", "区域生长", "分水岭算法"]
        )
        
        if method == "边缘检测":
            edge_method = st.selectbox("选择边缘检测方法", ["canny", "sobel", "laplacian"])
            if st.button("执行边缘检测"):
                processed = ImageProcessor.edge_detection(st.session_state.original_image, method=edge_method)
                st.session_state.processed_image = processed
                add_to_history("执行边缘检测", {'method': edge_method})
                st.rerun()
                
        elif method == "区域生长":
            st.write("点击图像选择种子点")
            # TODO: 实现图像点击功能获取种子点
            if st.button("执行区域生长"):
                # 临时使用图像中心点作为种子点
                h, w = np.array(st.session_state.original_image).shape[:2]
                seed_point = (w//2, h//2)
                processed = ImageProcessor.region_growing(st.session_state.original_image, seed_point)
                st.session_state.processed_image = processed
                add_to_history("执行区域生长分割")
                st.rerun()
                
        elif method == "分水岭算法":
            if st.button("执行分水岭分割"):
                processed = ImageProcessor.watershed_segmentation(np.array(st.session_state.original_image))
                st.session_state.processed_image = Image.fromarray(processed)
                add_to_history("执行分水岭分割")
                st.rerun()

def advanced_options():
    """高级处理选项"""
    st.subheader("高级处理")
    
    if st.session_state.original_image is not None:
        method = st.selectbox(
            "选择处理方法",
            ["形态学处理"]
        )
        
        if method == "形态学处理":
            operation = st.selectbox(
                "选择操作",
                ["dilate", "erode", "open", "close"]
            )
            kernel_size = st.slider("核大小", 3, 15, 5, step=2)
            
            if st.button("执行处理"):
                processed = ImageProcessor.morphological_processing(
                    st.session_state.original_image,
                    operation=operation,
                    kernel_size=kernel_size
                )
                st.session_state.processed_image = processed
                add_to_history("执行形态学处理", {'operation': operation, 'kernel_size': kernel_size})
                st.rerun()

def auto_optimize_image(image):
    """自动优化图像质量"""
    # 1. 自动调整对比度和亮度
    enhanced = ImageProcessor.auto_enhance(image)
    
    # 2. 自动去噪
    denoised = ImageProcessor.auto_denoise(enhanced)
    
    # 3. 自动锐化
    sharpened = ImageProcessor.auto_sharpen(denoised)
    
    return sharpened

def apply_current_processing(image):
    """应用当前的处理参数到图像"""
    if 'current_params' in st.session_state:
        # 应用当前的处理参数
        processed = ImageProcessor.apply_params(image, st.session_state.current_params)
        return processed
    return image

# 在文件末尾添加新的处理函数
def smart_diagnosis_options():
    """智能诊断选项"""
    st.subheader("智能诊断")
    
    if st.session_state.original_image is not None:
        analysis_type = "lesion_detection"  # 直接设置为病变检测
        st.info("当前功能：病变检测")
        
        if st.button("开始分析"):
            with st.spinner("正在进行智能分析..."):
                try:
                    # 确保图像是PIL Image对象
                    if not isinstance(st.session_state.original_image, Image.Image):
                        image = Image.fromarray(np.array(st.session_state.original_image))
                    else:
                        image = st.session_state.original_image
                    
                    # 执行分析
                    analysis_result = st.session_state.systems['smart_diagnosis'].analyze_image(
                        image,
                        analysis_type
                    )
                    
                    if 'error' not in analysis_result:
                        st.success("分析完成！")
                        
                        # 显示分析结果
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("### 原始图像")
                            st.image(image, use_column_width=True)
                        
                        with col2:
                            st.write("### 分析结果")
                            if 'lesions' in analysis_result:
                                # 创建标注后的图像
                                annotated_image = st.session_state.systems['smart_diagnosis'].draw_lesions(
                                    image,
                                    analysis_result['lesions'],
                                    analysis_result['confidence_scores']
                                )
                                st.image(annotated_image, use_column_width=True)
                                
                                # 显示详细信息
                                st.write("检测到的病变：")
                                for i, lesion in enumerate(analysis_result['lesions']):
                                    with st.expander(f"病变 {i+1} (置信度: {analysis_result['confidence_scores'][i]:.2%})"):
                                        st.write(f"- 位置: {lesion['bbox']}")
                                        st.write(f"- 面积: {lesion['area']:.2f}")
                                        st.write(f"- 中心点: {lesion['centroid']}")
                                        if 'circularity' in lesion:
                                            st.write(f"- 圆形度: {lesion['circularity']:.2f}")
                        
                        # 保存分析结果
                        if 'result_image' in analysis_result:
                            st.session_state.processed_image = analysis_result['result_image']
                        elif annotated_image is not None:
                            st.session_state.processed_image = annotated_image
                        add_to_history("智能诊断-病变检测", {'results': analysis_result})
                        
                    else:
                        st.error(f"分析失败: {analysis_result['error']}")
                        
                except Exception as e:
                    st.error(f"处理出错: {str(e)}")
                    st.info("请检查图像格式是否正确")
    else:
        st.info("请先上传图像")

def quality_control_options():
    """质量控制选项"""
    st.subheader("质量控制")
    
    if st.session_state.original_image is not None:
        # 创建选项卡
        tab1, tab2, tab3 = st.tabs(["📊 质量评估", "🔄 优化建议", "📈 历史趋势"])
        
        with tab1:
            # 添加图像选择
            image_choice = st.radio(
                "选择要评估的图像",
                ["原始图像", "处理后的图像"],
                help="选择要进行质量评估的图像"
            )
            
            # 根据选择获取要评估的图像
            target_image = (st.session_state.original_image if image_choice == "原始图像" 
                          else st.session_state.processed_image if st.session_state.processed_image is not None 
                          else st.session_state.original_image)
            
            # 显示选中的图像
            st.image(target_image, caption=f"当前选择: {image_choice}", use_column_width=True)
            
            if st.button("评估图像质量", key="assess_quality"):
                with st.spinner("正在评估图像质量..."):
                    result = st.session_state.systems['quality_control'].assess_image_quality(target_image)
                    
                    if 'success' in result:
                        # 显示总体评分
                        st.markdown(f"""
                            <div style='text-align: center; padding: 1rem;'>
                                <h2>总体评分: {result['overall_score']:.2f}</h2>
                                <h3>质量等级: {result['quality_grade']}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # 使用列布局显示详细指标
                        col1, col2, col3 = st.columns(3)
                        metrics = result['metrics']
                        assessment = result['assessment']
                        
                        # 第一列：基础指标
                        with col1:
                            st.markdown("### 基础指标")
                            st.metric("对比度", f"{metrics['contrast']:.2f}")
                            st.metric("亮度", f"{metrics['brightness']:.2f}")
                            st.metric("清晰度", f"{metrics['sharpness']:.2f}")
                            
                            for metric in ['contrast', 'brightness', 'sharpness']:
                                if assessment[metric]['status'] != 'good':
                                    st.warning(assessment[metric]['recommendation'])
                        
                        # 第二列：噪声指标
                        with col2:
                            st.markdown("### 噪声指标")
                            st.metric("噪声水平", f"{metrics['noise_level']:.2f}")
                            st.metric("信噪比", f"{metrics['snr']:.2f}")
                            st.metric("动态范围", f"{metrics['dynamic_range']:.2f}")
                            
                            for metric in ['noise_level', 'snr', 'dynamic_range']:
                                if assessment[metric]['status'] != 'good':
                                    st.warning(assessment[metric]['recommendation'])
                        
                        # 第三列：高级指标
                        with col3:
                            st.markdown("### 高级指标")
                            st.metric("信息熵", f"{metrics['entropy']:.2f}")
                            st.metric("均匀性", f"{metrics['uniformity']:.2f}")
                            st.metric("边缘密度", f"{metrics['edge_density']:.2f}")
                            
                            for metric in ['entropy', 'uniformity', 'edge_density']:
                                if assessment[metric]['status'] != 'good':
                                    st.warning(assessment[metric]['recommendation'])
                        
                        # 如果是处理后的图像，显示与原始图像的对比
                        if image_choice == "处理后的图像" and st.session_state.original_image is not None:
                            st.markdown("### 📈 与原始图像对比")
                            original_result = st.session_state.systems['quality_control'].assess_image_quality(
                                st.session_state.original_image
                            )
                            
                            if 'success' in original_result:
                                # 计算改进百分比
                                improvement = ((result['overall_score'] - original_result['overall_score']) / 
                                            original_result['overall_score'] * 100)
                                
                                st.metric(
                                    "质量改进",
                                    f"{improvement:+.1f}%",
                                    delta=improvement,
                                    help="相比原始图像的质量提升百分比"
                                )
                                
                                # 显示详细指标对比
                                st.markdown("#### 指标对比")
                                cols = st.columns(3)
                                metrics_to_compare = [
                                    ('contrast', '对比度'),
                                    ('brightness', '亮度'),
                                    ('sharpness', '清晰度'),
                                    ('noise_level', '噪声水平'),
                                    ('snr', '信噪比'),
                                    ('dynamic_range', '动态范围')
                                ]
                                
                                for i, (metric, label) in enumerate(metrics_to_compare):
                                    with cols[i % 3]:
                                        change = ((metrics[metric] - original_result['metrics'][metric]) / 
                                                original_result['metrics'][metric] * 100)
                                        st.metric(
                                            label,
                                            f"{metrics[metric]:.2f}",
                                            f"{change:+.1f}%"
                                        )
                    else:
                        st.error(f"评估失败: {result['error']}")
        
        with tab2:
            # 同样添加图像选择
            image_choice = st.radio(
                "选择要优化的图像",
                ["原始图像", "处理后的图像"],
                help="选择要进行优化的图像",
                key="optimize_image_choice"
            )
            
            target_image = (st.session_state.original_image if image_choice == "原始图像" 
                          else st.session_state.processed_image if st.session_state.processed_image is not None 
                          else st.session_state.original_image)
            
            st.image(target_image, caption=f"当前选择: {image_choice}", use_column_width=True)
            
            if st.button("获取优化建议", key="get_improvements"):
                with st.spinner("正在分析优化建议..."):
                    result = st.session_state.systems['quality_control'].suggest_improvements(target_image)
                    
                    if 'success' in result:
                        # 显示当前质量状态
                        st.markdown(f"""
                            <div style='text-align: center; padding: 1rem;'>
                                <h3>当前质量评分: {result['current_quality_score']:.2f}</h3>
                                <h4>质量等级: {result['quality_grade']}</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # 显示优化建议
                        st.markdown("### 🎯 优化建议")
                        for improvement in result['improvements']:
                            with st.expander(f"改进 {improvement['metric']} (优先级: {improvement['priority']:.2f})"):
                                st.write(f"当前值: {improvement['current_value']:.2f}")
                                st.write(f"目标范围: {improvement['target_range'][0]:.2f} - {improvement['target_range'][1]:.2f}")
                                st.info(improvement['recommendation'])
                        
                        # 显示处理流程
                        st.markdown("### 📝 建议处理流程")
                        
                        # 显示各个步骤的详细信息和单独执行按钮
                        for step in result['processing_pipeline']:
                            with st.expander(f"步骤 {step['step']}: {step['operation']}"):
                                st.write(f"原因: {step['reason']}")
                                st.write("处理参数:")
                                st.json(step['parameters'])
                                
                                # 添加单步执行按钮
                                if st.button(f"执行此步骤", key=f"execute_step_{step['step']}"):
                                    with st.spinner(f"正在执行步骤 {step['step']}..."):
                                        try:
                                            # 获取要处理的图像
                                            input_image = (st.session_state.processed_image 
                                                         if st.session_state.processed_image is not None 
                                                         else target_image).copy()
                                            
                                            # 根据操作类型执行相应的处理
                                            if step['operation'] == 'contrast_adjustment':
                                                processed = ImageProcessor.adjust_contrast_brightness(
                                                    input_image,
                                                    contrast=step['parameters'].get('contrast', 1.0),
                                                    brightness=step['parameters'].get('brightness', 0)
                                                )
                                            elif step['operation'] == 'denoise':
                                                processed = ImageProcessor.denoise(
                                                    input_image,
                                                    method=step['parameters'].get('method', 'gaussian'),
                                                    **{k:v for k,v in step['parameters'].items() if k != 'method'}
                                                )
                                            elif step['operation'] == 'sharpen':
                                                processed = ImageProcessor.sharpen(
                                                    input_image,
                                                    **step['parameters']
                                                )
                                            elif step['operation'] == 'histogram_equalization':
                                                processed = ImageProcessor.histogram_equalization(input_image)
                                            elif step['operation'] == 'auto_enhance':
                                                result = ImageProcessor.auto_optimize_image(input_image)
                                                if not result['success']:
                                                    raise ValueError(result.get('error', '自动优化失败'))
                                                processed = result['optimized_image']
                                            
                                            # 更新处理后的图像
                                            st.session_state.processed_image = processed
                                            add_to_history(f"质量优化-{step['operation']}", step['parameters'])
                                            
                                            # 显示处理结果
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.write("处理前")
                                                st.image(input_image, use_column_width=True)
                                            with col2:
                                                st.write("处理后")
                                                st.image(processed, use_column_width=True)
                                            
                                            st.success(f"步骤 {step['step']} 执行成功！")
                                            
                                            # 评估处理效果
                                            new_result = st.session_state.systems['quality_control'].assess_image_quality(processed)
                                            if 'success' in new_result:
                                                improvement = ((new_result['overall_score'] - result['current_quality_score']) / 
                                                            result['current_quality_score'] * 100)
                                                st.metric(
                                                    "质量改进",
                                                    f"{new_result['overall_score']:.2f}",
                                                    f"{improvement:+.1f}%",
                                                    help="相比处理前的质量提升"
                                                )
                                        except Exception as e:
                                            st.error(f"步骤执行失败: {str(e)}")
                                            st.info("请尝试调整参数或选择其他处理方法")
                    else:
                        st.error(f"获取优化建议失败: {result['error']}")
        
        with tab3:
            if len(st.session_state.systems['quality_control'].quality_history) > 0:
                st.markdown("### 📈 质量趋势分析")
                
                # 创建趋势图
                history = st.session_state.systems['quality_control'].quality_history
                dates = [record['timestamp'] for record in history]
                scores = [record.get('overall_score', 0) for record in history]
                
                # 使用Plotly创建交互式图表
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=scores, mode='lines+markers',
                                       name='总体质量评分'))
                
                fig.update_layout(
                    title='质量评分趋势',
                    xaxis_title='时间',
                    yaxis_title='质量评分',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示统计信息
                st.markdown("### 📊 统计信息")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均评分", f"{np.mean(scores):.2f}")
                with col2:
                    st.metric("最高评分", f"{np.max(scores):.2f}")
                with col3:
                    st.metric("最低评分", f"{np.min(scores):.2f}")
            else:
                st.info("暂无历史记录")
    else:
        st.info("请先上传图像")

def display_quality_assessment(quality_assessment):
    """显示质量评估结果
    
    Args:
        quality_assessment: 包含质量评估结果的字典
    """
    try:
        # 显示评分对比
        orig_score = quality_assessment['original']['overall_score']
        opt_score = quality_assessment['optimized']['overall_score']
        improvement = ((opt_score - orig_score) / orig_score * 100)
        
        # 使用列布局显示评分
        score_cols = st.columns(3)
        with score_cols[0]:
            st.metric("原始评分", f"{orig_score:.2f}")
        with score_cols[1]:
            st.metric("优化后评分", f"{opt_score:.2f}")
        with score_cols[2]:
            st.metric("改进幅度", f"{improvement:+.1f}%")
        
        # 显示详细指标对比
        st.markdown("#### 详细指标对比")
        metric_cols = st.columns(2)
        
        with metric_cols[0]:
            st.markdown("**原始图像指标**")
            for metric, value in quality_assessment['original']['metrics'].items():
                st.write(f"- {metric}: {value:.2f}")
        
        with metric_cols[1]:
            st.markdown("**优化后指标**")
            for metric, value in quality_assessment['optimized']['metrics'].items():
                st.write(f"- {metric}: {value:.2f}")
        
        # 显示验证结果
        st.markdown("#### 优化验证")
        validation = quality_assessment.get('validation', {})
        if validation.get('is_valid', False):
            st.success("✓ 优化效果符合预期")
        else:
            st.warning("! 优化效果需要改进")
            
            if 'warnings' in validation:
                st.markdown("**警告：**")
                for warning in validation['warnings']:
                    st.warning(f"- {warning}")
                    
            if 'suggestions' in validation:
                st.markdown("**建议：**")
                for suggestion in validation['suggestions']:
                    st.info(f"- {suggestion}")
    except Exception as e:
        st.error(f"显示质量评估结果时出错: {str(e)}")
        st.info("请检查质量评估结果的格式是否正确")

if __name__ == "__main__":
    main() 