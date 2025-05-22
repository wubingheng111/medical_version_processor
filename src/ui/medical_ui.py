import streamlit as st
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import os
import io

class MedicalUI:
    """医学影像处理系统UI管理器"""
    
    def __init__(self, systems: Dict):
        """初始化UI管理器
        
        Args:
            systems: 包含各个系统组件的字典
        """
        self.systems = systems
        self._init_session_state()
        self._setup_styles()
        self.current_mode = 'manual'  # 'manual' or 'workflow'
        self.current_image = None
        self.current_results = None
        
        # 初始化VLM模型
        if 'smart_diagnosis' in self.systems:
            self.systems['smart_diagnosis'].initialize_vlm()
    
    def _init_session_state(self):
        """初始化会话状态"""
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
    
    def _setup_styles(self):
        """设置UI样式"""
        st.markdown("""
            <style>
            /* 全局样式 */
            .main {
                padding: 0rem 1rem;
                background-color: #1a1a1a;
                color: #e0e0e0;
            }
            
            /* 标题样式 */
            .main-title {
                color: #ffffff;
                text-align: center;
                padding: 1rem;
                margin-bottom: 2rem;
                background: linear-gradient(120deg, #2c5282, #1a365d);
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            
            /* 卡片样式 */
            .stCard {
                background-color: #2d3748;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                margin-bottom: 1rem;
                border: 1px solid #4a5568;
                color: #e0e0e0;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.markdown("""
                <div style='text-align: center; padding: 1rem;'>
                    <h2 style='color: white;'>MedVision Pro</h2>
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
                ["图像预处理", "图像增强", "图像分割", "高级处理", "智能诊断", "远程协作", "质量控制"],
                format_func=lambda x: {
                    "图像预处理": "⚙️ 图像预处理",
                    "图像增强": "✨ 图像增强",
                    "图像分割": "✂️ 图像分割",
                    "高级处理": "🔧 高级处理",
                    "智能诊断": "🏥 智能诊断",
                    "远程协作": "👥 远程协作",
                    "质量控制": "📊 质量控制"
                }[x]
            )
            
            return operation
    
    def render_main_content(self, operation: str):
        """渲染主要内容"""
        st.markdown("""
            <div class='main-title'>
                <h1>MedVision Pro</h1>
                <p style='font-size: 1.2rem; opacity: 0.8;'>基于视觉语言模型的智能医学影像处理与分析系统</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 文件上传区域
        uploaded_file = self.render_upload_section()
        
        if uploaded_file is not None:
            self.handle_uploaded_file(uploaded_file, operation)
    
    def render_upload_section(self):
        """渲染文件上传区域"""
        st.markdown("""
            <div style='text-align: center; padding: 2rem; background: #2d3748; border-radius: 10px; border: 2px dashed #4a5568; margin: 20px 0;'>
                <h3 style='color: #90cdf4; margin-bottom: 10px;'>📁 选择医学图像文件</h3>
                <p style='color: #718096; font-size: 0.9em;'>支持格式: JPG, PNG, DICOM, NIfTI</p>
                <p style='color: #718096; font-size: 0.9em;'>拖拽文件到此处或点击选择</p>
            </div>
        """, unsafe_allow_html=True)
        
        return st.file_uploader("", type=["jpg", "png", "dcm", "nii.gz"])
    
    def handle_uploaded_file(self, uploaded_file, operation: str):
        """处理上传的文件"""
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
            
            # 根据选择的功能显示不同的处理选项
            if operation == "图像预处理":
                self.render_preprocessing_options()
            elif operation == "图像增强":
                self.render_enhancement_options()
            elif operation == "图像分割":
                self.render_segmentation_options()
            elif operation == "高级处理":
                self.render_advanced_options()
            elif operation == "智能诊断":
                self.render_diagnosis_options()
            elif operation == "远程协作":
                self.render_collaboration_options()
            elif operation == "质量控制":
                self.render_quality_control_options()
            
        except Exception as e:
            st.error(f"错误：{str(e)}")
    
    def render_preprocessing_options(self):
        """渲染图像预处理选项"""
        st.subheader("图像预处理")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("灰度转换"):
                self.handle_grayscale_conversion()
        
        with col2:
            if st.button("直方图均衡化"):
                self.handle_histogram_equalization()
    
    def render_enhancement_options(self):
        """渲染图像增强选项"""
        st.subheader("图像增强")
        
        if st.session_state.original_image is not None:
            contrast = st.slider("对比度调整", 0.1, 3.0, 1.0)
            brightness = st.slider("亮度调整", -100, 100, 0)
            
            if st.button("应用增强"):
                self.handle_image_enhancement(contrast, brightness)
    
    def render_segmentation_options(self):
        """渲染图像分割选项"""
        st.subheader("图像分割")
        
        if st.session_state.original_image is not None:
            method = st.selectbox(
                "选择分割方法",
                ["边缘检测", "区域生长", "分水岭算法"]
            )
            
            if st.button("执行分割"):
                self.handle_image_segmentation(method)
    
    def render_diagnosis_options(self):
        """渲染智能诊断选项"""
        st.subheader("智能诊断")
        
        if st.session_state.original_image is not None:
            # 选择分析模式
            analysis_mode = st.radio(
                "选择分析模式",
                ["手动模式", "工作流模式"],
                format_func=lambda x: {
                    "手动模式": "🔧 手动模式（自定义参数）",
                    "工作流模式": "🔄 工作流模式（自动化处理）"
                }[x]
            )
            
            if analysis_mode == "手动模式":
                # 原有的手动分析功能
                analysis_type = st.selectbox(
                    "选择分析类型",
                    ["lesion_detection", "organ_segmentation", "measurement"],
                    format_func=lambda x: {
                        "lesion_detection": "病变检测",
                        "organ_segmentation": "器官分割",
                        "measurement": "关键指标测量"
                    }[x]
                )
                
                # 添加参数调整滑块
                with st.expander("高级参数设置"):
                    if analysis_type == "lesion_detection":
                        confidence_threshold = st.slider(
                            "置信度阈值",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.4,
                            step=0.1
                        )
                        min_area = st.slider(
                            "最小区域比例",
                            min_value=0.0001,
                            max_value=0.01,
                            value=0.001,
                            format="%.4f"
                        )
                        max_area = st.slider(
                            "最大区域比例",
                            min_value=0.01,
                            max_value=0.5,
                            value=0.3
                        )
                        
                        custom_params = {
                            'confidence_threshold': confidence_threshold,
                            'min_area_ratio': min_area,
                            'max_area_ratio': max_area
                        }
                    else:
                        custom_params = {}
                
                if st.button("开始分析", key="manual_analysis"):
                    self.handle_smart_diagnosis(analysis_type, custom_params)
            
            else:  # 工作流模式
                # 选择图像类型
                image_type = st.selectbox(
                    "选择图像类型",
                    ["ct", "mri", "xray"],
                    format_func=lambda x: {
                        "ct": "CT扫描",
                        "mri": "核磁共振",
                        "xray": "X光片"
                    }[x]
                )
                
                # 选择分析目标
                analysis_goal = st.selectbox(
                    "选择分析目标",
                    ["lesion_detection", "organ_segmentation"],
                    format_func=lambda x: {
                        "lesion_detection": "病变检测",
                        "organ_segmentation": "器官分割"
                    }[x]
                )
                
                # 获取推荐工作流
                workflow = self.systems['smart_workflow'].get_recommended_workflow(
                    image_type=image_type,
                    analysis_goal=analysis_goal
                )
                
                if 'error' not in workflow:
                    # 显示工作流信息
                    with st.expander("查看工作流详情"):
                        st.json(workflow['workflow'])
                    
                    # 执行工作流
                    if st.button("开始分析", key="workflow_analysis"):
                        with st.spinner("正在执行工作流..."):
                            result = self.systems['smart_workflow'].execute_workflow(
                                workflow_name=workflow['workflow']['name'],
                                input_image=st.session_state.original_image
                            )
                            
                            if 'error' not in result:
                                st.success("分析完成！")
                                # 显示最终结果
                                self.display_workflow_results(result['execution_result'])
                            else:
                                st.error(f"分析失败: {result['error']}")
                else:
                    st.error(f"获取工作流失败: {workflow['error']}")
        else:
            st.error("请先上传图像")
    
    def handle_smart_diagnosis(self, analysis_type: str, custom_params: Dict = None):
        """处理智能诊断"""
        try:
            if st.session_state.original_image is not None:
                with st.spinner("正在进行智能分析..."):
                    # 使用VLM进行初步分析
                    vlm_result = self.systems['smart_diagnosis'].vlm_analyzer.analyze_image(
                        st.session_state.original_image,
                        analysis_type
                    )
                    
                    # 结合VLM分析结果和自定义参数进行最终分析
                    result = self.systems['smart_diagnosis'].analyze_image(
                        st.session_state.original_image,
                        analysis_type,
                        custom_params=custom_params,
                        vlm_analysis=vlm_result.get('analysis', {})
                    )
                    
                    if 'error' not in result:
                        st.success("分析完成！")
                        self.display_diagnosis_results(result, analysis_type, vlm_result)
                    else:
                        st.error(f"分析失败: {result['error']}")
            else:
                st.error("请先上传图像")
        except Exception as e:
            st.error(f"智能诊断失败: {str(e)}")
    
    def display_diagnosis_results(self, result: Dict, analysis_type: str, vlm_result: Dict = None):
        """显示诊断结果"""
        # 创建选项卡
        tabs = st.tabs(["检测结果", "VLM分析", "详细信息"])
        
        # 检测结果选项卡
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### 原始图像")
                st.image(st.session_state.original_image, use_column_width=True)
            with col2:
                st.write("### 分析结果")
                if analysis_type == "lesion_detection" and 'lesions' in result:
                    annotated_image = self.systems['smart_diagnosis'].draw_lesions(
                        st.session_state.original_image,
                        result['lesions'],
                        result['confidence_scores']
                    )
                    st.image(annotated_image, use_column_width=True)
        
        # VLM分析选项卡
        with tabs[1]:
            if vlm_result and 'analysis' in vlm_result:
                st.write("### VLM模型分析结果")
                st.write(vlm_result['analysis'])
                if 'suggestions' in vlm_result:
                    st.write("### 建议")
                    for suggestion in vlm_result['suggestions']:
                        st.info(suggestion)
            else:
                st.info("无VLM分析结果")
        
        # 详细信息选项卡
        with tabs[2]:
            if analysis_type == "lesion_detection" and 'lesions' in result:
                st.write("### 检测到的病变")
                for i, lesion in enumerate(result['lesions']):
                    with st.expander(f"病变 {i+1} (置信度: {result['confidence_scores'][i]:.2%})"):
                        st.write(f"- 位置: {lesion['bbox']}")
                        st.write(f"- 面积: {lesion['area']:.2f}")
                        st.write(f"- 中心点: {lesion['centroid']}")
                        if 'circularity' in lesion:
                            st.write(f"- 圆形度: {lesion['circularity']:.2f}")
                        
                        # 显示其他特征
                        if 'features' in lesion:
                            st.write("### 病变特征")
                            for feature, value in lesion['features'].items():
                                st.metric(feature, f"{value:.2f}")
    
    def render_collaboration_options(self):
        """渲染远程协作选项"""
        st.subheader("远程协作")
        
        if st.session_state.original_image is not None:
            if st.button("创建协作会话"):
                self.handle_collaboration_session()
    
    def render_quality_control_options(self):
        """渲染质量控制选项"""
        st.subheader("质量控制")
        
        if st.session_state.original_image is not None:
            if st.button("评估图像质量"):
                self.handle_quality_assessment()
    
    def handle_grayscale_conversion(self):
        """处理灰度转换"""
        try:
            if st.session_state.original_image is not None:
                processed = self.systems['image_processor'].to_grayscale(
                    st.session_state.original_image
                )
                st.session_state.processed_image = processed
                self.add_to_history("灰度转换")
                st.rerun()
            else:
                st.error("请先上传图像")
        except Exception as e:
            st.error(f"灰度转换失败: {str(e)}")
    
    def handle_histogram_equalization(self):
        """处理直方图均衡化"""
        try:
            if st.session_state.original_image is not None:
                processed = self.systems['image_processor'].histogram_equalization(
                    st.session_state.original_image
                )
                st.session_state.processed_image = processed
                self.add_to_history("直方图均衡化")
                st.rerun()
            else:
                st.error("请先上传图像")
        except Exception as e:
            st.error(f"直方图均衡化失败: {str(e)}")
    
    def handle_image_enhancement(self, contrast: float, brightness: int):
        """处理图像增强"""
        try:
            if st.session_state.original_image is not None:
                processed = self.systems['image_processor'].adjust_contrast_brightness(
                    st.session_state.original_image,
                    contrast=contrast,
                    brightness=brightness
                )
                st.session_state.processed_image = processed
                self.add_to_history("图像增强", {
                    'contrast': contrast,
                    'brightness': brightness
                })
                st.rerun()
            else:
                st.error("请先上传图像")
        except Exception as e:
            st.error(f"图像增强失败: {str(e)}")
    
    def handle_image_segmentation(self, method: str):
        """处理图像分割"""
        try:
            if st.session_state.original_image is not None:
                if method == "边缘检测":
                    processed = self.systems['image_processor'].edge_detection(
                        st.session_state.original_image
                    )
                elif method == "区域生长":
                    # 使用图像中心点作为种子点
                    h, w = np.array(st.session_state.original_image).shape[:2]
                    seed_point = (w//2, h//2)
                    processed = self.systems['image_processor'].region_growing(
                        st.session_state.original_image,
                        seed_point
                    )
                elif method == "分水岭算法":
                    processed = self.systems['image_processor'].watershed_segmentation(
                        np.array(st.session_state.original_image)
                    )
                    processed = Image.fromarray(processed)
                
                st.session_state.processed_image = processed
                self.add_to_history("图像分割", {'method': method})
                st.rerun()
            else:
                st.error("请先上传图像")
        except Exception as e:
            st.error(f"图像分割失败: {str(e)}")
    
    def handle_collaboration_session(self):
        """处理协作会话"""
        try:
            session = self.systems['collaboration'].create_session(
                session_name=f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                creator_id="current_user"
            )
            if 'error' not in session:
                st.success(f"会话创建成功！ID: {session['id']}")
                st.session_state.current_session = session
            else:
                st.error(f"创建会话失败: {session['error']}")
        except Exception as e:
            st.error(f"创建协作会话失败: {str(e)}")
    
    def handle_quality_assessment(self):
        """处理质量评估"""
        try:
            if st.session_state.original_image is not None:
                with st.spinner("正在评估图像质量..."):
                    result = self.systems['quality_control'].assess_image_quality(
                        st.session_state.original_image
                    )
                    
                    if 'success' in result:
                        self.display_quality_results(result)
                    else:
                        st.error(f"评估失败: {result['error']}")
            else:
                st.error("请先上传图像")
        except Exception as e:
            st.error(f"质量评估失败: {str(e)}")
    
    def display_workflow_results(self, execution_result: Dict):
        """显示工作流执行结果"""
        # 创建选项卡
        tabs = st.tabs(["处理结果", "中间结果", "执行统计"])
        
        # 处理结果选项卡
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### 原始图像")
                st.image(st.session_state.original_image, use_column_width=True)
            with col2:
                st.write("### 最终结果")
                if execution_result['final_image'] is not None:
                    st.image(execution_result['final_image'], use_column_width=True)
        
        # 中间结果选项卡
        with tabs[1]:
            for step_id, result in execution_result['intermediate_results'].items():
                with st.expander(f"步骤 {step_id}"):
                    # 显示处理后的图像
                    if 'processed_image' in result:
                        st.image(result['processed_image'], use_column_width=True)
                    
                    # 显示处理参数
                    if 'parameters' in result:
                        st.write("使用的参数：")
                        st.json(result['parameters'])
                    
                    # 显示质量指标
                    if 'metrics' in result:
                        st.write("质量指标：")
                        for metric, value in result['metrics'].items():
                            st.metric(metric, f"{value:.2f}")
        
        # 执行统计选项卡
        with tabs[2]:
            # 计算总执行时间
            start_time = datetime.fromisoformat(execution_result['started_at'])
            end_time = datetime.fromisoformat(execution_result['completed_at'])
            execution_time = (end_time - start_time).total_seconds()
            
            # 显示执行统计
            st.metric("总执行时间", f"{execution_time:.2f}秒")
            st.metric("处理步骤数", len(execution_result['steps_executed']))
            
            # 显示每个步骤的执行时间
            st.write("### 步骤执行时间")
            for i, step in enumerate(execution_result['steps_executed']):
                step_time = datetime.fromisoformat(step['executed_at'])
                if i > 0:
                    prev_time = datetime.fromisoformat(
                        execution_result['steps_executed'][i-1]['executed_at']
                    )
                    step_duration = (step_time - prev_time).total_seconds()
                else:
                    step_duration = (step_time - start_time).total_seconds()
                
                st.metric(
                    f"步骤 {step['step_id']} ({step['step_type']})",
                    f"{step_duration:.2f}秒"
                )
    
    def display_quality_results(self, result: Dict):
        """显示质量评估结果"""
        st.write("### 质量评估结果")
        metrics = result['metrics']
        assessment = result['assessment']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("对比度", f"{metrics['contrast']:.2f}")
            st.metric("亮度", f"{metrics['brightness']:.2f}")
            st.metric("清晰度", f"{metrics['sharpness']:.2f}")
        with col2:
            st.metric("噪声水平", f"{metrics['noise_level']:.2f}")
            st.metric("信噪比", f"{metrics['snr']:.2f}")
        
        st.write("### 改进建议")
        for metric, assess in assessment.items():
            if assess['status'] != 'good':
                st.warning(f"- {assess['recommendation']}")
    
    def add_to_history(self, operation: str, params: Dict = None):
        """添加操作到处理历史"""
        try:
            if st.session_state.processed_image is None:
                raise ValueError("处理后的图像不可用")
            
            processed_image = st.session_state.processed_image
            if not isinstance(processed_image, Image.Image):
                if isinstance(processed_image, np.ndarray):
                    if len(processed_image.shape) == 2:
                        processed_image = Image.fromarray(processed_image, 'L')
                    elif len(processed_image.shape) == 3:
                        if processed_image.shape[2] == 4:
                            processed_image = Image.fromarray(processed_image, 'RGBA')
                        else:
                            processed_image = Image.fromarray(processed_image, 'RGB')
                else:
                    raise ValueError("无法将图像转换为PIL Image格式")
            
            processed_image_copy = processed_image.copy()
            
            st.session_state.processing_history.append({
                'operation': operation,
                'params': params or {},
                'image': processed_image_copy
            })
            
            st.session_state.processed_image = processed_image_copy
            st.session_state.current_params = params or {}
            st.session_state.last_analysis = None
            st.session_state.redo_history = []
            
        except Exception as e:
            st.error(f"添加处理历史失败: {str(e)}")
            st.info("请检查图像处理是否成功完成。")

    def setup_sidebar(self):
        """设置侧边栏"""
        st.sidebar.title('医学图像处理系统')
        
        # 模式选择
        self.current_mode = st.sidebar.radio(
            '选择操作模式',
            ['手动模式', '工作流模式'],
            format_func=lambda x: '手动模式' if x == 'manual' else '工作流模式'
        )
        
        st.sidebar.markdown('---')
        
        # 图像上传
        uploaded_file = st.sidebar.file_uploader(
            '上传医学图像',
            type=['png', 'jpg', 'jpeg', 'dicom']
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                self.current_image = np.array(image)
                st.sidebar.success('图像上传成功')
            except Exception as e:
                st.sidebar.error(f'图像加载失败: {str(e)}')
    
    def show_manual_mode(self, image_processor, smart_diagnosis):
        """显示手动模式界面"""
        if self.current_image is None:
            st.info('请先上传图像')
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('原始图像')
            st.image(self.current_image, use_column_width=True)
            
            # 图像处理参数
            st.subheader('图像处理参数')
            brightness = st.slider('亮度', -100, 100, 0)
            contrast = st.slider('对比度', 0.1, 3.0, 1.0)
            detail_enhancement = st.slider('细节增强', 0.0, 2.0, 1.0)
            
            if st.button('应用处理'):
                # 应用图像处理
                result = image_processor.adjust_brightness_contrast(
                    self.current_image,
                    brightness=brightness,
                    contrast=contrast
                )
                if result['success']:
                    enhanced_result = image_processor.enhance_details(
                        result['image'],
                        strength=detail_enhancement
                    )
                    if enhanced_result['success']:
                        self.current_results = enhanced_result
                        st.success('处理完成')
                    else:
                        st.error(enhanced_result['error'])
                else:
                    st.error(result['error'])
        
        with col2:
            st.subheader('处理结果')
            if self.current_results and 'image' in self.current_results:
                st.image(self.current_results['image'], use_column_width=True)
            
            # VLM分析
            st.subheader('VLM分析')
            prompt = st.text_input('输入分析提示', '检测图像中的异常区域')
            
            if st.button('开始分析'):
                with st.spinner('正在分析...'):
                    result = smart_diagnosis.analyze_with_vlm(
                        self.current_image,
                        prompt=prompt
                    )
                    if result['success']:
                        st.success(f"分析完成，相似度: {result['similarity']:.2f}")
                    else:
                        st.error(result['error'])
    
    def show_workflow_mode(self, workflow_system):
        """显示工作流模式界面"""
        if self.current_image is None:
            st.info('请先上传图像')
            return
        
        st.subheader('工作流模式')
        
        # 获取工作流模板
        templates = workflow_system.get_workflow_templates()
        if not templates['success']:
            st.error('获取工作流模板失败')
            return
        
        # 选择工作流模板
        template_names = {
            'lesion_detection': '病变检测工作流',
            'organ_segmentation': '器官分割工作流',
            'quality_optimization': '图像质量优化工作流'
        }
        
        selected_template = st.selectbox(
            '选择工作流模板',
            list(template_names.keys()),
            format_func=lambda x: template_names[x]
        )
        
        if st.button('执行工作流'):
            with st.spinner('正在执行工作流...'):
                # 创建工作流
                workflow = workflow_system.create_workflow(
                    name=template_names[selected_template],
                    template_id=selected_template
                )
                
                if not workflow['success']:
                    st.error('创建工作流失败')
                    return
                
                # 执行工作流
                context = {
                    'image': self.current_image,
                    'workflow_id': workflow['workflow']['id']
                }
                
                result = workflow_system.execute_workflow(
                    workflow['workflow']['id'],
                    context
                )
                
                if result['success']:
                    st.success('工作流执行完成')
                    
                    # 显示结果
                    for step_result in result['results']:
                        with st.expander(f"步骤: {step_result['step']}"):
                            if step_result['success']:
                                if 'image' in step_result['result']:
                                    st.image(
                                        step_result['result']['image'],
                                        use_column_width=True
                                    )
                                st.json(step_result['result'])
                            else:
                                st.error(step_result['error'])
                else:
                    st.error(result['error'])
    
    def show_results(self, results: Dict[str, Any]):
        """显示处理结果"""
        if not results:
            return
        
        st.subheader('处理结果')
        
        # 显示图像结果
        if 'image' in results:
            st.image(results['image'], use_column_width=True)
        
        # 显示检测结果
        if 'lesions' in results:
            st.subheader(f"检测到 {len(results['lesions'])} 个病变区域")
            for i, lesion in enumerate(results['lesions']):
                with st.expander(f"病变区域 {i+1}"):
                    st.json(lesion)
        
        # 显示分析结果
        if 'analysis' in results:
            st.subheader('分析结果')
            st.json(results['analysis'])
    
    def show_error(self, error: str):
        """显示错误信息"""
        st.error(f"错误: {error}")
    
    def show_success(self, message: str):
        """显示成功信息"""
        st.success(message)
    
    def get_current_image(self) -> np.ndarray:
        """获取当前图像"""
        return self.current_image
    
    def get_current_mode(self) -> str:
        """获取当前模式"""
        return self.current_mode 