import os
import base64
from io import BytesIO
import zhipuai
import io
from PIL import Image
from dotenv import load_dotenv
import numpy as np
import json
from datetime import datetime

# 加载环境变量
load_dotenv()

class VLMAnalyzer:
    """使用智谱AI大模型进行图像分析"""
    
    def __init__(self, api_key=None):
        """初始化VLM分析器
        
        Args:
            api_key: 可选的API密钥。如果未提供，将从环境变量ZHIPUAI_API_KEY中读取
        """
        self.api_key = api_key or os.getenv('ZHIPUAI_API_KEY')
        if not self.api_key:
            print("警告: 未设置ZHIPUAI_API_KEY环境变量")
        zhipuai.api_key = self.api_key
        self.client = zhipuai.ZhipuAI(api_key=self.api_key)  # 创建客户端实例
        self.model = "glm-4v-flash"  # 使用免费的GLM-4V-Flash模型
        self.base_prompt = """
        你现在是一位拥有20年临床经验的医学影像专家，专门从事医学图像处理和分析工作。
        请基于专业知识，使用精确的医学术语和定量分析方法评估图像。

        评估标准：
        1. 图像技术质量 (定量指标)
           - 信噪比 (SNR)
           - 对比噪声比 (CNR)
           - 空间分辨率 (lp/mm)
           - 灰度动态范围
           
        2. 临床诊断价值
           - 解剖结构完整性
           - 病理改变的可见度
           - 图像伪影的影响
           - 测量准确性
           
        3. 处理建议
           - 具体的参数调整值
           - 推荐的处理流程
           - 可能的优化方向
           
        请确保分析结果：
        1. 包含具体的数值评估
        2. 使用标准的医学术语
        3. 给出明确的改进建议
        """

    def _encode_image(self, image):
        """将PIL Image转换为base64编码
        
        Args:
            image: PIL Image对象
        
        Returns:
            str: base64编码的图像字符串
        """
        try:
            # 确保图像是RGB或L模式
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # 创建一个新的BytesIO对象
            buffered = BytesIO()
            
            # 保存为PNG格式
            image.save(buffered, format="PNG", optimize=True)
            
            # 获取base64编码
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 关闭BytesIO对象
            buffered.close()
            
            return img_str
            
        except Exception as e:
            print(f"图像编码失败: {str(e)}")
            raise ValueError(f"图像编码失败: {str(e)}")

    def _validate_image(self, image):
        """验证图像是否有效
        
        Args:
            image: PIL Image对象
        
        Returns:
            bool: 图像是否有效
        """
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL Image对象")
        
        try:
            # 验证图像是否可以正常访问
            image.verify()
            return True
        except Exception as e:
            raise ValueError(f"无效的图像: {str(e)}")

    def _resize_image_to_match(self, image, target_size):
        """将图像调整为目标尺寸
        
        Args:
            image: PIL Image对象
            target_size: 目标尺寸元组 (width, height)
            
        Returns:
            PIL Image: 调整后的图像
        """
        if image.size == target_size:
            return image
        return image.resize(target_size, Image.Resampling.LANCZOS)

    def analyze_image_changes(self, original_image, processed_image, operation):
        """分析图像处理前后的变化"""
        try:
            # 添加详细的调试信息
            print(f"开始分析图像变化...")
            print(f"原始图像类型: {type(original_image)}")
            print(f"处理后图像类型: {type(processed_image)}")
            
            # 验证两张图像是否都存在
            if original_image is None:
                print("错误：原始图像为None")
                return {
                    'success': False,
                    'error': '缺少原始图像，无法进行对比分析。请同时提供处理前后的图像。'
                }
            
            if processed_image is None:
                print("错误：处理后图像为None")
                return {
                    'success': False,
                    'error': '缺少处理后的图像，无法进行对比分析。请同时提供处理前后的图像。'
                }
            
            # 确保图像是PIL Image对象
            try:
                if not isinstance(original_image, Image.Image):
                    print(f"转换原始图像为PIL Image，当前类型: {type(original_image)}")
                    if isinstance(original_image, np.ndarray):
                        original_image = Image.fromarray(original_image)
                    else:
                        raise ValueError(f"无法处理的原始图像类型: {type(original_image)}")
                
                if not isinstance(processed_image, Image.Image):
                    print(f"转换处理后图像为PIL Image，当前类型: {type(processed_image)}")
                    if isinstance(processed_image, np.ndarray):
                        processed_image = Image.fromarray(processed_image)
                    else:
                        raise ValueError(f"无法处理的处理后图像类型: {type(processed_image)}")
                
                # 打印图像信息
                print(f"原始图像大小: {original_image.size}, 模式: {original_image.mode}")
                print(f"处理后图像大小: {processed_image.size}, 模式: {processed_image.mode}")
                
                # 确保图像模式一致
                if original_image.mode != processed_image.mode:
                    print(f"图像模式不一致，进行转换...")
                    if original_image.mode == 'L' and processed_image.mode != 'L':
                        processed_image = processed_image.convert('L')
                    elif processed_image.mode == 'L' and original_image.mode != 'L':
                        original_image = original_image.convert('L')
                    else:
                        original_image = original_image.convert('RGB')
                        processed_image = processed_image.convert('RGB')
                
                # 验证图像是否可以访问
                original_image.load()
                processed_image.load()
                
            except Exception as e:
                print(f"图像验证/转换失败: {str(e)}")
                return {
                    'success': False,
                    'error': f'图像验证失败: {str(e)}\n请确保提供的是有效的医学图像。'
                }
            
            # 处理图像尺寸不一致的情况
            if original_image.size != processed_image.size:
                print("检测到图像尺寸不一致，进行调整...")
                # 记录原始尺寸信息
                original_size = original_image.size
                processed_size = processed_image.size
                
                # 选择较大的尺寸作为目标尺寸
                target_size = (
                    max(original_size[0], processed_size[0]),
                    max(original_size[1], processed_size[1])
                )
                
                print(f"调整图像至目标尺寸: {target_size}")
                # 调整图像尺寸
                original_image = self._resize_image_to_match(original_image, target_size)
                processed_image = self._resize_image_to_match(processed_image, target_size)
                
                # 在提示词中添加尺寸变化信息
                size_change_info = f"""
                注意：检测到图像尺寸发生变化
                - 原始图像尺寸: {original_size}
                - 处理后图像尺寸: {processed_size}
                - 为便于分析，已将两张图像调整至相同尺寸: {target_size}
                请在分析中考虑尺寸变化对图像质量的潜在影响。
                """
            else:
                size_change_info = ""
            
            print("准备进行图像编码...")
            # 将两张图片转换为base64
            try:
                original_b64 = self._encode_image(original_image)
                processed_b64 = self._encode_image(processed_image)
                print("图像编码完成")
            except Exception as e:
                print(f"图像编码失败: {str(e)}")
                return {
                    'success': False,
                    'error': f"图像编码失败: {str(e)}"
                }
            
            prompt = f"""
            {self.base_prompt}
            
            请作为医学影像专家，对比分析处理前后的图像变化：

            图像处理操作: {operation}
            {size_change_info}
            
            请严格按照以下方面进行分析：

            1. 具体变化分析（定量+定性）
               - 图像整体亮度变化（提供具体数值区间）
               - 对比度变化百分比
               - 细节区域的信息保留程度
               - 是否出现新的伪影或噪声
               - 尺寸变化对图像质量的影响（如果有）
            
            2. 医学诊断影响
               - 对关键解剖结构识别的影响
               - 对病变区域显示的改善程度
               - 是否有信息损失风险
               - 对定量测量的影响
               - 尺寸变化对诊断的影响（如果有）
            
            3. 处理效果评估
               - 处理参数是否合适（过度/不足）
               - 建议的参数调整范围
               - 是否需要配合其他处理方法
               - 临床应用注意事项
               - 尺寸调整建议（如果需要）

            请用专业、具体的语言描述，避免模糊表述，给出明确的数值和建议。
            对于每个变化，请明确说明是改善(+)还是劣化(-)。
            
            注意：这是一次{operation}操作的前后对比分析。请确保分析内容针对性强，避免泛泛而谈。
            """
            
            print("开始调用API...")
            try:
                # 调用API
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{original_b64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{processed_b64}"
                            }
                        }
                    ]
                }]
                
                # 打印API请求内容（不包含图像数据）
                print("API请求内容:")
                print(f"Model: {self.model}")
                print(f"Message count: {len(messages)}")
                print(f"Content items: {len(messages[0]['content'])}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                
                print("API调用成功，提取分析结果...")
                # 提取分析结果
                
                analysis = response.choices[0].message.content
                
                # 验证分析结果是否为空
                if not analysis or analysis.strip() == "":
                    print("API返回空结果")
                    return {
                        'success': False,
                        'error': 'AI分析器返回了空结果，请重试。'
                    }
                
                print("分析完成")
                return {'success': True, 'analysis': analysis}
                
            except Exception as e:
                error_msg = str(e)
                print(f"API调用失败: {error_msg}")
                if "1113" in error_msg:
                    return {
                        'success': False, 
                        'error': "API调用失败，请检查：\n1. API密钥是否正确\n2. 是否超出免费额度\n3. 网络连接是否正常"
                    }
                return {'success': False, 'error': f"分析失败: {error_msg}"}
        except Exception as e:
            print(f"发生未知错误: {str(e)}")
            return {
                'success': False,
                'error': f"未知错误: {str(e)}"
            }

    def suggest_improvements(self, processed_image, operation):
        """根据处理后的图像提供改进建议"""
        try:
            # 验证图像
            try:
                if not isinstance(processed_image, Image.Image):
                    print(f"转换处理后图像为PIL Image，当前类型: {type(processed_image)}")
                    if isinstance(processed_image, np.ndarray):
                        processed_image = Image.fromarray(processed_image)
                    else:
                        raise ValueError(f"无法处理的图像类型: {type(processed_image)}")
                
                # 打印图像信息
                print(f"图像大小: {processed_image.size}, 模式: {processed_image.mode}")
                
                # 确保图像模式正确
                if processed_image.mode not in ['RGB', 'L']:
                    print(f"转换图像模式从 {processed_image.mode} 到 RGB")
                    processed_image = processed_image.convert('RGB')
                
                # 验证图像是否可以访问
                processed_image.load()
                
            except Exception as e:
                print(f"图像验证/转换失败: {str(e)}")
                return {
                    'success': False,
                    'error': f'图像验证失败: {str(e)}\n请确保提供的是有效的医学图像。'
                }
            
            prompt = f"""
            作为资深医学影像专家，请对这张经过{operation}处理的医学图像进行专业分析和优化建议：

            请严格按照以下方面进行分析：

            1. 图像质量评估（请提供具体数值）
               - 信噪比 (SNR): [数值范围]
               - 对比噪声比 (CNR): [数值范围]
               - 空间分辨率: [数值] lp/mm
               - 动态范围: [数值范围]
               - 图像清晰度评分: [1-10分]
               
            2. 临床应用评估
               - 解剖结构显示质量 [A-优秀/B-良好/C-一般/D-不足]
               - 病变区域识别难度 [定量评分]
               - 测量准确性影响 [误差估计]
               - 诊断可用性评分 [1-10分]
               
            3. 具体优化建议
               - 推荐的预处理步骤
               - 建议的处理参数范围
               - 后处理优化方法
               - 特殊区域处理技术
               
            4. 质量改进方案
               - 主要问题及解决方案
               - 优化参数具体数值
               - 建议的处理流程
               - 预期改善效果
               
            请提供详细的定量分析和具体可执行的优化方案。
            对每项建议，请说明：
            1. 预期改善效果（量化指标）
            2. 可能的风险和注意事项
            3. 具体的操作步骤和参数
            
            注意：这是一张经过{operation}处理的医学图像，请针对性地分析处理效果和优化空间。
            """
            
            print("准备进行图像编码...")
            try:
                # 将图片转换为base64
                processed_b64 = self._encode_image(processed_image)
                print("图像编码完成")
            except Exception as e:
                print(f"图像编码失败: {str(e)}")
                return {
                    'success': False,
                    'error': f"图像编码失败: {str(e)}"
                }
            
            print("开始调用API...")
            try:
                # 调用API
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{processed_b64}"
                            }
                        }
                    ]
                }]
                
                # 打印API请求内容（不包含图像数据）
                print("API请求内容:")
                print(f"Model: {self.model}")
                print(f"Message count: {len(messages)}")
                print(f"Content items: {len(messages[0]['content'])}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                
                print("API调用成功，提取分析结果...")
                # 提取建议
                suggestions = response.choices[0].message.content
                
                # 验证分析结果是否为空
                if not suggestions or suggestions.strip() == "":
                    print("API返回空结果")
                    return {
                        'success': False,
                        'error': 'AI分析器返回了空结果，请重试。'
                    }
                
                print("分析完成")
                return {'success': True, 'suggestions': suggestions}
                
            except Exception as e:
                error_msg = str(e)
                print(f"API调用失败: {error_msg}")
                if "1113" in error_msg:
                    return {
                        'success': False, 
                        'error': "API调用失败，请检查：\n1. API密钥是否正确\n2. 是否超出免费额度\n3. 网络连接是否正常"
                    }
                return {'success': False, 'error': f"生成建议失败: {error_msg}"}
        except Exception as e:
            print(f"发生未知错误: {str(e)}")
            return {
                'success': False,
                'error': f"未知错误: {str(e)}"
            }

    def _call_zhipuai(self, prompt, image):
        """调用智谱AI API进行图像分析
        
        Args:
            prompt: 提示文本
            image: PIL Image对象
        
        Returns:
            str: 分析结果
        """
        try:
            # 将图片转换为base64
            image_b64 = self._encode_image(image)
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64]
                }]
            )
            
            # 提取分析结果
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            if "1113" in error_msg:
                raise Exception("API调用失败，请检查：\n1. API密钥是否正确\n2. 是否超出免费额度\n3. 网络连接是否正常")
            raise Exception(f"分析失败: {error_msg}")

    def analyze_single_image(self, image):
        """分析单张图像的质量和特征"""
        try:
            # 验证图像是否存在
            if image is None:
                return {
                    'success': False,
                    'error': '未提供图像，无法进行分析。请提供有效的医学图像。'
                }
            
            # 验证图像
            try:
                self._validate_image(image)
            except ValueError as e:
                return {
                    'success': False,
                    'error': f'图像验证失败: {str(e)}\n请确保提供的是有效的医学图像。'
                }
            
            prompt = """
            请对这张医学图像进行专业的定量和定性分析：
            
            1. 图像质量定量分析
               - 信噪比(SNR)测量值
               - 对比噪声比(CNR)测量值
               - 空间分辨率评估(lp/mm)
               - 灰度/直方图分布特征
               - 边缘锐度评估(MTF)
            
            2. 解剖结构评估 [请使用A/B/C/D分级标准]
               - 组织边界清晰度
               - 软组织对比度
               - 骨骼结构显示
               - 血管走行显示
               - 病变区域特征
            
            3. 图像质量问题识别
               - 噪声类型和程度
               - 伪影位置和性质
               - 运动模糊评估
               - 散射线影响
               - 量子噪声水平
            
            4. 处理建议
               - 推荐的预处理方法
               - 建议的增强参数
               - 降噪方案选择
               - 特殊区域处理技术
               - 后处理优化步骤
            
            请确保：
            1. 提供具体的数值评估结果
            2. 使用标准的医学图像评价术语
            3. 给出明确的处理参数范围
            4. 说明每个建议的预期效果
            
            注意：请针对当前图像的具体特征进行分析，避免泛泛而谈。
            如果某些特征在图像中不明显，请明确指出。
            """
            
            try:
                # 将图片转换为base64
                image_b64 = self._encode_image(image)
                
                # 调用API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64]
                    }]
                )
                
                # 提取分析结果
                result = response.choices[0].message.content
                
                # 验证分析结果是否为空
                if not result or result.strip() == "":
                    return {
                        'success': False,
                        'error': 'AI分析器返回了空结果，请重试。'
                    }
                
                return {
                    'success': True,
                    'analysis': result
                }
            except Exception as e:
                error_msg = str(e)
                if "1113" in error_msg:
                    return {
                        'success': False,
                        'error': "API调用失败，请检查：\n1. API密钥是否正确\n2. 是否超出免费额度\n3. 网络连接是否正常"
                    }
                return {
                    'success': False,
                    'error': f"分析失败: {error_msg}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"未知错误: {str(e)}"
            }

    def analyze_with_prompt(self, image_base64, prompt):
        """使用智谱AI GLM-4V-Flash模型分析图像
        
        Args:
            image_base64: base64编码的图像
            prompt: 分析提示词
            
        Returns:
            dict: 分析结果
        """
        try:
            if not self.api_key:
                return {
                    'success': False,
                    'error': '未配置API密钥',
                    'analysis': None
                }
            
            response = zhipuai.model_api.invoke(
                model="glm-4v-flash",
                prompt=prompt,
                image=[image_base64]
            )
            
            if response['code'] == 200:
                return {
                    'success': True,
                    'analysis': response['data']['choices'][0]['content'],
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'error': f"API调用失败: {response['msg']}",
                    'analysis': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis': None
            }
            
    def get_medical_analysis(self, image_base64):
        """专门用于医学图像分析的方法
        
        Args:
            image_base64: base64编码的图像
            
        Returns:
            dict: 分析结果
        """
        prompt = """请仔细分析这张医学图像，并提供以下信息：
        1. 是否存在明显的病变区域？如果有，请详细描述每个病变区域的：
           - 位置（使用图像坐标）
           - 大小和形状特征
           - 颜色和纹理特征
           - 边界清晰度
        
        2. 对每个病变区域进行严重程度评估（0-1分值），并说明评估依据
        
        3. 是否存在以下特征：
           - 不规则边缘
           - 异常组织密度
           - 周围组织改变
           - 钙化或其他特殊征象
        
        4. 建议进一步关注和检查的重点
        
        请以结构化的方式输出分析结果，便于后续程序处理。
        """
        
        return self.analyze_with_prompt(image_base64, prompt)
        
    def save_analysis_result(self, result, save_dir="analysis_results"):
        """保存分析结果
        
        Args:
            result: 分析结果字典
            save_dir: 保存目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2) 