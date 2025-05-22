import cv2
import numpy as np
from PIL import Image, ImageEnhance
import SimpleITK as sitk
from skimage import exposure, filters, morphology, segmentation, feature
from scipy.ndimage import gaussian_filter
from skimage.restoration import estimate_sigma, denoise_wavelet, denoise_bilateral
from scipy.ndimage import uniform_filter
from .quality_control import QualityControl

class ImageProcessor:
    """图像处理工具类"""
    
    @staticmethod
    def _ensure_3channel(image):
        """确保图像是3通道的"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:  # 如果是单通道图像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # 如果是RGBA图像
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image

    @staticmethod
    def _to_pil_image(image):
        """将numpy数组转换为PIL Image"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # 如果是单通道图像
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # 如果是RGBA图像
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image

    @staticmethod
    def to_grayscale(image):
        """将图像转换为灰度图
        
        Args:
            image: PIL Image对象
            
        Returns:
            PIL Image: 灰度图像
        """
        try:
            # 确保输入是PIL Image对象
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("输入必须是PIL Image对象或numpy数组")
            
            # 如果已经是灰度图，直接返回副本
            if image.mode == 'L':
                return image.copy()
            
            # 转换为灰度图
            gray_image = image.convert('L')
            
            # 验证转换结果
            if not isinstance(gray_image, Image.Image):
                raise ValueError("灰度转换失败")
            
            if gray_image.mode != 'L':
                raise ValueError(f"转换后的图像模式不正确: {gray_image.mode}")
            
            return gray_image
            
        except Exception as e:
            raise ValueError(f"灰度转换失败: {str(e)}")
    
    @staticmethod
    def binary_threshold(image, threshold=127):
        """二值化处理"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return ImageProcessor._to_pil_image(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    
    @staticmethod
    def histogram_equalization(image):
        """直方图均衡化"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return ImageProcessor._to_pil_image(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR))
    
    @staticmethod
    def denoise(image, method="gaussian"):
        """降噪处理"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        
        if method == "gaussian":
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            denoised = cv2.medianBlur(image, 5)
        elif method == "bilateral":
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            denoised = image
            
        return ImageProcessor._to_pil_image(denoised)
    
    @staticmethod
    def adjust_contrast_brightness(image, contrast=1.0, brightness=0):
        """调整对比度和亮度"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        
        # 调整对比度
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return ImageProcessor._to_pil_image(adjusted)
    
    @staticmethod
    def edge_detection(image, method="canny"):
        """边缘检测"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == "canny":
            edges = cv2.Canny(gray, 100, 200)
        elif method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            edges = cv2.magnitude(sobelx, sobely)
            edges = cv2.convertScaleAbs(edges)
        
        return ImageProcessor._to_pil_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    
    @staticmethod
    def region_growing(image, seed_point):
        """区域生长分割"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 创建掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[seed_point[1], seed_point[0]] = 255
        
        # 区域生长参数
        threshold = 20
        kernel = np.ones((3,3), np.uint8)
        
        # 迭代生长
        prev_mask = np.zeros_like(mask)
        while not np.array_equal(prev_mask, mask):
            prev_mask = mask.copy()
            # 膨胀
            dilated = cv2.dilate(mask, kernel, iterations=1)
            # 获取新增像素
            new_pixels = cv2.subtract(dilated, mask)
            # 检查新增像素是否满足条件
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    if new_pixels[i,j] == 255:
                        if abs(int(gray[i,j]) - int(gray[seed_point[1], seed_point[0]])) < threshold:
                            mask[i,j] = 255
        
        # 应用掩码
        result = cv2.bitwise_and(image, image, mask=mask)
        return ImageProcessor._to_pil_image(result)
    
    @staticmethod
    def watershed_segmentation(image):
        """分水岭分割"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 阈值处理
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 噪声去除
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 找到未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 分水岭算法
        markers = cv2.watershed(image, markers)
        
        # 标记边界
        image[markers == -1] = [0, 0, 255]  # 红色边界
        
        return ImageProcessor._to_pil_image(image)
    
    @staticmethod
    def morphological_processing(image, operation="dilate", kernel_size=5):
        """形态学处理"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == "dilate":
            processed = cv2.dilate(image, kernel, iterations=1)
        elif operation == "erode":
            processed = cv2.erode(image, kernel, iterations=1)
        elif operation == "open":
            processed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            processed = image
            
        return ImageProcessor._to_pil_image(processed)
    
    @staticmethod
    def sharpen(image, amount=1.5, radius=1, threshold=3):
        """图像锐化
        
        Args:
            image: PIL Image对象
            amount: 锐化强度
            radius: 锐化半径
            threshold: 锐化阈值
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        
        # 使用高斯模糊创建模糊版本
        blurred = cv2.GaussianBlur(image, (radius*2+1, radius*2+1), 0)
        
        # 计算差异
        sharpened = cv2.addWeighted(
            image, 1.0 + amount, 
            blurred, -amount, 
            threshold
        )
        
        return ImageProcessor._to_pil_image(sharpened)
    
    @staticmethod
    def auto_enhance(image):
        """自动增强图像质量"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = ImageProcessor._ensure_3channel(image)
        
        # 1. 自动对比度调整
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. 自动锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # 3. 自动降噪
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
        
        return ImageProcessor._to_pil_image(enhanced)
    
    @staticmethod
    def auto_denoise(image):
        """自动去噪"""
        img_array = np.array(image)
        
        # 估计噪声水平（指定channel_axis以避免警告）
        if len(img_array.shape) == 3:  # 彩色图像
            sigma_est = np.mean(estimate_sigma(img_array, channel_axis=-1))
        else:  # 灰度图像
            sigma_est = np.mean(estimate_sigma(img_array))
        
        # 根据噪声水平选择去噪方法
        if sigma_est > 30:
            if len(img_array.shape) == 3:  # 彩色图像
                # 对每个通道分别进行小波去噪
                denoised = np.dstack([
                    denoise_wavelet(
                        img_array[:,:,i],
                        wavelet='db1',
                        mode='soft',
                        rescale_sigma=True
                    ) for i in range(img_array.shape[2])
                ])
            else:  # 灰度图像
                denoised = denoise_wavelet(img_array)
        else:
            if len(img_array.shape) == 3:  # 彩色图像
                denoised = denoise_bilateral(
                    img_array,
                    sigma_color=0.1,
                    sigma_spatial=15,
                    channel_axis=-1
                )
            else:  # 灰度图像
                denoised = denoise_bilateral(img_array)
        
        # 转换回uint8类型
        denoised = (denoised * 255).astype(np.uint8)
        return Image.fromarray(denoised)
    
    @staticmethod
    def auto_sharpen(image):
        """自动锐化"""
        img_array = np.array(image)
        
        # 使用Laplacian算子进行边缘检测
        edges = np.abs(filters.laplace(img_array))
        
        # 根据边缘强度自适应调整锐化程度
        strength = np.mean(edges) * 2
        kernel = np.array([[-strength, -strength, -strength],
                         [-strength, 1 + 8*strength, -strength],
                         [-strength, -strength, -strength]])
        
        sharpened = cv2.filter2D(img_array, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255)
        
        return Image.fromarray(sharpened.astype(np.uint8))
    
    @staticmethod
    def apply_params(image, params):
        """应用处理参数
        
        Args:
            image: PIL Image对象
            params: 包含处理参数的字典
        """
        if not isinstance(image, Image.Image):
            image = ImageProcessor._to_pil_image(image)
        
        processed = image
        
        # 根据参数类型选择处理方法
        if 'operation' in params:
            operation = params['operation']
            operation_params = params.get('parameters', {})
            
            if operation == 'denoise':
                processed = ImageProcessor.denoise(processed, **operation_params)
            elif operation == 'contrast_adjustment':
                processed = ImageProcessor.adjust_contrast_brightness(processed, **operation_params)
            elif operation == 'sharpen':
                processed = ImageProcessor.sharpen(processed, **operation_params)
            elif operation == 'histogram_equalization':
                processed = ImageProcessor.histogram_equalization(processed)
            elif operation == 'edge_detection':
                processed = ImageProcessor.edge_detection(processed, **operation_params)
            elif operation == 'auto_enhance':
                processed = ImageProcessor.auto_enhance(processed)
        
        return processed
    
    @staticmethod
    def medical_enhance(image, tissue_type):
        """医学专用图像增强
        
        Args:
            image: 输入图像
            tissue_type: 组织类型,如'bone', 'soft_tissue', 'lung'等
        """
        img_array = np.array(image)
        
        if tissue_type == 'bone':
            # 骨骼增强
            enhanced = exposure.adjust_sigmoid(img_array, cutoff=0.5, gain=10)
        elif tissue_type == 'soft_tissue':
            # 软组织增强
            enhanced = exposure.adjust_gamma(img_array, gamma=0.8)
        elif tissue_type == 'lung':
            # 肺部增强
            enhanced = exposure.adjust_log(img_array, gain=0.7)
        else:
            enhanced = img_array
            
        return Image.fromarray((enhanced * 255).astype(np.uint8))
    
    @staticmethod
    def detect_anomalies(image, threshold=0.1):
        """检测图像中的异常区域"""
        img_array = np.array(image)
        
        # 1. 计算局部统计特征
        mean = uniform_filter(img_array, size=5)
        mean_sq = uniform_filter(img_array**2, size=5)
        var = mean_sq - mean**2
        
        # 2. 检测异常区域
        anomalies = np.abs(img_array - mean) > threshold * np.sqrt(var)
        
        # 3. 形态学处理去除噪声
        anomalies = morphology.remove_small_objects(anomalies, min_size=50)
        
        return Image.fromarray((anomalies * 255).astype(np.uint8))
    
    @staticmethod
    def measure_quality(image):
        """测量图像质量指标"""
        img_array = np.array(image)
        
        # 1. 计算对比度
        contrast = np.std(img_array)
        
        # 2. 计算清晰度(使用Laplacian方差)
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # 3. 估计噪声水平
        noise_sigma = estimate_sigma(img_array)
        
        return {
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'noise_level': float(np.mean(noise_sigma))
        }
    
    @staticmethod
    def auto_optimize_image(image, progress_callback=None):
        """优化的自动图像处理流程"""
        try:
            # 创建质量控制实例
            quality_control = QualityControl()
            
            # 确保输入是 PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # 灰度图
                    image = Image.fromarray(image, 'L')
                elif len(image.shape) == 3:  # RGB或RGBA
                    if image.shape[2] == 4:  # RGBA
                        image = Image.fromarray(image, 'RGBA')
                    else:  # RGB
                        image = Image.fromarray(image, 'RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError("输入必须是 PIL Image 或 numpy array")

            # 评估原始图像质量
            original_assessment = quality_control.assess_image_quality(image)
            
            if progress_callback:
                progress_callback(0.1)

            # 转换为numpy数组
            img_array = np.array(image)
            
            # 检查图像是否为空或无效
            if img_array.size == 0:
                raise ValueError("输入图像为空")

            # 确保图像是RGB格式
            if len(img_array.shape) == 2:  # 灰度图转RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA转RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

            # 转换为浮点型并归一化（确保值在0-1之间）
            img_float = img_array.astype(np.float32) / 255.0
            img_float = np.clip(img_float, 0, 1)  # 确保值在0-1之间

            if progress_callback:
                progress_callback(0.2)

            # 1. 对比度拉伸（使用exposure.rescale_intensity）
            p2, p98 = np.percentile(img_float, (2, 98))
            img_rescale = exposure.rescale_intensity(img_float, in_range=(p2, p98))

            if progress_callback:
                progress_callback(0.3)

            # 2. 自适应直方图均衡化（仅对亮度通道）
            # 转换到LAB色彩空间
            lab = cv2.cvtColor((img_rescale * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 对亮度通道进行CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # 合并通道
            lab = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            if progress_callback:
                progress_callback(0.4)

            # 3. 降噪处理（根据噪声水平自适应调整参数）
            noise_level = original_assessment['metrics'].get('noise_level', 0)
            if noise_level > 10:  # 只在噪声明显时进行降噪
                h_param = min(10, noise_level / 2)  # 根据噪声水平调整强度
                denoised = cv2.fastNlMeansDenoisingColored(
                    enhanced,
                    None,
                    h=h_param,
                    hColor=h_param,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            else:
                denoised = enhanced

            if progress_callback:
                progress_callback(0.6)

            # 4. 自适应锐化
            sharpness = original_assessment['metrics'].get('sharpness', 0)
            if sharpness < 100:  # 只在图像不够锐利时进行锐化
                kernel_size = 3
                sigma = 1.0
                blur = cv2.GaussianBlur(denoised, (kernel_size, kernel_size), sigma)
                sharpened = cv2.addWeighted(
                    denoised, 
                    1.0 + min(0.5, (100 - sharpness) / 100),  # 根据锐度不足程度调整增强强度
                    blur, 
                    -0.5, 
                    0
                )
            else:
                sharpened = denoised

            if progress_callback:
                progress_callback(0.8)

            # 5. 最终的微调（轻微的对比度和亮度调整）
            contrast = original_assessment['metrics'].get('contrast', 0)
            brightness = original_assessment['metrics'].get('brightness', 0)
            
            # 只在对比度或亮度不足时进行调整
            if contrast < 80 or brightness < 40:
                alpha = min(1.1, 80 / max(contrast, 1))  # 对比度调整因子
                beta = max(0, min(5, (40 - brightness) / 2))  # 亮度调整因子
                final_image = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
            else:
                final_image = sharpened

            # 转换回RGB格式
            final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

            if progress_callback:
                progress_callback(0.9)

            # 获取优化后的图像
            optimized_image = Image.fromarray(final_image)
            
            # 评估优化后的图像质量
            optimized_assessment = quality_control.assess_image_quality(optimized_image)
            
            # 验证处理结果
            validation_result = quality_control.validate_processing_result(
                image,
                optimized_image,
                'auto_enhance'
            )

            # 如果优化后的质量评分反而下降，则返回原始图像
            if (optimized_assessment['overall_score'] < original_assessment['overall_score'] * 0.95):  # 允许5%的误差
                print("优化后质量下降，返回原始图像")
                return {
                    'success': True,
                    'optimized_image': image,
                    'quality_assessment': {
                        'original': original_assessment,
                        'optimized': original_assessment,
                        'validation': {
                            'is_valid': True,
                            'message': "保持原始图像不变"
                        }
                    }
                }

            if progress_callback:
                progress_callback(1.0)

            # 返回处理结果
            return {
                'success': True,
                'optimized_image': optimized_image,
                'quality_assessment': {
                    'original': original_assessment,
                    'optimized': optimized_assessment,
                    'validation': validation_result
                }
            }

        except Exception as e:
            import traceback
            error_msg = f"自动优化失败: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # 打印详细错误信息到控制台
            return {
                'success': False,
                'error': str(e)
            }

    @staticmethod
    def get_quality_grade(assessment_result):
        """获取图像质量等级
        
        Args:
            assessment_result: 质量评估结果
            
        Returns:
            str: 质量等级 (A+, A, B, C, D)
        """
        if assessment_result is None or 'metrics' not in assessment_result:
            return 'N/A'
            
        # 计算总分
        metrics = assessment_result['metrics']
        score = 0
        weights = {
            'contrast': 0.3,
            'brightness': 0.2,
            'sharpness': 0.3,
            'noise_level': 0.2
        }
        
        for metric, weight in weights.items():
            if metric in metrics:
                normalized_score = min(max(metrics[metric] / 100, 0), 1)
                score += normalized_score * weight
        
        # 根据总分返回等级
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