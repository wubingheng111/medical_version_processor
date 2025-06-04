# 基于chatVLM的医学图像处理系统

这是一个基于Python的医学图像处理系统，集成了图像处理、智能诊断和工作流管理功能。

## 功能特点

1. 图像处理
   - 亮度和对比度调整
   - 图像增强
   - 滤波处理
   - 细节增强

2. 智能诊断
   - 病变区域检测
   - VLM模型分析
   - 特征提取和分析
   - 诊断报告生成

3. 工作流管理
   - 预定义工作流模板
   - 自定义工作流创建
   - 工作流执行和监控
   - 处理历史记录

4. 用户界面
   - 手动模式和工作流模式
   - 实时图像预览
   - 参数调整滑块
   - 结果可视化

## 系统要求

- Python 3.8+
- CUDA支持（可选，用于GPU加速）
- 足够的磁盘空间用于存储图像和模型

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/medical_image_processor.git
cd medical_image_processor
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

1. 启动应用：
```bash
streamlit run src/app.py
```

2. 在浏览器中访问应用（默认地址：http://localhost:8501）

3. 使用流程：
   - 上传医学图像
   - 选择操作模式（手动/工作流）
   - 根据需要调整参数或选择工作流
   - 查看和导出结果

## 项目结构

```
medical_image_processor/
├── src/
│   ├── core/
│   │   ├── image_processor.py
│   │   ├── smart_diagnosis.py
│   │   └── smart_workflow.py
│   ├── ui/
│   │   └── medical_ui.py
│   └── app.py
├── data/
│   ├── models/
│   └── workflows/
├── test_data/
├── requirements.txt
└── README.md
```

## 开发说明

1. 代码风格
   - 遵循PEP 8规范
   - 使用类型注解
   - 添加适当的注释和文档字符串

2. 测试
   - 使用pytest进行单元测试
   - 运行测试：`pytest tests/`

3. 贡献指南
   - Fork项目
   - 创建功能分支
   - 提交更改
   - 发起Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或发送邮件至：your.email@example.com 
