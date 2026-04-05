# 数字识别AI项目

这是一个基于TensorFlow的手写数字识别项目，使用卷积神经网络(CNN)实现对手写数字的识别。

## 项目结构

```
├── digit_recognizer.py    # 核心识别器类
├── digit_recognizer_model.h5  # 训练好的模型文件
├── requirements.txt       # 项目依赖
├── use_ai.py              # 使用AI的脚本
├── test_all.py            # 测试所有数字图像
├── test_digit_0.png       # 测试图像 - 数字0
├── test_digit_1.png       # 测试图像 - 数字1
├── test_digit_2.png       # 测试图像 - 数字2
├── test_digit_3.png       # 测试图像 - 数字3
├── test_digit_4.png       # 测试图像 - 数字4
├── test_digit_5.png       # 测试图像 - 数字5
├── test_digit_6.png       # 测试图像 - 数字6
├── test_digit_7.png       # 测试图像 - 数字7
├── test_digit_8.png       # 测试图像 - 数字8
├── test_digit_9.png       # 测试图像 - 数字9
└── README.md              # 项目说明文档
```

## 功能特点

- 使用卷积神经网络(CNN)进行数字识别
- 自动训练和加载模型
- 支持图像预处理（灰度转换、大小调整、二值化）
- 提供命令行接口和API调用方式
- 支持批量测试多个数字图像
- 显示预测结果和置信度

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

首次运行时，系统会自动训练模型：

```bash
python digit_recognizer.py
```

### 2. 识别单个数字

使用命令行参数指定图像路径：

```bash
python use_ai.py test_digit_5.png
```

如果不指定图像路径，将使用默认路径：

```bash
python use_ai.py
```

### 3. 批量测试所有数字

```bash
python test_all.py
```

## API使用

### 导入识别器

```python
from digit_recognizer import DigitRecognizer

# 创建识别器实例
recognizer = DigitRecognizer()

# 加载模型（如果模型不存在会自动训练）
recognizer.load_model()
```

### 预测数字

```python
# 方法1：显示预测结果（带图像）
recognizer.display_prediction('test_digit_3.png')

# 方法2：直接获取预测结果（不显示图像）
predicted_digit, confidence = recognizer.predict('test_digit_7.png')
print(f"预测结果: {predicted_digit}, 置信度: {confidence:.2f}")
```

## 模型说明

- 模型架构：卷积神经网络(CNN)
- 输入：28x28像素的灰度图像
- 输出：0-9的数字预测
- 训练数据集：MNIST手写数字数据集
- 训练轮数：5轮
- 批次大小：32

## 测试结果

模型在MNIST测试集上的准确率约为99%左右，在项目提供的测试图像上也能达到较高的识别准确率。

## 技术栈

- Python 3.12+
- TensorFlow 2.0+
- NumPy
- Pillow (图像处理)
- Matplotlib (图像显示)

## 注意事项

1. 首次运行时会下载MNIST数据集并训练模型，可能需要一些时间
2. 确保输入图像是清晰的手写数字，背景简单
3. 模型会自动处理图像的预处理，包括灰度转换、大小调整和二值化
4. 如果模型文件不存在，系统会自动重新训练

## 许可证

本项目采用MIT许可证，详见LICENSE文件。