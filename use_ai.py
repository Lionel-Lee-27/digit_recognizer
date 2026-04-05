import sys
from digit_recognizer import DigitRecognizer

# 创建识别器实例
recognizer = DigitRecognizer()

# 加载模型（如果模型不存在会自动训练）
recognizer.load_model()

# 检查命令行参数
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # 如果没有提供命令行参数，使用默认图像路径
    image_path = 'test_digit.png'
    print(f"未提供图像路径，使用默认路径: {image_path}")

# 方法1：显示预测结果（带图像）
print("正在预测并显示结果...")
recognizer.display_prediction(image_path)

# 方法2：直接获取预测结果（不显示图像）
print("正在预测...")
predicted_digit, confidence = recognizer.predict(image_path)
print(f"预测结果: {predicted_digit}, 置信度: {confidence:.2f}")