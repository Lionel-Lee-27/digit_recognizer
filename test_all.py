import os
import sys
from digit_recognizer import DigitRecognizer

# 创建识别器实例
recognizer = DigitRecognizer()

# 加载模型
recognizer.load_model()

# 测试图像列表
test_images = [
    'test_digit_0.png',
    'test_digit_1.png',
    'test_digit_2.png',
    'test_digit_3.png',
    'test_digit_4.png',
    'test_digit_5.png',
    'test_digit_6.png',
    'test_digit_7.png',
    'test_digit_8.png',
    'test_digit_9.png'
]

# 测试所有图像
correct_count = 0
total_count = len(test_images)

print("开始测试所有图像...")

for i, image_path in enumerate(test_images):
    expected_digit = str(i)  # 文件名中的数字是预期结果
    predicted_digit, confidence = recognizer.predict(image_path)
    is_correct = str(predicted_digit) == expected_digit
    
    if is_correct:
        correct_count += 1
    
    print(f"图像: {image_path}, 预期: {expected_digit}, 预测: {predicted_digit}, 置信度: {confidence:.2f}, {'正确' if is_correct else '错误'}")

# 计算准确率
accuracy = correct_count / total_count * 100
print(f"\n测试完成! 准确率: {accuracy:.2f}% ({correct_count}/{total_count})")