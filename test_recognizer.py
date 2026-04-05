from digit_recognizer import DigitRecognizer
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 创建识别器实例
recognizer = DigitRecognizer()

# 加载模型
recognizer.load_model()

# 创建测试数字图像
def create_test_digits():
    # 创建一个简单的数字测试图像
    digits = []
    for i in range(10):
        # 创建一个空白图像
        img = Image.new('L', (28, 28), 255)
        pixels = img.load()
        
        # 简单绘制数字
        if i == 0:
            # 绘制0
            for x in range(5, 23):
                pixels[x, 5] = 0
                pixels[x, 22] = 0
            for y in range(5, 23):
                pixels[5, y] = 0
                pixels[22, y] = 0
        elif i == 1:
            # 绘制1
            for y in range(5, 23):
                pixels[14, y] = 0
            for x in range(10, 19):
                pixels[x, 5] = 0
        elif i == 2:
            # 绘制2
            for x in range(5, 23):
                pixels[x, 5] = 0
                pixels[x, 14] = 0
                pixels[x, 22] = 0
            pixels[5, 10] = 0
            pixels[22, 18] = 0
        elif i == 3:
            # 绘制3
            for x in range(5, 23):
                pixels[x, 5] = 0
                pixels[x, 14] = 0
                pixels[x, 22] = 0
            pixels[22, 10] = 0
            pixels[22, 18] = 0
        elif i == 4:
            # 绘制4
            for y in range(5, 14):
                pixels[5, y] = 0
                pixels[22, y] = 0
            for x in range(5, 23):
                pixels[x, 14] = 0
            for y in range(14, 23):
                pixels[22, y] = 0
        elif i == 5:
            # 绘制5
            for x in range(5, 23):
                pixels[x, 5] = 0
                pixels[x, 14] = 0
                pixels[x, 22] = 0
            pixels[22, 10] = 0
            pixels[5, 18] = 0
        elif i == 6:
            # 绘制6
            for x in range(5, 23):
                pixels[x, 5] = 0
                pixels[x, 14] = 0
                pixels[x, 22] = 0
            for y in range(5, 23):
                pixels[5, y] = 0
            pixels[22, 18] = 0
        elif i == 7:
            # 绘制7
            for x in range(5, 23):
                pixels[x, 5] = 0
            for y in range(5, 23):
                pixels[22, y] = 0
        elif i == 8:
            # 绘制8
            for x in range(5, 23):
                pixels[x, 5] = 0
                pixels[x, 14] = 0
                pixels[x, 22] = 0
            for y in range(5, 23):
                pixels[5, y] = 0
                pixels[22, y] = 0
        elif i == 9:
            # 绘制9
            for x in range(5, 23):
                pixels[x, 5] = 0
                pixels[x, 14] = 0
            for y in range(5, 14):
                pixels[5, y] = 0
                pixels[22, y] = 0
            for y in range(14, 23):
                pixels[22, y] = 0
        
        # 保存图像
        img_path = f'test_digit_{i}.png'
        img.save(img_path)
        digits.append(img_path)
    
    return digits

# 创建测试数字
print("创建测试数字图像...")
test_digits = create_test_digits()

# 测试预测功能
print("\n测试数字识别功能...")
for img_path in test_digits:
    digit = int(img_path.split('_')[-1].split('.')[0])
    predicted, confidence = recognizer.predict(img_path)
    print(f"实际数字: {digit}, 预测结果: {predicted}, 置信度: {confidence:.2f}")

print("\n测试完成！")

# 清理测试文件
print("清理测试文件...")
for img_path in test_digits:
    if os.path.exists(img_path):
        os.remove(img_path)

print("所有测试文件已清理。")
