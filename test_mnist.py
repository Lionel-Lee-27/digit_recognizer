from digit_recognizer import DigitRecognizer
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 创建识别器实例
recognizer = DigitRecognizer()

# 加载模型
recognizer.load_model()

# 加载MNIST数据集进行测试
def test_with_mnist():
    # 加载MNIST数据集
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 选择前10个测试图像进行测试
    test_images = x_test[:10]
    test_labels = y_test[:10]
    
    # 测试预测功能
    print("使用MNIST测试图像进行测试...")
    correct = 0
    total = len(test_images)
    
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        # 保存图像
        img = Image.fromarray(image)
        img_path = f'mnist_test_{i}.png'
        img.save(img_path)
        
        # 预测
        predicted, confidence = recognizer.predict(img_path)
        
        # 检查预测是否正确
        if predicted == label:
            correct += 1
        
        print(f"图像 {i+1}: 实际数字: {label}, 预测结果: {predicted}, 置信度: {confidence:.2f}")
        
        # 清理测试文件
        if os.path.exists(img_path):
            os.remove(img_path)
    
    # 计算准确率
    accuracy = correct / total
    print(f"\n测试完成！准确率: {accuracy:.2f}")

if __name__ == "__main__":
    test_with_mnist()
