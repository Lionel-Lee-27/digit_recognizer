import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DigitRecognizer:
    def __init__(self):
        self.model = None
    
    def train_model(self):
        # 加载MNIST数据集
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # 数据预处理
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # 构建模型
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # 编译模型
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        # 训练模型
        print("开始训练模型...")
        history = self.model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
        
        # 评估模型
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print(f"测试准确率: {test_acc:.4f}")
        
        # 保存模型
        self.model.save('digit_recognizer_model.h5')
        print("模型保存成功!")
    
    def load_model(self):
        try:
            self.model = tf.keras.models.load_model('digit_recognizer_model.h5')
            print("模型加载成功!")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("正在训练新模型...")
            self.train_model()
    
    def preprocess_image(self, image_path):
        # 加载并预处理图像
        img = Image.open(image_path).convert('L')  # 转换为灰度图
        
        # 调整大小为28x28
        img = img.resize((28, 28))
        
        # 转换为数组
        img_array = np.array(img)
        
        # 反转图像（如果需要）- MNIST数据集中数字是白色，背景是黑色
        # 检查图像是否需要反转
        if np.mean(img_array) > 127:  # 如果图像整体较亮
            img_array = 255 - img_array  # 反转图像
        
        # 二值化处理 - 调整阈值以获得更好的效果
        img_array = np.where(img_array > 100, 255, 0)
        
        # 归一化
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
        return img_array
    
    def predict(self, image_path):
        if self.model is None:
            self.load_model()
        
        # 预处理图像
        img_array = self.preprocess_image(image_path)
        
        # 预测
        predictions = self.model.predict(img_array)
        predicted_digit = np.argmax(predictions)
        confidence = np.max(predictions)
        
        return predicted_digit, confidence
    
    def display_prediction(self, image_path):
        # 预测数字
        predicted_digit, confidence = self.predict(image_path)
        
        # 显示图像和预测结果
        img = Image.open(image_path)
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.title(f"预测结果: {predicted_digit}\n置信度: {confidence:.2f}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # 创建识别器实例
    recognizer = DigitRecognizer()
    
    # 训练模型（首次运行时）
    recognizer.train_model()
    
    # 示例：预测数字
    # 注意：这里需要替换为实际的数字图像路径
    # recognizer.display_prediction('path_to_digit_image.png')
    print("数字识别AI准备就绪！")
    print("使用方法：")
    print("1. 训练模型：运行本脚本")
    print("2. 预测数字：调用 recognizer.display_prediction('image_path')")
    print("3. 直接预测：调用 recognizer.predict('image_path')")
