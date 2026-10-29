"""预测入口：读取已训练的模型，识别测试集前 10 张手写数字。"""

from pathlib import Path

import numpy as np
from tensorflow import keras

from data import preprocess_images


def main():
    model_path = Path(__file__).resolve().parent / "outputs" / "mnist_cnn.keras"
    if not model_path.exists():
        raise SystemExit("还没有训练好的模型，请先运行：python train.py")

    # 加载结构和参数即可预测，不需要重新创建模型或重新训练。
    model = keras.models.load_model(model_path)
    (_, _), (images, labels) = keras.datasets.mnist.load_data()
    images = preprocess_images(images[:10])  # 与训练使用完全相同的预处理。

    # 输出形状为 (10, 10)：10 张图片，每张对应数字 0～9 的 10 个概率。
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)  # 每行最大概率的下标就是数字。

    for index, prediction in enumerate(predictions):
        probability = probabilities[index, prediction]
        print(f"图片 {index + 1:2d}：真实数字 {labels[index]}，预测数字 {prediction}，概率 {probability:.1%}")


if __name__ == "__main__":
    main()
