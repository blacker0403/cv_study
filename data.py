"""数据模块：下载 MNIST，把图片整理成 CNN 可以接收的格式。"""

import config
from tensorflow import keras


def preprocess_images(images):
    """输入 (图片数, 28, 28)，输出 (图片数, 28, 28, 1)。"""
    # 原始像素是 0～255 的整数。转成小数并缩放到 0～1，便于网络训练。
    images = images.astype("float32") / 255.0

    # CNN 需要通道维度：灰度图有 1 个通道，彩色图通常有 3 个通道。
    # 这里只补一个维度，不改变图片内容；-1 表示保留原来的图片数量。
    return images.reshape(-1, 28, 28, 1)


def load_data():
    """返回训练集、验证集、测试集；每一份都是 (图片, 正确数字)。"""
    # 首次运行会联网下载，之后直接读取本机的 ~/.keras/datasets/mnist.npz。
    # 官方提供 60,000 张训练图片和 10,000 张测试图片；标签是整数 0～9。
    (images, labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    images = preprocess_images(images)
    test_images = preprocess_images(test_images)

    # 默认从官方训练集拿出最后 10,000 张做验证，剩下 50,000 张用于更新参数。
    # 验证集只用来观察训练效果；测试集留到训练结束后做最后评估。
    # 防止把验证集设为 0 或占满训练集，导致切片意外或根本没有训练数据。
    if not 0 < config.VALIDATION_SIZE < len(images):
        raise ValueError("VALIDATION_SIZE 必须大于 0 且小于官方训练集数量。")
    split = len(images) - config.VALIDATION_SIZE
    train_data = (images[:split], labels[:split])
    val_data = (images[split:], labels[split:])
    test_data = (test_images, test_labels)
    return train_data, val_data, test_data
