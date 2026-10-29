"""模型模块：只描述图片如何变成 10 个数字的预测概率。"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    """使用基础层组合一个小 CNN，参数由训练过程学习。"""
    # Sequential 表示按列表顺序执行，每一层的输出交给下一层。
    return keras.Sequential(
        [
            # 一张输入图片：28 像素高 × 28 像素宽 × 1 个灰度通道。
            # 不写图片数量，因此一次可以处理 1 张或一批图片。
            keras.Input(shape=(28, 28, 1)),

            # 8 个可学习的 3×3 卷积核在图片上滑动，学习笔画等局部特征。
            # 不补边，步长为 1：28 - 3 + 1 = 26，得到 26×26×8。
            # ReLU 把负数变为 0，加入非线性，让网络能表示更复杂的规律。
            layers.Conv2D(8, kernel_size=3, activation="relu"),

            # 每个 2×2 区域取最大值，缩小特征图：26×26×8 → 13×13×8。
            # 池化本身没有需要学习的参数。
            layers.MaxPooling2D(pool_size=2),

            # 展平成一行 13×13×8 = 1352 个数；只改变形状，不学习参数。
            layers.Flatten(),

            # 全连接层综合这些特征，输出下标 0～9 对应的 10 个分数。
            # softmax 将分数转成总和为 1 的概率，最大概率对应预测数字。
            layers.Dense(10, activation="softmax"),
        ],
        name="simple_mnist_cnn",
    )
