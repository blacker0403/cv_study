"""训练入口：准备数据 → 创建模型 → 学习参数 → 测试 → 保存。"""

from pathlib import Path

from tensorflow import keras

from data import load_data
from model import build_model

# 初学时只需要改这里：epoch 是完整学习训练集的轮数，batch 是每批图片数。
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 以脚本所在目录为起点，避免从不同工作目录运行时把结果存到别处。
MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "mnist_cnn.keras"


def main():
    # 固定随机种子，让同一环境下的实验便于比较；跨平台仍可能有小差异。
    keras.utils.set_random_seed(42)
    (x_train, y_train), val_data, (x_test, y_test) = load_data()
    print(f"训练 {len(x_train)} 张，验证 {len(val_data[0])} 张，测试 {len(x_test)} 张")

    model = build_model()
    model.summary()  # 查看每一层的输出形状与参数数量。

    # compile 只指定学习规则，此时还没有开始训练。
    # Adam 根据梯度更新参数；学习率控制每次更新的步幅。
    # sparse 表示标签直接使用整数 0～9，不需要改成长度为 10 的向量。
    # 交叉熵衡量预测与正确标签的差距，越小越好；accuracy 是预测正确的比例。
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # fit 对每批数据依次执行：预测 → 计算损失 → 反向传播求梯度 → 更新参数。
    # Keras 负责自动求导，我们专注理解数据、模型与训练过程。
    # 每轮结束会检查验证集，验证时不会更新参数。
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data,
        shuffle=True,
        verbose=2,
    )

    # evaluate 只评估，不学习。测试集没有参与前面的参数更新和验证。
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试损失：{test_loss:.4f}，测试准确率：{test_accuracy:.2%}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)  # .keras 文件包含模型结构、学到的参数和优化器状态。
    print(f"模型已保存：{MODEL_PATH}")


# 直接运行 python train.py 才训练；被其他文件导入时不自动执行。
if __name__ == "__main__":
    main()
