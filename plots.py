"""绘图模块：训练曲线、混淆矩阵和预测图片；不负责训练或修改参数。"""

import config
import matplotlib

# 使用文件绘图后端，服务器和没有桌面的电脑也能生成图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_curves(history, run_dir):
    """history 每个列表按轮次排列；用曲线观察过拟合和学习率变化。"""
    epochs = np.arange(1, len(history["loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for axis, metric, title in zip(axes[:2], ["loss", "accuracy"], ["Loss (includes L2)", "Accuracy"]):
        axis.plot(epochs, history[metric], color="#0072B2", label="Train")
        axis.plot(epochs, history["val_" + metric], color="#D55E00", linestyle="--", label="Validation")
        axis.axvline(best_epoch, color="0.4", linestyle=":", label="Best val loss")
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[2].step(epochs, history["learning_rate"], where="post", color="#0072B2")
    axes[2].set(xlabel="Epoch", ylabel="Learning rate", title="Learning rate used", yscale="log")
    # 曲线另存 SVG，放大后文字和线条仍然清楚。
    figure.savefig(run_dir / "curves.png", dpi=180)
    figure.savefig(run_dir / "curves.svg")
    plt.close(figure)
    return figure


def save_confusion_matrix(labels, predictions, run_dir):
    """行是真实数字，列是预测数字；对角线之外的计数都是识别错误。"""
    matrix = np.zeros((10, 10), dtype=int)
    # 例如真实为 7、预测为 1，就把第 7 行第 1 列的计数加一。
    np.add.at(matrix, (labels, predictions), 1)
    figure, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    heatmap = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(heatmap, ax=axis, label="Number of images")
    for row in range(10):
        for column in range(10):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color, fontsize=8)
    axis.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit", title="Test confusion matrix (best model)")
    figure.savefig(run_dir / "confusion_matrix.png", dpi=180)
    figure.savefig(run_dir / "confusion_matrix.svg")
    plt.close(figure)
    return figure


def save_predictions(images, labels, probabilities, path, title):
    """显示最多 10 张图；标题同时列出真实/预测数字，便于发现失败模式。"""
    count = min(len(images), 10)
    figure, axes = plt.subplots(2, 5, figsize=(11, 5), layout="constrained")
    predictions = np.argmax(probabilities, axis=1)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < count:
            prediction = predictions[index]
            probability = probabilities[index, prediction]
            axis.imshow(images[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"True: {labels[index]} / Pred: {prediction}\nProbability: {probability:.1%}", fontsize=10)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return figure
