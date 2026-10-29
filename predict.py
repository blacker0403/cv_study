"""预测入口：选择 best 或 last，识别测试图片；不会继续训练或更新参数。"""

import argparse
from pathlib import Path

import config  # 先检查 Python 3.11。
import numpy as np
from tensorflow import keras

from data import preprocess_images
from plots import save_predictions


def main():
    parser = argparse.ArgumentParser(description="加载训练好的 best 或 last 模型进行预测")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best", help="默认使用验证损失最低的 best")
    parser.add_argument("--run-dir", type=Path, help="指定一次实验的目录；默认选择最近完整完成的实验")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        completed_runs = sorted(config.OUTPUT_DIR.glob("*/metrics.json"))
        if not completed_runs:
            raise SystemExit("没有完整完成的实验，请先运行：python train.py")
        run_dir = completed_runs[-1].parent
    model_path = run_dir / f"{args.checkpoint}.keras"
    if not model_path.exists():
        raise SystemExit(f"模型不存在：{model_path}，请检查实验目录和训练是否成功。")

    model = keras.models.load_model(model_path)
    (_, _), (images, labels) = keras.datasets.mnist.load_data()
    if not 1 <= config.PREDICTION_COUNT <= len(images):
        raise ValueError("PREDICTION_COUNT 必须在 1 到测试图片总数之间。")
    images = preprocess_images(images[:config.PREDICTION_COUNT])
    labels = labels[:config.PREDICTION_COUNT]
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print(f"使用模型：{model_path}")
    for index, prediction in enumerate(predictions):
        # 标签只用于展示对照，没有输入模型；未知标签的图片同样可以预测。
        probability = probabilities[index, prediction]
        print(f"图片 {index + 1:2d}：真实 {labels[index]}，预测 {prediction}，概率 {probability:.1%}")
    image_path = run_dir / f"predictions_{args.checkpoint}.png"
    save_predictions(images, labels, probabilities, image_path, f"Test predictions ({args.checkpoint})")
    print(f"预测图片已保存（最多展示前 10 张）：{image_path}")


if __name__ == "__main__":
    main()
