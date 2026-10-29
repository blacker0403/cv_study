"""训练入口：配置 → 数据 → 模型 → 训练控制 → 选择 best → 最终测试。"""

import json
import platform
from datetime import datetime
from importlib.metadata import version

import config  # 先检查 Python 3.11，再导入第三方依赖。
import numpy as np
from tensorflow import keras

from data import load_data
from model import build_model
from plots import save_confusion_matrix, save_curves, save_predictions
from training_logger import TrainingLogger


class RecordLearningRate(keras.callbacks.Callback):
    """Callback 是训练过程的钩子；这里在每轮结束时记录本轮使用的学习率。"""

    def on_epoch_end(self, epoch, logs=None):
        # 放在学习率调度器之前执行，记录的是本轮值，而不是下一轮的新值。
        logs["learning_rate"] = float(keras.backend.get_value(self.model.optimizer.learning_rate))


def build_callbacks(run_dir):
    """训练规则集中放在这里，模型模块不需要知道如何保存、降学习率或早停。"""
    callbacks = [keras.callbacks.TerminateOnNaN(), RecordLearningRate()]
    if config.USE_LR_SCHEDULER:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE, min_lr=config.MIN_LEARNING_RATE,
            min_delta=0.0, verbose=0,
        ))

    callbacks.extend([
        # 只比较验证损失，越低越好。绝不能按测试准确率挑选 best。
        keras.callbacks.ModelCheckpoint(
            run_dir / "best.keras", monitor="val_loss", mode="min",
            save_best_only=True, verbose=0,
        ),
        # 每个完整轮次结束都覆盖 last，保留这一轮的模型和优化器状态。
        keras.callbacks.ModelCheckpoint(run_dir / "last.keras", save_best_only=False),
        # 每轮一行记录，即使后续中断，已完成轮次的指标也可以查看。
        keras.callbacks.CSVLogger(run_dir / "history.csv"),
    ])
    if config.USE_EARLY_STOPPING:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE,
            # 不把内存模型恢复成 best，否则容易误把 best 当作 last 保存。
            restore_best_weights=False, min_delta=0.0, verbose=0,
        ))
    return callbacks


def main():
    keras.utils.set_random_seed(config.SEED)
    # 固定种子便于比较，但不同硬件/软件版本仍可能产生数值差异。
    (x_train, y_train), val_data, (x_test, y_test) = load_data()

    # 每次训练使用新目录，旧实验的 best、last、曲线和参数不会被覆盖。
    run_dir = config.OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    settings = {name: value for name, value in vars(config).items() if name.isupper()}
    # 包版本也影响结果；依赖清单未固定版本，因此记录实际安装环境。
    settings["VERSIONS"] = {
        "python": platform.python_version(),
        "tensorflow": version("tensorflow"), "keras": version("keras"),
        "numpy": version("numpy"), "matplotlib": version("matplotlib"),
        "tqdm": version("tqdm"),
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")

    model = build_model()
    print(f"MNIST CNN  Python-{platform.python_version()}  TensorFlow-{version('tensorflow')}  Keras-{version('keras')}")
    print(f"Model: {model.name}, {model.count_params():,} parameters, input={x_train.shape[1:]}")
    print(f"Data: train={len(x_train)}, val={len(val_data[0])}, test={len(x_test)}, batch={config.BATCH_SIZE}")
    print(f"Optimizer: Adam, lr={config.LEARNING_RATE:g}, epochs={config.EPOCHS}, seed={config.SEED}")
    print(f"Save directory: {run_dir}")
    # compile 只设置学习规则；fit 才执行前向传播、计算损失、求梯度、更新参数。
    # softmax 输出概率，因此交叉熵使用默认 from_logits=False；标签是整数。
    # clipnorm 限制梯度范数，不能代替正确的数据检查和合适的学习率。
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, clipnorm=config.CLIPNORM),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    progress = TrainingLogger(len(val_data[0]), config.BATCH_SIZE)
    # 关闭 Keras 的重复输出；显示回调放在最后，只读取其他回调处理后的状态。
    try:
        history = model.fit(
            x_train, y_train,
            batch_size=config.BATCH_SIZE, epochs=config.EPOCHS,
            validation_data=val_data, shuffle=True,
            callbacks=[*build_callbacks(run_dir), progress], verbose=0,
        )
    finally:
        progress.close()

    # NaN/Inf 表示训练异常；避免把损坏的模型当作正常完成的实验。
    for metric, values in history.history.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"{metric} 出现 NaN/Inf，请检查数据和学习率。日志位于：{run_dir}")
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    last_epoch = len(history.history["loss"])
    if last_epoch < config.EPOCHS:
        print(f"Early stopping: val_loss did not improve for {config.EARLY_STOPPING_PATIENCE} epochs.")
    save_curves(history.history, run_dir)

    # 训练结束后加载 best 做最终测试。last 保留最后完成轮次，不因评估而覆盖。
    best_model = keras.models.load_model(run_dir / "best.keras")
    print("Evaluating best.keras on the test set...")
    test_metrics = best_model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    # evaluate 用标签评分；predict 只接收图片，输出每一类的概率。
    probabilities = best_model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    save_confusion_matrix(y_test, predictions, run_dir)
    wrong_indices = np.flatnonzero(predictions != y_test)
    if len(wrong_indices):
        selected = wrong_indices[:10]
        save_predictions(
            x_test[selected], y_test[selected], probabilities[selected],
            run_dir / "mistakes.png", "First test mistakes (best model)",
        )

    # metrics.json 最后写入，也用作“本次实验已完整完成”的标记。
    report = {
        "best_epoch": best_epoch, "last_epoch": last_epoch,
        "selection_metric": "val_loss", "best_val_loss": float(min(history.history["val_loss"])),
        "best_model_test": test_metrics, "test_errors": int(len(wrong_indices)),
        "train_size": len(x_train), "validation_size": len(val_data[0]), "test_size": len(x_test),
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "".join(f"{name:>12}" for name in ("Model", "Epoch", "Val_loss", "Test_loss", "Test_acc")))
    print(f"{'best.keras':>12}{best_epoch:>12}{report['best_val_loss']:>12.4f}"
          f"{test_metrics['loss']:>12.4f}{test_metrics['accuracy']:>12.4f}")
    print(f"{'last.keras':>12}{last_epoch:>12}{history.history['val_loss'][-1]:>12.4f}{'-':>12}{'-':>12}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
