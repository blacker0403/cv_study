"""训练显示模块：只读取 Keras 提供的指标，不计算损失、不修改训练规则。"""

import math
import sys
from time import perf_counter

import config  # 独立导入本模块时也先检查 Python 版本。
from tensorflow import keras
from tqdm import tqdm


class TrainingLogger(keras.callbacks.Callback):
    """终端显示训练/验证进度条；重定向到文件时每轮只写一行指标。"""

    def __init__(self, validation_size, batch_size, stream=None):
        super().__init__()
        self.validation_size = validation_size
        self.validation_steps = math.ceil(validation_size / batch_size)
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = self.stream.isatty()
        self.bar = None

    def write(self, message):
        print(message, file=self.stream, flush=True)

    @staticmethod
    def metric_columns(logs, names):
        # Keras 给出截至当前批次的累计指标，不能再对这些值求一次平均。
        values = [f"{logs[name]:.4f}" if name in logs else "-" for name in names]
        return "".join(f"{value:>9}" for value in values)

    def train_row(self, logs):
        epoch = f"{self.epoch + 1}/{self.params['epochs']}"
        return (f"{epoch:>9}" + self.metric_columns(logs, ("loss", "accuracy"))
                + f"{self.learning_rate:>9.4g}")

    def validation_row(self, logs):
        return (f"{'val':>9}{self.validation_size:>9}"
                + self.metric_columns(logs, ("loss", "accuracy")))

    def start_bar(self, total, description):
        # ASCII 进度条适配常见 Windows 终端；限速刷新减少闪烁。
        self.bar = tqdm(
            total=total, desc=description, file=self.stream, ascii=True,
            dynamic_ncols=True, mininterval=0.2, unit="it",
        )

    def close(self):
        # 正常结束或中断时都补齐换行，避免错误信息挤在进度条后面。
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def on_train_begin(self, logs=None):
        self.started = perf_counter()
        if not self.interactive:
            self.write("\n" + "".join(f"{name:>9}" for name in (
                "Epoch", "Loss", "Acc", "Val_loss", "Val_acc", "LR", "Time",
            )))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_started = perf_counter()
        self.learning_rate = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        if self.interactive:
            self.write("\n" + "".join(f"{name:>9}" for name in ("Epoch", "Loss", "Acc", "LR")))
            self.start_bar(self.params["steps"], self.train_row({}))

    def on_train_batch_end(self, batch, logs=None):
        if self.bar is not None:
            self.bar.set_description_str(self.train_row(logs), refresh=False)
            self.bar.update(batch + 1 - self.bar.n)

    def on_test_begin(self, logs=None):
        # fit 在每轮训练后调用验证钩子，先结束训练条，再显示验证条。
        if self.interactive:
            self.close()
            self.write("".join(f"{name:>9}" for name in ("Split", "Images", "Loss", "Acc")))
            self.start_bar(self.validation_steps, self.validation_row({}))

    def on_test_batch_end(self, batch, logs=None):
        if self.bar is not None:
            self.bar.set_description_str(self.validation_row(logs), refresh=False)
            self.bar.update(batch + 1 - self.bar.n)

    def on_test_end(self, logs=None):
        self.close()

    def on_epoch_end(self, epoch, logs=None):
        self.close()
        if not self.interactive:
            epoch_label = f"{epoch + 1}/{self.params['epochs']}"
            elapsed = f"{perf_counter() - self.epoch_started:.1f}s"
            self.write(
                f"{epoch_label:>9}"
                + self.metric_columns(logs, ("loss", "accuracy", "val_loss", "val_accuracy"))
                + f"{self.learning_rate:>9.4g}{elapsed:>9}"
            )
        # 本回调排在调度器之后；表格显示本轮值，这里仅提示下一轮的新值。
        next_rate = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        if next_rate != self.learning_rate:
            self.write(f"LR: {self.learning_rate:.4g} -> {next_rate:.4g} (next epoch)")

    def on_train_end(self, logs=None):
        self.close()
        self.write(f"\n{self.epoch + 1}/{self.params['epochs']} epochs completed "
                   f"in {perf_counter() - self.started:.1f}s")
