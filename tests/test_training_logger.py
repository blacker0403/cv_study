"""日志只改变显示，不参与指标计算和参数更新。"""

import io
import unittest
from unittest.mock import patch

import config
from tensorflow import keras

from training_logger import TrainingLogger


class TerminalStream(io.StringIO):
    def isatty(self):
        return True


class TrainingLoggerTests(unittest.TestCase):
    def run_epochs(self, stream):
        model = keras.Sequential([keras.Input(shape=(1,)), keras.layers.Dense(1)])
        model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
        logger = TrainingLogger(validation_size=3, batch_size=2, stream=stream)
        logger.set_model(model)
        logger.set_params({"epochs": 2, "steps": 2, "verbose": 0})
        logger.on_train_begin()
        for epoch in range(2):
            logger.on_epoch_begin(epoch)
            for batch in range(2):
                logger.on_train_batch_end(batch, {"loss": 0.8 - batch * 0.2, "accuracy": 0.5})
            logger.on_test_begin()
            for batch in range(2):
                logger.on_test_batch_end(batch, {"loss": 0.4 - batch * 0.1, "accuracy": 0.75})
            logger.on_test_end()
            logs = {"loss": 0.6, "accuracy": 0.5, "val_loss": 0.3, "val_accuracy": 0.75,
                    "learning_rate": float(model.optimizer.learning_rate)}
            original = logs.copy()
            # 模拟调度器在第一轮结束时降低下一轮的学习率。
            if epoch == 0:
                model.optimizer.learning_rate.assign(0.005)
            logger.on_epoch_end(epoch, logs)
            self.assertEqual(logs, original)
            self.assertIsNone(logger.bar)
        logger.on_train_end()
        return stream.getvalue()

    def test_redirected_output_has_one_row_per_epoch_and_used_learning_rate(self):
        output = self.run_epochs(io.StringIO())
        self.assertNotIn("\r", output)
        self.assertNotIn("\x1b", output)
        self.assertEqual(output.count("Val_loss"), 1)
        rows = [line.split() for line in output.splitlines()
                if len(line.split()) == 7 and line.strip().startswith(("1/2 ", "2/2 "))]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][:6], ["1/2", "0.6000", "0.5000", "0.3000", "0.7500", "0.01"])
        self.assertEqual(rows[1][:6], ["2/2", "0.6000", "0.5000", "0.3000", "0.7500", "0.005"])
        self.assertEqual(output.count("LR:"), 1)
        self.assertIn("0.01 -> 0.005", output)

    def test_terminal_has_separate_train_and_validation_progress(self):
        # 模拟 80 列终端，确认完整指标与进度时间不会因固定长进度条被截断。
        with patch("tqdm.std._screen_shape_wrapper", return_value=lambda stream: (80, 24)):
            output = self.run_epochs(TerminalStream())
        self.assertIn("\r", output)
        self.assertEqual(output.count("Images"), 2)
        self.assertIn("2/2", output)
        self.assertIn("100%", output)
        self.assertIn("0.3000", output)
        completed = [line for line in output.splitlines() if "100%" in line]
        self.assertTrue(all(line.endswith("it/s]") for line in completed))
        for unused_column in ["GPU_mem", "box_loss", "mAP"]:
            self.assertNotIn(unused_column, output)

    def test_progress_can_close_when_training_is_interrupted(self):
        model = keras.Sequential([keras.Input(shape=(1,)), keras.layers.Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        logger = TrainingLogger(validation_size=3, batch_size=2, stream=TerminalStream())
        logger.set_model(model)
        logger.set_params({"epochs": 2, "steps": 2})
        logger.on_train_begin()
        logger.on_epoch_begin(0)
        logger.on_train_batch_end(0, {"loss": 1.0, "accuracy": 0.5})
        bar = logger.bar
        logger.close()
        logger.close()
        self.assertIsNone(logger.bar)
        self.assertTrue(bar.disable)


if __name__ == "__main__":
    unittest.main()
