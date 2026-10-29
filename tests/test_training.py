"""验证关键训练语义；人为指定的验证损失只用于测试保存逻辑，不代表实验精度。"""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import numpy as np
from tensorflow import keras

from data import load_data, preprocess_images
from train import build_callbacks


class TrainingTests(unittest.TestCase):
    def test_preprocessing_and_independent_split(self):
        images = np.full((6, 28, 28), 255, dtype="uint8")
        labels = np.arange(6)
        original = images.copy()
        result = preprocess_images(images)
        self.assertEqual(result.shape, (6, 28, 28, 1))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, 1)
        np.testing.assert_array_equal(images, original)
        raw = ((images, labels), (images[:2], np.array([8, 9])))
        with patch.object(config, "VALIDATION_SIZE", 2), patch("data.keras.datasets.mnist.load_data", return_value=raw):
            train, validation, test = load_data()
        np.testing.assert_array_equal(train[1], [0, 1, 2, 3])
        np.testing.assert_array_equal(validation[1], [4, 5])
        np.testing.assert_array_equal(test[1], [8, 9])
        self.assertFalse(np.shares_memory(train[0], validation[0]))
        for invalid_size in [0, -1, len(images)]:
            with self.subTest(invalid_size=invalid_size), patch.object(config, "VALIDATION_SIZE", invalid_size):
                with patch("data.keras.datasets.mnist.load_data", return_value=raw), self.assertRaises(ValueError):
                    load_data()

    def test_best_and_last_remain_different_after_early_stop(self):
        # 一个参数的小模型便于观察保存的是哪一轮，而不是比较识别能力。
        model = keras.Sequential([keras.Input(shape=(1,)), keras.layers.Dense(1, use_bias=False)])
        model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
        model.optimizer.build(model.trainable_variables)
        model.stop_training = False
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            config, USE_EARLY_STOPPING=True, EARLY_STOPPING_PATIENCE=1,
            USE_LR_SCHEDULER=True, LR_PATIENCE=1,
        ):
            run_dir = Path(directory)
            callbacks = build_callbacks(run_dir)
            for callback in callbacks:
                callback.set_model(model)
                callback.set_params({"epochs": 3, "steps": 1, "verbose": 0})
                callback.on_train_begin()
            # 第二轮验证损失最低，第三轮恶化，应 best=2、last=3，并触发早停。
            for epoch, val_loss in enumerate([0.6, 0.2, 0.4]):
                for callback in callbacks:
                    callback.on_epoch_begin(epoch)
                model.set_weights([np.array([[epoch + 1]], dtype="float32")])
                model.optimizer.iterations.assign(epoch + 1)
                logs = {"loss": 1.0, "val_loss": val_loss, "accuracy": 0.5, "val_accuracy": 0.5}
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs)
            for callback in callbacks:
                callback.on_train_end()
            best = keras.models.load_model(run_dir / "best.keras")
            last = keras.models.load_model(run_dir / "last.keras")
            np.testing.assert_array_equal(best.get_weights()[0], [[2]])
            np.testing.assert_array_equal(last.get_weights()[0], [[3]])
            np.testing.assert_array_equal(model.get_weights()[0], [[3]])
            self.assertEqual(int(best.optimizer.iterations), 2)
            self.assertEqual(int(last.optimizer.iterations), 3)
            self.assertTrue(model.stop_training)
            self.assertAlmostEqual(float(last.optimizer.learning_rate), 0.005, places=6)
            with (run_dir / "history.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            # 降低的学习率供下一轮使用；本轮 CSV 仍应记录 0.01。
            self.assertAlmostEqual(float(rows[-1]["learning_rate"]), 0.01, places=6)

    def test_wrong_python_version_exits_before_loading_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        for script in ["train.py", "predict.py"]:
            command = f"import sys, runpy; sys.version_info = (3, 12); runpy.run_path({script!r}, run_name='__main__')"
            result = subprocess.run([sys.executable, "-c", command], cwd=root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("本教程仅支持 Python 3.11.x", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
