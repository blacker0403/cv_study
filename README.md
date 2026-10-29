# 从 MNIST 学会一套完整的 CNN 训练流程

这是面向深度学习初学者的中文教程：使用 **Python 3.11.x + TensorFlow/Keras**，自己组合一个小 CNN，走通数据处理、训练、验证、保存、测试、预测和失败样本分析。没有 PyTorch 依赖。

建议先按本页跑通，再阅读 [综合教程：每一步为什么这样做、哪里可以调整](docs/tutorial.md)。代码关键位置都有中文注释。

## 1. 安装环境

使用 **64 位 Python 3.11.x**，允许 3.11 系列补丁更新。`.python-version` 记录版本选择，代码启动时也会检查版本；这个文件不会自动替换电脑或旧虚拟环境里的 Python。

```bash
git clone https://github.com/blacker0403/cv_study.git
cd cv_study
```

选择一组命令创建并激活环境。如果已有 `.venv` 使用其他 Python 版本，请先退出该环境并将旧目录改名保存，再创建新环境。

**Windows PowerShell：**

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows CMD：**

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS：**

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows 的 `-3.11` 属于 `py` 启动器，不能写成 `python -3.11`。找不到 `py` 或 `python3.11` 时，先安装 Python 3.11 并确认命令可用。

激活后安装依赖，`python --version` 应显示 `Python 3.11.x`：

```bash
python --version
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，可以换用 CMD，或用 `.\.venv\Scripts\python.exe` 替代命令里的 `python`，直接运行虚拟环境解释器。

普通 CPU 即可运行，不需要配置 GPU。第一次运行会下载 MNIST，缓存于用户目录的 `.keras/datasets/mnist.npz`。依赖清单当前没有固定包版本，本次验证使用的版本在实测记录中单独列出。

## 2. 训练并选择模型预测

```bash
python train.py
python predict.py
python predict.py --checkpoint last
```

默认最多训练 **50 轮**。学习率调整和早停根据验证损失工作，可能在第 50 轮之前结束。预测默认选择最近一次完整完成的实验中的 `best`；`--checkpoint last` 可以查看最后完成轮次的模型。

要指定以前的实验，把目录替换成训练时打印的真实目录：

```bash
python predict.py --run-dir outputs/你的实验目录 --checkpoint best
python predict.py --run-dir outputs/你的实验目录 --checkpoint last
```

每次训练建立独立目录，不覆盖旧实验：

```text
outputs/一次实验的时间戳/
├── best.keras              # 验证损失最低时的完整模型
├── last.keras              # 最后一个完整训练轮次的完整模型
├── config.json             # 本次参数与实际 Python/依赖版本
├── history.csv             # 逐轮损失、准确率、学习率
├── metrics.json            # 最佳轮次、最后轮次、best 的最终测试结果
├── curves.png / .svg       # 损失、准确率、学习率曲线
├── confusion_matrix.*      # best 的测试集混淆矩阵
├── mistakes.png            # 前 10 张错分图片；没有错误时不生成
└── predictions_best.png    # 运行预测后生成；last 对应 predictions_last.png
```

两个 `.keras` 文件包含相同网络结构、各自轮次学到的参数及优化器状态。**best 按验证集选出，last 保留训练终点，不能用测试集挑 best。** 最佳轮次恰好是最后一轮时，两个文件的参数相同是正常的。

模型和日志保存在本机 `outputs/`，不提交 Git。旧版的 `outputs/mnist_cnn.keras` 会保留，但新版预测入口读取新实验目录中的 best/last。

## 3. 模块如何分工

| 文件 | 职责 |
|---|---|
| `config.py` | Python 版本检查、超参数和输出目录 |
| `data.py` | 下载、归一化、补通道、训练/验证/测试划分 |
| `model.py` | 卷积、池化、展平、Dropout、全连接输出 |
| `train.py` | 编译、训练、保存回调、早停、调度与最终测试 |
| `predict.py` | 选择 best/last、输出预测和图片 |
| `plots.py` | 训练曲线、混淆矩阵和失败样本绘图 |
| `tests/test_training.py` | 检查数据划分和 best/last 等关键语义 |

阅读顺序：**config → data → model → train → predict → plots**。不会的术语在 [综合教程](docs/tutorial.md) 中逐步解释。

## 4. 先认识这几个可调参数

完整参数都在 `config.py`。第一次建议只把 `EPOCHS` 改为 3，先走通流程；之后一次只改一个设置，用验证集比较。

| 参数 | 默认值 | 调整的内容 |
|---|---:|---|
| `EPOCHS` | 50 | 最多训练轮数 |
| `BATCH_SIZE` | 64 | 每批图片数量 |
| `LEARNING_RATE` | 0.001 | 初始更新步幅 |
| `CONV_FILTERS` | 8 | 卷积核数量 |
| `KERNEL_SIZE` | 3 | 卷积窗口尺寸 |
| `DROPOUT_RATE` | 0.2 | 训练时随机丢弃的特征比例；0 为关闭 |
| `L2_FACTOR` | 0.0001 | 权重正则化强度；0 为关闭 |
| `CLIPNORM` | 1.0 | 梯度范数限制；None 为关闭 |
| `LR_PATIENCE` | 3 | 连续几轮验证损失不降后降低学习率 |
| `EARLY_STOPPING_PATIENCE` | 8 | 连续几轮验证损失不降后结束训练 |

设置 `USE_LR_SCHEDULER=False` 或 `USE_EARLY_STOPPING=False`，可以分别关闭调度或早停。初始设置用于教学，没有经过最优参数搜索。

## 5. 如何判断训练是否正常

- `loss` / `val_loss` 是训练/验证总损失，包含交叉熵和 L2 项，通常希望降低。
- `accuracy` / `val_accuracy` 是训练/验证正确率；`0.95` 表示 95%。
- 训练分数持续提高、验证损失却变大时，可能过拟合；结合曲线和失败样本判断。
- 验证准确率高于训练准确率并不一定异常：训练时启用 Dropout，而且训练指标是参数不断更新时的批次汇总。
- 最后只对验证集选出的 best 做一次完整测试，避免根据测试分数反复调参。

[综合教程](docs/tutorial.md) 包含标签的作用、损失与准确率的区别、欠拟合/过拟合/NaN 的排查、调参实验和保存后继续训练的边界。

## 6. 验证记录

2026-10-29 使用 Python 3.11.15、TensorFlow 2.21.0、Keras 3.15.1、NumPy 2.4.6、Matplotlib 3.11.2，在 Linux/WSL 下用 CPU 完整训练默认的 50 轮：

| 轮数 | 训练准确率 | 验证准确率 | 验证总损失 |
|---|---:|---:|---:|
| 1 | 87.17% | 94.00% | 0.2163 |
| 3 | 94.41% | 95.92% | 0.1528 |
| 25 | 97.72% | 98.19% | 0.0819 |
| 50 | 98.16% | 98.33% | 0.0753 |

本次 best 和 last 都来自第 50 轮：best 的独立测试准确率为 **98.22%**，总损失 **0.0704**，10,000 张测试图中有 **178** 张错误。两个模型文件均保留；最佳轮次恰好是最后一轮，所以参数相同。

学习率在第 23、41 轮开始分别使用 0.0005、0.00025；本次未触发早停。保存/重载、best/last 独立预测、数据划分、版本错误提示和图像检查均通过。这是一次实测，没有进行最优参数搜索，也没有通过单因素实验证明新增机制一定提升效果。

Windows/macOS 安装命令已对照 Python 官方文档核对，尚未在对应系统执行。历史简化版的 96.29% / 97.71% 属于其他设置，不能替代本版实测记录。

![本次实际训练曲线](docs/images/curves.svg)


关键行为测试可以自行运行：

```bash
python -m unittest discover -s tests -v
```

其中 best/last 测试人为给出验证损失 `0.6 → 0.2 → 0.4`，检查 best 保存第 2 轮、last 保存第 3 轮，且早停不把 last 恢复为 best。这个测试验证保存逻辑，不代表模型精度。

## 参考

- [Python 3.11：虚拟环境](https://docs.python.org/3.11/library/venv.html)、[Windows 启动器](https://docs.python.org/3.11/using/windows.html#python-launcher-for-windows)
- [Keras：MNIST CNN 示例](https://keras.io/examples/vision/mnist_convnet/)
- [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/)、[EarlyStopping](https://keras.io/api/callbacks/early_stopping/)、[ReduceLROnPlateau](https://keras.io/api/callbacks/reduce_lr_on_plateau/)
- [Dropout](https://keras.io/api/layers/regularization_layers/dropout/)
