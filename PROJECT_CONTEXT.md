# Project Context

## Project Overview

面向深度学习零基础读者的极简 MNIST CNN，提供可运行代码、中文核心注释与学习说明。

## Stable Decisions

- 使用 TensorFlow/Keras 的基础层与自动求导，不依赖 PyTorch。
- 核心文件：config.py、data.py、model.py、train.py、predict.py、plots.py；训练显示独立在 training_logger.py，避免增加通用训练框架。
- 单卷积、最大池化、展平、Dropout、带 L2 的 10 类输出，共 13,610 参数。
- 训练/验证/测试分别为 50,000 / 10,000 / 10,000，测试集不参与训练或训练期间验证。
- Python 统一使用 3.11.x；使用 Miniconda 的 cv_study 命名环境，环境内以 python -m pip 安装 requirements.txt。
- .python-version 保留版本提示，Conda 通过创建命令指定 Python 3.11；训练/预测入口继续检查版本。
- README 使用标准项目文档结构，语气简明客观；详细教学原理留在 docs/tutorial.md。
- 当前默认设置：50 轮、batch 64、Adam 学习率 0.001、随机种子 42。
- best 按最低 val_loss 选取，last 为最后完整轮次；早停 restore_best_weights=False。
- 每次实验建立独立 outputs 子目录，保存配置、版本、日志、模型和图像；outputs/ 与数据缓存不提交 Git。

## Current State

- Python 3.11 限定、best/last、综合教学代码和中文教程已完成。
- 依赖清单保留未固定版本方式，新增 matplotlib。实际验证：Python 3.11.15、TensorFlow 2.21.0、Keras 3.15.1、NumPy 2.4.6、Matplotlib 3.11.2。
- README 按环境、安装、使用、结构、配置、输出、实验记录、测试和常见问题组织；docs/tutorial.md 保留 16 个训练主题并补充 Conda 环境前置步骤。
- config.py 集中超参数，train.py 保存 best/last、CSV、配置和环境版本、最终测试报告；plots.py 输出曲线、混淆矩阵与错分图片。

## Recent Sessions

### 2026-10-29：训练终端输出

- 用户要求参考 YOLO 官方风格整理训练输出，只展示当前分类代码的真实指标；已先 git pull --ff-only，结果 Already up to date。
- training_logger.py 使用 Keras 回调和 tqdm 显示对齐的训练/验证进度；终端显示 Epoch/Loss/Acc/LR 和验证 Images/Loss/Acc，重定向到文件时每轮一行，包含验证指标和耗时。
- 关闭 fit、模型保存、调度和早停的重复 verbose 输出；启动只打印必要配置，结束汇总 best/last，保留早停与学习率变化提示。详细网络结构可独立运行 summary() 查看。
- 日志只读取累计指标，表格使用本轮学习率；训练规则、best/last 和 CSV 含义保持不变。新增未锁版本依赖 tqdm，记录实际安装版本；没有引入 Ultralytics 或 PyTorch。
- 六项 unittest、依赖检查通过。真实 MNIST 子集（训练 257、验证 65、测试 131，2 轮）在普通终端与重定向输出中跑通，完成保存和绘图；有/无日志回调的逐轮指标及最终权重一致。该检查验证显示和流程，不作为模型精度实验；历史 50 轮结果保持原记录。
- 80 列终端采用自适应进度条，避免固定长度进度条截断末尾的时间/速度。尚未验证 Windows 原生终端；TensorFlow 的诊断和警告不屏蔽。


### 2026-10-29：Miniconda 与 README 文风

- 用户指定 Miniconda 管理环境，并要求 README 使用常见项目文档写法，保持专业且可供新手操作。
- 已先 git pull --ff-only，结果 Already up to date。创建命令统一为 conda create -n cv_study python=3.11 pip -y，激活后使用 python。
- README 补充解释器检查、重新打开终端后的激活步骤、常见环境问题；综合教程同步 Conda 前置步骤。
- 本次仅更新文档，命令对照官方 Conda/Miniconda 文档核验，没有创建或删除用户环境，也没有重新训练。原 Linux/WSL 实验记录保留其来源。
- 工作区原有 tests/test_training.py 删除状态不属于本次文档变更，提交时保留。

### 2026-10-29：综合教程与 best/last

- 用户要求保存 best 和 last，并把项目扩展为综合训练教程；保留原有 Python 3.11 和 pull-更新-推送要求。
- 使用现成 Keras 自动求导和回调，加入 Dropout、L2、梯度裁剪、学习率调度、早停、逐轮 CSV 和图像分析；每次实验独立目录。
- 综合版默认完整训练 50 轮：best=last=50，验证准确率 98.33%，独立测试准确率 98.22%，测试总损失 0.070368655，178 张测试错分。
- 本轮 best 恰为最后一轮，因此两个文件参数相同；没有为了制造差异而搜索训练设置。
- 可控测试强制第 2 轮验证损失最低，确认 best=2、last=3、早停后内存仍为第 3 轮、优化器与学习率记录正确。共 3 项 unittest 通过，包含数据边界/切分和错误版本提示。
- 完整训练的两模型加载、预测、更新次数和评分一致性通过；图像机器审计、人工读图和曲线灰度检查通过。额外 1 轮小样本检查只验证新增版本记录字段，不作为模型精度结果。
- 真实模型与图像保留在 outputs/20261029-134650-304701/；Git 只包含教学源码、文档和两张 SVG 示例图。

### 2026-10-29：README 核对与 Python 3.11

- 按用户要求先 git pull --ff-only，结果 Already up to date。
- 保留用户已有 EPOCHS=50 和未固定版本的 requirements；修正 README 的 Windows 启动器/激活路径、Linux/macOS 指定解释器命令及默认轮数。
- 补充三份数据标签的作用、evaluate 与 predict 的区别、训练到预测的时间顺序。
- .python-version 选择 3.11；两入口使用其他主次版本时给出提示并退出。
- Python 3.12 退出提示、Python 3.11 依赖安装与 pip check、语法检查和 README/代码一致性检查通过。
- Python 3.11.15 / Linux-WSL / CPU 完整训练 50 轮：最终训练准确率 99.89%，验证准确率 97.70%，测试准确率 97.71%，测试损失 0.1114。
- 独立预测、模型保存/重载一致性通过；优化器更新次数 39100。验证输出位于任务独立目录，既有 outputs/ 未覆盖。
- README 记录后期验证效果回落，不把当前 50 轮设置称为最优设置。

### 2026-10-29

- 完成数据处理、模型、训练、预测模块及中文教学文档。
- Python 3.12.3 / Linux-WSL / CPU 验证：数据形状、像素范围、类别、数据切分与模型概率输出通过。
- 默认完整训练 3 轮，最终验证准确率 96.42%，测试准确率 96.29%，测试损失 0.1273。
- 模型保存/重载概率一致，优化器保留 2346 次更新；从其他工作目录运行独立预测入口通过。前 10 张预测对 9 张，演示结果没有被当成全测试集准确率。
- compileall、依赖兼容性检查通过。交付仓库为 blacker0403/cv_study，提交状态以 Git 历史为准。

## Open Questions

- 当前未在 Windows 原生环境或 macOS 中运行安装与训练。

## Next Steps

- 按 README 的 data → model → train → predict 顺序学习；比较设置时使用验证集。
