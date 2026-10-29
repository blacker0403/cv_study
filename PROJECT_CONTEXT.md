# Project Context

## Project Overview

面向深度学习零基础读者的极简 MNIST CNN，提供可运行代码、中文核心注释与学习说明。

## Stable Decisions

- 使用 TensorFlow/Keras 的基础层与自动求导，不依赖 PyTorch。
- 四个核心文件：data.py、model.py、train.py、predict.py；避免增加通用抽象。
- 单卷积、最大池化、展平、10 类输出，共 13,610 参数。
- 训练/验证/测试分别为 50,000 / 10,000 / 10,000，测试集不参与训练或训练期间验证。
- 固定默认设置：3 轮、batch 64、Adam 学习率 0.001、随机种子 42。
- outputs/ 与数据缓存不提交 Git。

## Current State

- 核心代码和中文入门 README 已完成，默认真实训练已通过。
- 依赖：TensorFlow 2.20.0、Keras 3.15.1、NumPy 2.2.6。

## Recent Sessions

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
