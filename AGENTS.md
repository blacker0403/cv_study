# 项目约定

- 面向深度学习初学者，保持四个核心模块短小，用中文注释解释关键步骤。
- 使用 TensorFlow/Keras 基础层，不引入 PyTorch、预训练模型或通用训练框架。
- 数据处理放在 data.py，网络结构放在 model.py，训练和预测使用独立入口。
- 常用命令：`python train.py`、`python predict.py`。
- 改动后验证相关流程，区分真实训练结果与未经验证的预期。

## Project Context

Long-running project context is maintained in `PROJECT_CONTEXT.md`.
Read it before non-trivial changes and update it before finishing a task.
