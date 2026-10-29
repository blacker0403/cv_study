# 项目约定

- 面向深度学习初学者，保持四个核心模块短小，用中文注释解释关键步骤。
- Python 统一使用 3.11.x；训练和预测入口在导入第三方依赖前检查版本。
- 环境教程统一使用 Miniconda 的 cv_study 命名环境，通过该环境的 python -m pip 安装 requirements.txt。
- README 采用标准项目文档结构，措辞简明、客观；详细教学原理放在 docs/tutorial.md。
- 使用 TensorFlow/Keras 基础层，不引入 PyTorch、预训练模型或通用训练框架。
- 数据处理放在 data.py，网络结构放在 model.py，训练和预测使用独立入口。
- 可调设置集中放在 config.py，图表逻辑放在 plots.py，原理说明放在 docs/tutorial.md。
- 训练显示由 training_logger.py 负责，采用 YOLO 风格的分类指标表；不添加未计算的检测指标，不改变训练和保存规则。
- best 只按 val_loss 选取；last 保留最后完整训练轮次。早停不得恢复 best 后覆盖 last。
- 每次实验使用独立 outputs 子目录；不删除或覆盖以往实验，不提交数据或模型产物到 Git。
- 常用命令：`python train.py`、`python predict.py --checkpoint best`、`python predict.py --checkpoint last`。
- 关键检查：`python -m unittest discover -s tests -v`。
- 改动后验证相关流程，区分真实训练结果与未经验证的预期。

## Project Context

Long-running project context is maintained in `PROJECT_CONTEXT.md`.
Read it before non-trivial changes and update it before finishing a task.
