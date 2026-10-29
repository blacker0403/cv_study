"""配置模块：先固定其他条件，一次只修改一个设置，观察验证集上的变化。"""

import sys
from pathlib import Path

# 所有模块先导入本文件，保证错误版本在加载第三方依赖前就给出清楚提示。
if sys.version_info[:2] != (3, 11):
    raise SystemExit("本教程仅支持 Python 3.11.x，请先激活 Python 3.11 虚拟环境。")

# 数据与复现：验证集从官方训练集中单独划出，不拿测试集帮助调参。
SEED = 42
VALIDATION_SIZE = 10000

# 网络：卷积核越多，通常容量和计算量越大；改变尺寸时注意后续形状。
CONV_FILTERS = 8
KERNEL_SIZE = 3
# Dropout 只在训练时随机丢弃部分特征，缓解过拟合；设为 0 即关闭。
DROPOUT_RATE = 0.2
# L2 惩罚过大的权重；设为 0 即关闭。过强可能使模型学不充分。
L2_FACTOR = 0.0001

# 训练：EPOCHS 是最多训练几轮，启用早停后可能提前结束。
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
# 限制梯度向量的范数，缓解异常大梯度；设为 None 可关闭。
CLIPNORM = 1.0

# 验证损失连续若干轮不改善时，降低学习率，给参数更小的更新步幅。
USE_LR_SCHEDULER = True
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 0.00001

# 早停的耐心应给降低学习率后继续学习留出时间；比较固定轮数时可关闭。
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 8

# 每次训练建立独立子目录，保留以前的模型与实验记录。
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PREDICTION_COUNT = 10
