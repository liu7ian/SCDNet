"""
MTSCDNet配置文件
"""
import os


class Config:
    """MTSCDNet配置类"""

    # ===== 数据集配置 =====
    # 数据集根目录(需要修改为实际路径)
    DATA_ROOT = './data/SECOND'

    # 图像尺寸
    IMG_SIZE = 512

    # 类别数(包括未变化类)
    NUM_CLASSES = 7

    # 类别名称
    CLASS_NAMES = ['Unchanged', 'Ground', 'Low-veg', 'Building', 'Tree', 'Playground', 'Water']

    # RGB标签映射到类别索引
    RGB_TO_CLASS = {
        (255, 255, 255): 0,  # Unchanged - 白色
        (128, 128, 128): 1,  # Ground - 灰色
        (0, 128, 0): 2,      # Low-veg - 深绿色
        (128, 0, 0): 3,      # Building - 深红色
        (0, 255, 0): 4,      # Tree - 亮绿色
        (255, 0, 0): 5,      # Playground - 红色
        (0, 0, 255): 6,      # Water - 蓝色
    }

    # ===== 训练配置 =====
    # 训练轮数
    EPOCHS = 100

    # 批次大小
    BATCH_SIZE = 6

    # 初始学习率(根据论文)
    LEARNING_RATE = 0.00015

    # Warm-up轮数
    WARMUP_EPOCHS = 20

    # Poly学习率的power参数
    POWER = 0.9

    # 优化器权重衰减
    WEIGHT_DECAY = 0.01

    # ===== 损失函数配置 =====
    # BCD子任务权重
    ALPHA_TASK = 1.0

    # SS子任务权重
    BETA_TASK = 1.0

    # Tversky损失参数
    ALPHA_TVERSKY = 0.3  # 假阴性权重
    BETA_TVERSKY = 0.7   # 假阳性权重

    # ===== 模型配置 =====
    # Swin Transformer配置(根据论文)
    SWIN_DEPTHS = [2, 2, 18, 2]
    SWIN_NUM_HEADS = [4, 8, 16, 32]
    SWIN_WINDOW_SIZE = 7
    SWIN_EMBED_DIM = 128
    SWIN_MLP_RATIO = 4.0
    SWIN_QKV_BIAS = True
    SWIN_DROP_RATE = 0.0
    SWIN_ATTN_DROP_RATE = 0.0

    # 预训练权重路径(可选)
    PRETRAINED_PATH = None

    # ===== 数据增强配置 =====
    # 数据增强概率
    AUGMENTATION_PROB = 0.4

    # ImageNet归一化参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ===== 保存和日志配置 =====
    # 检查点保存目录
    CHECKPOINT_DIR = './checkpoints'

    # 日志保存目录
    LOG_DIR = './logs'

    # 结果保存目录
    RESULT_DIR = './results'

    # 保存模型的频率(epoch)
    SAVE_FREQ = 10

    # 评估的频率(epoch)
    EVAL_FREQ = 1

    # ===== 其他配置 =====
    # 数据加载器工作进程数
    NUM_WORKERS = 4

    # 变化检测阈值
    THRESHOLD = 0.5

    # 是否使用GPU
    USE_CUDA = True

    # 随机种子
    SEED = 42

    @classmethod
    def from_args(cls, args):
        """从argparse参数创建配置"""
        config = cls()

        # 更新配置
        for key, value in vars(args).items():
            key_upper = key.upper()
            if hasattr(config, key_upper):
                setattr(config, key_upper, value)

        return config

    def __repr__(self):
        """打印配置"""
        config_str = "MTSCDNet配置:\n"
        config_str += "=" * 60 + "\n"

        for key, value in vars(self.__class__).items():
            if not key.startswith('_') and not callable(value):
                config_str += f"{key:25s}: {value}\n"

        config_str += "=" * 60
        return config_str


# 默认配置实例
default_config = Config()


if __name__ == '__main__':
    # 测试配置
    config = Config()
    print(config)
