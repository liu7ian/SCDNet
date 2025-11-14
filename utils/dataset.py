"""
SECOND数据集加载器
根据MTSCDNet论文实现语义变化检测数据集加载
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


class SECONDDataset(Dataset):
    """
    SECOND语义变化检测数据集
    数据集结构:
    - im1: 时间1的RGB图像
    - im2: 时间2的RGB图像
    - label1_rgb: 时间1的语义标签(RGB格式)
    - label2_rgb: 时间2的语义标签(RGB格式)
    """

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

    CLASS_NAMES = ['Unchanged', 'Ground', 'Low-veg', 'Building', 'Tree', 'Playground', 'Water']

    def __init__(self, root_dir, split='train', augmentation=True, augmentation_prob=0.4):
        """
        Args:
            root_dir: SECOND数据集根目录
            split: 'train', 'val', 'test'
            augmentation: 是否使用数据增强
            augmentation_prob: 数据增强的概率
        """
        self.root_dir = root_dir
        self.split = split
        self.augmentation = augmentation and (split == 'train')
        self.augmentation_prob = augmentation_prob

        # 数据路径
        self.split_dir = os.path.join(root_dir, split)
        self.im1_dir = os.path.join(self.split_dir, 'im1')
        self.im2_dir = os.path.join(self.split_dir, 'im2')
        self.label1_dir = os.path.join(self.split_dir, 'label1_rgb')
        self.label2_dir = os.path.join(self.split_dir, 'label2_rgb')

        # 获取所有图像文件名
        self.image_files = sorted([f for f in os.listdir(self.im1_dir)
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        print(f"加载 {split} 数据集: {len(self.image_files)} 个样本")

    def __len__(self):
        return len(self.image_files)

    def rgb_to_label(self, rgb_label):
        """
        将RGB标签转换为类别索引
        Args:
            rgb_label: PIL Image (H, W, 3)
        Returns:
            label: numpy array (H, W)
        """
        rgb_array = np.array(rgb_label, dtype=np.int32)
        h, w = rgb_array.shape[:2]
        label = np.zeros((h, w), dtype=np.int64)

        for rgb, class_idx in self.RGB_TO_CLASS.items():
            mask = np.all(rgb_array == rgb, axis=-1)
            label[mask] = class_idx

        return label

    def generate_change_label(self, label1, label2):
        """
        生成二元变化检测标签
        Args:
            label1: 时间1的标签 (H, W)
            label2: 时间2的标签 (H, W)
        Returns:
            change_label: 变化标签 (H, W), 0=未变化, 1=变化
        """
        # 白色(0类)为未变化区域
        # 非白色区域如果两个时间的标签不同,则为变化
        change_label = (label1 != label2).astype(np.int64)

        # 确保未变化区域(白色)标记为0
        unchanged_mask = (label1 == 0) | (label2 == 0)
        change_label[unchanged_mask] = 0

        return change_label

    def apply_augmentation(self, im1, im2, label1, label2):
        """
        应用数据增强
        根据论文,使用随机组合的增强方法,概率为0.4
        """
        if random.random() > self.augmentation_prob:
            return im1, im2, label1, label2

        # 转换为numpy数组
        im1_np = np.array(im1)
        im2_np = np.array(im2)
        label1_np = np.array(label1)
        label2_np = np.array(label2)

        # 随机水平翻转
        if random.random() > 0.5:
            im1_np = np.fliplr(im1_np).copy()
            im2_np = np.fliplr(im2_np).copy()
            label1_np = np.fliplr(label1_np).copy()
            label2_np = np.fliplr(label2_np).copy()

        # 随机垂直翻转
        if random.random() > 0.5:
            im1_np = np.flipud(im1_np).copy()
            im2_np = np.flipud(im2_np).copy()
            label1_np = np.flipud(label1_np).copy()
            label2_np = np.flipud(label2_np).copy()

        # 随机旋转 (90, 180, 270度)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            im1_np = np.rot90(im1_np, k).copy()
            im2_np = np.rot90(im2_np, k).copy()
            label1_np = np.rot90(label1_np, k).copy()
            label2_np = np.rot90(label2_np, k).copy()

        # 颜色抖动(只应用于图像,不应用于标签)
        if random.random() > 0.5:
            # 亮度
            brightness_factor = random.uniform(0.8, 1.2)
            im1_np = np.clip(im1_np * brightness_factor, 0, 255).astype(np.uint8)
            im2_np = np.clip(im2_np * brightness_factor, 0, 255).astype(np.uint8)

            # 对比度
            contrast_factor = random.uniform(0.8, 1.2)
            im1_mean = im1_np.mean(axis=(0, 1), keepdims=True)
            im2_mean = im2_np.mean(axis=(0, 1), keepdims=True)
            im1_np = np.clip((im1_np - im1_mean) * contrast_factor + im1_mean, 0, 255).astype(np.uint8)
            im2_np = np.clip((im2_np - im2_mean) * contrast_factor + im2_mean, 0, 255).astype(np.uint8)

        # 转换回PIL Image
        im1 = Image.fromarray(im1_np)
        im2 = Image.fromarray(im2_np)
        label1 = Image.fromarray(label1_np)
        label2 = Image.fromarray(label2_np)

        return im1, im2, label1, label2

    def __getitem__(self, idx):
        """
        返回一个数据样本
        Returns:
            dict: {
                'im1': 时间1图像 (3, H, W)
                'im2': 时间2图像 (3, H, W)
                'label1': 时间1语义标签 (H, W)
                'label2': 时间2语义标签 (H, W)
                'change_label': 变化检测标签 (H, W)
                'filename': 文件名
            }
        """
        filename = self.image_files[idx]

        # 加载图像
        im1_path = os.path.join(self.im1_dir, filename)
        im2_path = os.path.join(self.im2_dir, filename)
        label1_path = os.path.join(self.label1_dir, filename)
        label2_path = os.path.join(self.label2_dir, filename)

        im1 = Image.open(im1_path).convert('RGB')
        im2 = Image.open(im2_path).convert('RGB')
        label1_rgb = Image.open(label1_path).convert('RGB')
        label2_rgb = Image.open(label2_path).convert('RGB')

        # 数据增强
        if self.augmentation:
            im1, im2, label1_rgb, label2_rgb = self.apply_augmentation(
                im1, im2, label1_rgb, label2_rgb)

        # 转换RGB标签为类别索引
        label1 = self.rgb_to_label(label1_rgb)
        label2 = self.rgb_to_label(label2_rgb)

        # 生成变化检测标签
        change_label = self.generate_change_label(label1, label2)

        # 转换为tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        im1 = transform(im1)
        im2 = transform(im2)
        label1 = torch.from_numpy(label1).long()
        label2 = torch.from_numpy(label2).long()
        change_label = torch.from_numpy(change_label).long()

        return {
            'im1': im1,
            'im2': im2,
            'label1': label1,
            'label2': label2,
            'change_label': change_label,
            'filename': filename
        }


def get_dataloader(root_dir, split='train', batch_size=6, num_workers=4, shuffle=None):
    """
    创建数据加载器
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = SECONDDataset(root_dir, split=split, augmentation=(split == 'train'))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


if __name__ == '__main__':
    # 测试数据加载器
    dataset = SECONDDataset('/path/to/SECOND', split='train')
    print(f"数据集大小: {len(dataset)}")

    sample = dataset[0]
    print(f"im1 shape: {sample['im1'].shape}")
    print(f"im2 shape: {sample['im2'].shape}")
    print(f"label1 shape: {sample['label1'].shape}")
    print(f"label2 shape: {sample['label2'].shape}")
    print(f"change_label shape: {sample['change_label'].shape}")
    print(f"label1 unique values: {torch.unique(sample['label1'])}")
    print(f"label2 unique values: {torch.unique(sample['label2'])}")
    print(f"change_label unique values: {torch.unique(sample['change_label'])}")
