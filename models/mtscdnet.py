"""
MTSCD-Net完整模型实现
基于多任务学习的语义变化检测网络
论文: MTSCD-Net: A network based on multi-task learning for semantic change detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swin_transformer import SwinTransformer
from models.modules import (
    MultiScaleFeatureAggregation,
    ChangeInformationExtraction,
    SpatialFeatureEnhancement
)


class MTSCDNet(nn.Module):
    """
    MTSCD-Net: 多任务语义变化检测网络

    网络结构:
    1. Siamese Swin Transformer编码器(权重共享)
    2. 多尺度特征聚合模块
    3. 变化信息提取模块
    4. 空间特征增强模块
    5. 双分支解码器(SS子任务和BCD子任务)
    """
    def __init__(self, img_size=512, num_classes=7, pretrained=None):
        """
        Args:
            img_size: 输入图像尺寸
            num_classes: 语义类别数(包括未变化类)
            pretrained: 预训练权重路径
        """
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        # Siamese语义感知编码器(基于Swin Transformer,权重共享)
        # 根据论文配置: depths=[2, 2, 18, 2]
        self.encoder = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.
        )

        # 多尺度特征聚合模块
        self.feature_aggregation = MultiScaleFeatureAggregation()

        # 变化信息提取模块
        self.change_extraction = ChangeInformationExtraction()

        # 空间特征增强模块
        self.spatial_enhancement = SpatialFeatureEnhancement(num_classes=num_classes)

        # 如果有预训练权重,加载
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained_path):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 只加载编码器部分
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    encoder_state_dict[k.replace('encoder.', '')] = v

            if len(encoder_state_dict) > 0:
                self.encoder.load_state_dict(encoder_state_dict, strict=False)
                print(f"成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")

    def forward(self, im1, im2):
        """
        前向传播
        Args:
            im1: 时间1图像 (B, 3, H, W)
            im2: 时间2图像 (B, 3, H, W)
        Returns:
            outputs: dict {
                'seg1': 时间1的语义分割 (B, num_classes, H, W)
                'seg2': 时间2的语义分割 (B, num_classes, H, W)
                'change': 变化检测 (B, 1, H, W)
            }
        """
        # 1. Siamese编码器提取特征(权重共享)
        features1 = self.encoder(im1)  # [X1_1, X2_1, X3_1, X4_1]
        features2 = self.encoder(im2)  # [X1_2, X2_2, X3_2, X4_2]

        # 2. 多尺度特征聚合
        FA = self.feature_aggregation(features1[0], features1[1], features1[2], features1[3])
        FB = self.feature_aggregation(features2[0], features2[1], features2[2], features2[3])

        # 3. 变化信息提取
        SegA, SegB, D5 = self.change_extraction(FA, FB)

        # 4. 空间特征增强并输出结果
        Fsa, Fsb, Fc = self.spatial_enhancement(SegA, SegB, D5, output_size=(self.img_size, self.img_size))

        outputs = {
            'seg1': Fsa,      # (B, num_classes, H, W)
            'seg2': Fsb,      # (B, num_classes, H, W)
            'change': Fc      # (B, 1, H, W)
        }

        return outputs

    def inference(self, im1, im2, threshold=0.5):
        """
        推理模式
        Args:
            im1: 时间1图像 (B, 3, H, W)
            im2: 时间2图像 (B, 3, H, W)
            threshold: 变化检测的阈值
        Returns:
            outputs: dict {
                'seg1': 时间1的语义类别 (B, H, W)
                'seg2': 时间2的语义类别 (B, H, W)
                'change_prob': 变化概率图 (B, H, W)
                'change_mask': 二值变化掩码 (B, H, W)
                'scd_result': 语义变化检测结果 (B, H, W)
            }
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(im1, im2)

            # 语义分割结果
            seg1 = torch.argmax(outputs['seg1'], dim=1)  # (B, H, W)
            seg2 = torch.argmax(outputs['seg2'], dim=1)  # (B, H, W)

            # 变化检测结果
            change_prob = torch.sigmoid(outputs['change']).squeeze(1)  # (B, H, W)
            change_mask = (change_prob > threshold).long()  # (B, H, W)

            # 语义变化检测结果:使用change_mask来掩码语义分割结果
            # 未变化区域设为0,变化区域保留语义类别
            scd_result = seg2.clone()
            scd_result[change_mask == 0] = 0  # 未变化区域设为0(Unchanged类)

            return {
                'seg1': seg1,
                'seg2': seg2,
                'change_prob': change_prob,
                'change_mask': change_mask,
                'scd_result': scd_result
            }


def build_mtscdnet(config):
    """
    构建MTSCD-Net模型
    Args:
        config: 配置字典或配置对象
    Returns:
        model: MTSCDNet模型
    """
    model = MTSCDNet(
        img_size=getattr(config, 'img_size', 512),
        num_classes=getattr(config, 'num_classes', 7),
        pretrained=getattr(config, 'pretrained', None)
    )
    return model


if __name__ == '__main__':
    # 测试MTSCDNet模型
    print("=" * 60)
    print("测试MTSCDNet模型")
    print("=" * 60)

    model = MTSCDNet(img_size=512, num_classes=7)
    model.eval()

    # 创建测试输入
    im1 = torch.randn(2, 3, 512, 512)
    im2 = torch.randn(2, 3, 512, 512)

    # 测试训练模式
    print("\n训练模式:")
    outputs = model(im1, im2)
    print(f"seg1 shape: {outputs['seg1'].shape}")  # (2, 7, 512, 512)
    print(f"seg2 shape: {outputs['seg2'].shape}")  # (2, 7, 512, 512)
    print(f"change shape: {outputs['change'].shape}")  # (2, 1, 512, 512)

    # 测试推理模式
    print("\n推理模式:")
    inference_outputs = model.inference(im1, im2, threshold=0.5)
    print(f"seg1 shape: {inference_outputs['seg1'].shape}")  # (2, 512, 512)
    print(f"seg2 shape: {inference_outputs['seg2'].shape}")  # (2, 512, 512)
    print(f"change_prob shape: {inference_outputs['change_prob'].shape}")  # (2, 512, 512)
    print(f"change_mask shape: {inference_outputs['change_mask'].shape}")  # (2, 512, 512)
    print(f"scd_result shape: {inference_outputs['scd_result'].shape}")  # (2, 512, 512)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("MTSCDNet模型测试通过!")
    print("=" * 60)
