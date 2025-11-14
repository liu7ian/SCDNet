"""
MTSCDNet的各个模块实现
包括:多尺度特征聚合、SAM、CAM、变化信息提取、空间特征增强等模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureAggregation(nn.Module):
    """
    多尺度特征聚合模块
    根据论文图5实现,从高层到低层逐步聚合特征
    输入:
        X1: (B, 128, H/4, W/4)
        X2: (B, 256, H/8, W/8)
        X3: (B, 512, H/16, W/16)
        X4: (B, 1024, H/32, W/32)
    输出:
        F: (B, 600, H/4, W/4)
    """
    def __init__(self):
        super().__init__()

        # X4 + X3 -> 1024通道
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # (X4+X3) + X2 -> 1024通道
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # (X4+X3+X2) + X1 -> 600通道
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(1152, 600, kernel_size=1, bias=False),
            nn.BatchNorm2d(600),
            nn.ReLU(inplace=True)
        )

    def forward(self, X1, X2, X3, X4):
        """
        Args:
            X1: (B, 128, H/4, W/4)
            X2: (B, 256, H/8, W/8)
            X3: (B, 512, H/16, W/16)
            X4: (B, 1024, H/32, W/32)
        Returns:
            F: (B, 600, H/4, W/4)
        """
        # 阶段1: X4上采样并与X3 concat
        X4_up = F.interpolate(X4, size=X3.shape[2:], mode='bilinear', align_corners=True)
        X4_3 = torch.cat([X4_up, X3], dim=1)  # (B, 1536, H/16, W/16)
        X4_3 = self.conv4_3(X4_3)  # (B, 1024, H/16, W/16)

        # 阶段2: X4_3上采样并与X2 concat
        X4_3_up = F.interpolate(X4_3, size=X2.shape[2:], mode='bilinear', align_corners=True)
        X4_3_2 = torch.cat([X4_3_up, X2], dim=1)  # (B, 1280, H/8, W/8)
        X4_3_2 = self.conv3_2(X4_3_2)  # (B, 1024, H/8, W/8)

        # 阶段3: X4_3_2上采样并与X1 concat
        X4_3_2_up = F.interpolate(X4_3_2, size=X1.shape[2:], mode='bilinear', align_corners=True)
        X4_3_2_1 = torch.cat([X4_3_2_up, X1], dim=1)  # (B, 1152, H/4, W/4)
        F_out = self.conv2_1(X4_3_2_1)  # (B, 600, H/4, W/4)

        return F_out


class ChannelAttention(nn.Module):
    """
    通道注意力模块(CAM)
    根据论文图7(a)实现
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W) 经过通道注意力加权的特征
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out * x


class SpatialAttention(nn.Module):
    """
    空间注意力模块(SAM)
    根据论文图7(b)实现
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attention_map: (B, 1, H, W) 空间注意力图
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_cat))
        return attention_map


class ASPP(nn.Module):
    """
    空洞空间金字塔池化模块(ASPP)
    用于变化信息提取模块中
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 不同膨胀率的卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 融合所有分支
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=True)

        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.conv_out(x_cat)

        return out


class ChangeInformationExtraction(nn.Module):
    """
    变化信息提取模块
    根据论文图6实现,使用两级差异特征
    输入:
        FA: (B, 600, H/4, W/4)
        FB: (B, 600, H/4, W/4)
    输出:
        SegA: (B, 150, H/4, W/4) - 用于SS子任务
        SegB: (B, 150, H/4, W/4) - 用于SS子任务
        D5: (B, 600, H/4, W/4) - 用于BCD子任务
    """
    def __init__(self):
        super().__init__()

        # Seg_head模块:共享权重
        self.seg_head = nn.Sequential(
            nn.Conv2d(600, 150, kernel_size=1, bias=False),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace=True)
        )

        # D2上采样到D4
        self.conv_d2_upsample = nn.Sequential(
            nn.Conv2d(150, 600, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(600),
            nn.ReLU(inplace=True)
        )

        # 空间注意力模块
        self.sam = SpatialAttention(kernel_size=7)

        # 通道注意力模块
        self.cam = ChannelAttention(600, reduction=16)

        # ASPP模块
        self.aspp = ASPP(600, 600)

    def forward(self, FA, FB):
        """
        Args:
            FA: (B, 600, H/4, W/4)
            FB: (B, 600, H/4, W/4)
        Returns:
            SegA: (B, 150, H/4, W/4)
            SegB: (B, 150, H/4, W/4)
            D5: (B, 600, H/4, W/4)
        """
        # 第一级差异特征 D1
        D1 = torch.abs(FA - FB)  # (B, 600, H/4, W/4)

        # 通过Seg_head获取SegA和SegB
        SegA = self.seg_head(FA)  # (B, 150, H/4, W/4)
        SegB = self.seg_head(FB)  # (B, 150, H/4, W/4)

        # 第二级差异特征 D2
        D2 = torch.abs(SegA - SegB)  # (B, 150, H/4, W/4)

        # 使用SAM生成空间注意力图S1
        S1 = self.sam(D2)  # (B, 1, H/4, W/4)

        # D3 = D1 * S1
        D3 = D1 * S1  # (B, 600, H/4, W/4)

        # D2上采样得到D4
        D4 = self.conv_d2_upsample(D2)  # (B, 600, H/4, W/4)

        # D5 = CAM(D3) + D4 (残差连接)
        D3_enhanced = self.cam(D3)  # (B, 600, H/4, W/4)
        D5_temp = D3_enhanced + D4  # (B, 600, H/4, W/4)

        # 通过ASPP得到最终的D5
        D5 = self.aspp(D5_temp)  # (B, 600, H/4, W/4)

        return SegA, SegB, D5


class SpatialFeatureEnhancement(nn.Module):
    """
    空间特征增强模块
    根据论文图8实现,使用BCD特征为SS特征提供空间先验
    """
    def __init__(self, num_classes=7):
        super().__init__()

        # Seg-Conv模块 (表1)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(150, 75, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(75),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(75, num_classes, kernel_size=1)
        )

        # Change-Conv模块 (表2)
        self.change_conv = nn.Sequential(
            nn.Conv2d(600, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1)
        )

        # 生成空间注意力权重图
        self.sam = SpatialAttention(kernel_size=7)

    def forward(self, SegA, SegB, D5, output_size=(512, 512)):
        """
        Args:
            SegA: (B, 150, H/4, W/4)
            SegB: (B, 150, H/4, W/4)
            D5: (B, 600, H/4, W/4)
            output_size: 输出尺寸(H, W)
        Returns:
            Fsa: (B, num_classes, H, W) - 时间1的语义分割结果
            Fsb: (B, num_classes, H, W) - 时间2的语义分割结果
            Fc: (B, 1, H, W) - 变化检测结果
        """
        # 从D5生成空间注意力权重图
        Ms = self.sam(D5)  # (B, 1, H/4, W/4)

        # 用Ms增强SegA和SegB
        SegA_enhanced = SegA * Ms  # (B, 150, H/4, W/4)
        SegB_enhanced = SegB * Ms  # (B, 150, H/4, W/4)
        D5_enhanced = D5 * Ms  # (B, 600, H/4, W/4)

        # 通过Seg-Conv和上采样得到语义分割结果
        Fsa = self.seg_conv(SegA_enhanced)  # (B, num_classes, H/4, W/4)
        Fsb = self.seg_conv(SegB_enhanced)  # (B, num_classes, H/4, W/4)
        Fsa = F.interpolate(Fsa, size=output_size, mode='bilinear', align_corners=True)
        Fsb = F.interpolate(Fsb, size=output_size, mode='bilinear', align_corners=True)

        # 通过Change-Conv和上采样得到变化检测结果
        Fc = self.change_conv(D5_enhanced)  # (B, 1, H/4, W/4)
        Fc = F.interpolate(Fc, size=output_size, mode='bilinear', align_corners=True)

        return Fsa, Fsb, Fc


if __name__ == '__main__':
    # 测试各个模块

    # 测试多尺度特征聚合
    print("=" * 50)
    print("测试多尺度特征聚合模块")
    msfa = MultiScaleFeatureAggregation()
    X1 = torch.randn(2, 128, 128, 128)
    X2 = torch.randn(2, 256, 64, 64)
    X3 = torch.randn(2, 512, 32, 32)
    X4 = torch.randn(2, 1024, 16, 16)
    F_out = msfa(X1, X2, X3, X4)
    print(f"输出形状: {F_out.shape}")  # 应该是 (2, 600, 128, 128)

    # 测试变化信息提取模块
    print("=" * 50)
    print("测试变化信息提取模块")
    cie = ChangeInformationExtraction()
    FA = torch.randn(2, 600, 128, 128)
    FB = torch.randn(2, 600, 128, 128)
    SegA, SegB, D5 = cie(FA, FB)
    print(f"SegA形状: {SegA.shape}")  # (2, 150, 128, 128)
    print(f"SegB形状: {SegB.shape}")  # (2, 150, 128, 128)
    print(f"D5形状: {D5.shape}")  # (2, 600, 128, 128)

    # 测试空间特征增强模块
    print("=" * 50)
    print("测试空间特征增强模块")
    sfe = SpatialFeatureEnhancement(num_classes=7)
    Fsa, Fsb, Fc = sfe(SegA, SegB, D5, output_size=(512, 512))
    print(f"Fsa形状: {Fsa.shape}")  # (2, 7, 512, 512)
    print(f"Fsb形状: {Fsb.shape}")  # (2, 7, 512, 512)
    print(f"Fc形状: {Fc.shape}")  # (2, 1, 512, 512)

    print("=" * 50)
    print("所有模块测试通过!")
