"""
MTSCDNet损失函数
根据论文2.5节实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky Loss
    论文中使用参数: α21=0.3, α22=0.7
    用于增加假阴性的权重,减少样本不平衡的影响
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        """
        Args:
            alpha: 控制假阴性的权重
            beta: 控制假阳性的权重
            smooth: 平滑项,避免除零
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (B, 1, H, W), 经过sigmoid
            target: 目标值 (B, 1, H, W) 或 (B, H, W), 值为0或1
        Returns:
            loss: Tversky损失
        """
        # 确保pred经过sigmoid
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # 确保target是正确的形状
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # 展平
        pred = pred.view(-1)
        target = target.view(-1).float()

        # 计算TP, FP, FN
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()

        # Tversky指数
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        # Tversky损失
        loss = 1 - tversky_index

        return loss


class BCELoss(nn.Module):
    """
    二元交叉熵损失
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (B, 1, H, W), logits(未经过sigmoid)
            target: 目标值 (B, 1, H, W) 或 (B, H, W), 值为0或1
        Returns:
            loss: BCE损失
        """
        # 确保target是正确的形状
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        target = target.float()
        loss = self.bce(pred, target)

        return loss


class ChangeLoss(nn.Module):
    """
    变化检测子任务的损失函数
    L_c = BCE + Tversky
    """
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.bce_loss = BCELoss()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (B, 1, H, W), logits
            target: 目标值 (B, H, W) 或 (B, 1, H, W), 值为0或1
        Returns:
            loss: 变化检测损失
        """
        bce = self.bce_loss(pred, target)

        # Tversky需要sigmoid后的预测值
        pred_sigmoid = torch.sigmoid(pred)
        tversky = self.tversky_loss(pred_sigmoid, target)

        loss = bce + tversky

        return loss


class SemanticLoss(nn.Module):
    """
    语义分割子任务的损失函数
    L_s = 交叉熵损失
    """
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (B, C, H, W), logits
            target: 目标值 (B, H, W), 类别索引
        Returns:
            loss: 语义分割损失
        """
        loss = self.ce_loss(pred, target)
        return loss


class MTSCDLoss(nn.Module):
    """
    MTSCDNet总损失函数
    根据论文方程(7):
    L_total = α * L_c + β * L_s

    其中:
    - L_c: 变化检测子任务损失 (BCE + Tversky)
    - L_s: 语义分割子任务损失 (交叉熵)
    - α, β: 两个子任务的损失权重
    """
    def __init__(self, alpha_task=1.0, beta_task=1.0, alpha_tversky=0.3, beta_tversky=0.7):
        """
        Args:
            alpha_task: BCD子任务的权重
            beta_task: SS子任务的权重
            alpha_tversky: Tversky损失中假阴性的权重
            beta_tversky: Tversky损失中假阳性的权重
        """
        super().__init__()
        self.alpha_task = alpha_task
        self.beta_task = beta_task

        self.change_loss = ChangeLoss(alpha=alpha_tversky, beta=beta_tversky)
        self.semantic_loss = SemanticLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: 模型输出字典 {
                'seg1': (B, C, H, W),
                'seg2': (B, C, H, W),
                'change': (B, 1, H, W)
            }
            targets: 目标字典 {
                'label1': (B, H, W),
                'label2': (B, H, W),
                'change_label': (B, H, W)
            }
        Returns:
            loss_dict: 损失字典 {
                'total_loss': 总损失,
                'change_loss': 变化检测损失,
                'seg1_loss': 时间1语义分割损失,
                'seg2_loss': 时间2语义分割损失,
                'semantic_loss': 总语义分割损失
            }
        """
        # 变化检测损失
        change_loss = self.change_loss(outputs['change'], targets['change_label'])

        # 语义分割损失(两个时间的损失相加)
        seg1_loss = self.semantic_loss(outputs['seg1'], targets['label1'])
        seg2_loss = self.semantic_loss(outputs['seg2'], targets['label2'])
        semantic_loss = seg1_loss + seg2_loss

        # 总损失
        total_loss = self.alpha_task * change_loss + self.beta_task * semantic_loss

        loss_dict = {
            'total_loss': total_loss,
            'change_loss': change_loss,
            'seg1_loss': seg1_loss,
            'seg2_loss': seg2_loss,
            'semantic_loss': semantic_loss
        }

        return loss_dict


if __name__ == '__main__':
    # 测试损失函数
    print("=" * 60)
    print("测试损失函数")
    print("=" * 60)

    batch_size = 4
    num_classes = 7
    H, W = 512, 512

    # 创建模拟数据
    outputs = {
        'seg1': torch.randn(batch_size, num_classes, H, W),
        'seg2': torch.randn(batch_size, num_classes, H, W),
        'change': torch.randn(batch_size, 1, H, W)
    }

    targets = {
        'label1': torch.randint(0, num_classes, (batch_size, H, W)),
        'label2': torch.randint(0, num_classes, (batch_size, H, W)),
        'change_label': torch.randint(0, 2, (batch_size, H, W))
    }

    # 测试Tversky损失
    print("\n测试Tversky损失:")
    tversky_loss_fn = TverskyLoss()
    pred = torch.sigmoid(torch.randn(batch_size, 1, H, W))
    target = torch.randint(0, 2, (batch_size, H, W))
    tversky_loss = tversky_loss_fn(pred, target)
    print(f"Tversky Loss: {tversky_loss.item():.4f}")

    # 测试BCE损失
    print("\n测试BCE损失:")
    bce_loss_fn = BCELoss()
    pred = torch.randn(batch_size, 1, H, W)
    target = torch.randint(0, 2, (batch_size, H, W))
    bce_loss = bce_loss_fn(pred, target)
    print(f"BCE Loss: {bce_loss.item():.4f}")

    # 测试变化检测损失
    print("\n测试变化检测损失:")
    change_loss_fn = ChangeLoss()
    change_loss = change_loss_fn(outputs['change'], targets['change_label'])
    print(f"Change Loss: {change_loss.item():.4f}")

    # 测试语义分割损失
    print("\n测试语义分割损失:")
    semantic_loss_fn = SemanticLoss()
    seg_loss = semantic_loss_fn(outputs['seg1'], targets['label1'])
    print(f"Semantic Loss: {seg_loss.item():.4f}")

    # 测试MTSCDNet总损失
    print("\n测试MTSCDNet总损失:")
    mtscd_loss_fn = MTSCDLoss(alpha_task=1.0, beta_task=1.0)
    loss_dict = mtscd_loss_fn(outputs, targets)

    print(f"Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Change Loss: {loss_dict['change_loss'].item():.4f}")
    print(f"Seg1 Loss: {loss_dict['seg1_loss'].item():.4f}")
    print(f"Seg2 Loss: {loss_dict['seg2_loss'].item():.4f}")
    print(f"Semantic Loss: {loss_dict['semantic_loss'].item():.4f}")

    print("\n" + "=" * 60)
    print("损失函数测试通过!")
    print("=" * 60)
