"""
MTSCDNet训练脚本
根据论文实现训练流程
"""
import os
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models.mtscdnet import MTSCDNet
from utils.dataset import SECONDDataset, get_dataloader
from utils.losses import MTSCDLoss
from utils.metrics import SCDMetrics, BCDMetrics


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='训练MTSCDNet')

    # 数据集参数
    parser.add_argument('--data_root', type=str, default='./data/SECOND',
                       help='数据集根目录')
    parser.add_argument('--num_classes', type=int, default=7,
                       help='类别数(包括未变化类)')
    parser.add_argument('--img_size', type=int, default=512,
                       help='输入图像尺寸')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=6,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.00015,
                       help='初始学习率')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help='warm-up轮数')
    parser.add_argument('--power', type=float, default=0.9,
                       help='poly学习率的power参数')

    # 损失函数参数
    parser.add_argument('--alpha_task', type=float, default=1.0,
                       help='BCD子任务权重')
    parser.add_argument('--beta_task', type=float, default=1.0,
                       help='SS子任务权重')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='保存模型的频率(epoch)')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='评估的频率(epoch)')

    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, args):
    """
    调整学习率
    使用warm-up + poly策略
    """
    if epoch <= args.warmup_epochs:
        # Warm-up阶段:线性增长
        lr = args.lr * epoch / args.warmup_epochs
    else:
        # Poly策略
        lr = args.lr * (1 - (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)) ** args.power

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, args, writer):
    """训练一个epoch"""
    model.train()

    total_loss = 0.0
    total_change_loss = 0.0
    total_semantic_loss = 0.0

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # 获取数据
        im1 = batch['im1'].cuda()
        im2 = batch['im2'].cuda()
        label1 = batch['label1'].cuda()
        label2 = batch['label2'].cuda()
        change_label = batch['change_label'].cuda()

        # 前向传播
        outputs = model(im1, im2)

        # 计算损失
        targets = {
            'label1': label1,
            'label2': label2,
            'change_label': change_label
        }
        loss_dict = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()

        # 累积损失
        total_loss += loss_dict['total_loss'].item()
        total_change_loss += loss_dict['change_loss'].item()
        total_semantic_loss += loss_dict['semantic_loss'].item()

        # 打印信息
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch}/{args.epochs}] '
                  f'Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {loss_dict["total_loss"].item():.4f} '
                  f'Change: {loss_dict["change_loss"].item():.4f} '
                  f'Semantic: {loss_dict["semantic_loss"].item():.4f} '
                  f'Time: {elapsed:.2f}s')
            start_time = time.time()

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_change_loss = total_change_loss / len(train_loader)
    avg_semantic_loss = total_semantic_loss / len(train_loader)

    # 记录到tensorboard
    if writer is not None:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/ChangeLoss', avg_change_loss, epoch)
        writer.add_scalar('Train/SemanticLoss', avg_semantic_loss, epoch)

    return avg_loss, avg_change_loss, avg_semantic_loss


def evaluate(model, val_loader, args):
    """评估模型"""
    model.eval()

    scd_metrics = SCDMetrics(num_classes=args.num_classes)
    bcd_metrics = BCDMetrics()

    with torch.no_grad():
        for batch in val_loader:
            im1 = batch['im1'].cuda()
            im2 = batch['im2'].cuda()
            label2 = batch['label2'].cuda()
            change_label = batch['change_label'].cuda()

            # 推理
            inference_outputs = model.inference(im1, im2, threshold=0.5)

            # 更新指标
            scd_pred = inference_outputs['scd_result'].cpu()
            scd_target = label2.cpu()
            scd_metrics.update(scd_pred, scd_target)

            bcd_pred = inference_outputs['change_mask'].cpu()
            bcd_target = change_label.cpu()
            bcd_metrics.update(bcd_pred, bcd_target)

    # 获取指标
    scd_results = scd_metrics.get_metrics()
    bcd_results = bcd_metrics.get_metrics()

    print(f'\n{"="*60}')
    print('评估结果:')
    print(f'  OA: {scd_results["OA"]:.2f}%')
    print(f'  mIoU: {scd_results["mIoU"]:.2f}%')
    print(f'  SeK: {scd_results["SeK"]:.2f}%')
    print(f'  Fscd: {scd_results["Fscd"]:.2f}%')
    print(f'  Score: {scd_results["Score"]:.2f}%')
    print(f'\n  BCD Precision: {bcd_results["Precision"]:.2f}%')
    print(f'  BCD Recall: {bcd_results["Recall"]:.2f}%')
    print(f'  BCD F1: {bcd_results["F1"]:.2f}%')
    print(f'  BCD IoU: {bcd_results["IoU"]:.2f}%')
    print(f'{"="*60}\n')

    return scd_results, bcd_results


def main():
    args = get_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 创建tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'run_{timestamp}')
    writer = SummaryWriter(log_dir)

    print("=" * 60)
    print("MTSCDNet训练")
    print("=" * 60)
    print(f"数据集: {args.data_root}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"初始学习率: {args.lr}")
    print(f"Warm-up轮数: {args.warmup_epochs}")
    print("=" * 60)

    # 创建数据加载器
    print("\n加载数据集...")
    train_loader = get_dataloader(
        args.data_root, split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    val_loader = get_dataloader(
        args.data_root, split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")

    # 创建模型
    print("\n创建模型...")
    model = MTSCDNet(img_size=args.img_size, num_classes=args.num_classes)
    model = model.cuda()

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # 创建损失函数
    criterion = MTSCDLoss(
        alpha_task=args.alpha_task,
        beta_task=args.beta_task,
        alpha_tversky=0.3,
        beta_tversky=0.7
    )

    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 恢复训练
    start_epoch = 0
    best_score = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n加载检查点: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']
            print(f"从epoch {start_epoch}继续训练")
        else:
            print(f"找不到检查点: {args.resume}")

    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 调整学习率
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f'\nEpoch [{epoch}/{args.epochs}] 学习率: {lr:.6f}')
        writer.add_scalar('Train/LearningRate', lr, epoch)

        # 训练一个epoch
        avg_loss, avg_change_loss, avg_semantic_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, args, writer
        )

        print(f'Epoch [{epoch}/{args.epochs}] '
              f'平均损失: {avg_loss:.4f} '
              f'变化损失: {avg_change_loss:.4f} '
              f'语义损失: {avg_semantic_loss:.4f}')

        # 评估
        if (epoch + 1) % args.eval_freq == 0:
            print(f'\n评估 epoch {epoch}...')
            scd_results, bcd_results = evaluate(model, val_loader, args)

            # 记录到tensorboard
            writer.add_scalar('Val/OA', scd_results['OA'], epoch)
            writer.add_scalar('Val/mIoU', scd_results['mIoU'], epoch)
            writer.add_scalar('Val/SeK', scd_results['SeK'], epoch)
            writer.add_scalar('Val/Fscd', scd_results['Fscd'], epoch)
            writer.add_scalar('Val/Score', scd_results['Score'], epoch)
            writer.add_scalar('Val/BCD_F1', bcd_results['F1'], epoch)

            # 保存最佳模型
            score = scd_results['Score']
            if score > best_score:
                best_score = score
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_score': best_score,
                    'scd_results': scd_results,
                    'bcd_results': bcd_results
                }
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(checkpoint, save_path)
                print(f'保存最佳模型到: {save_path} (Score: {best_score:.2f}%)')

        # 定期保存模型
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_score
            }
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f'保存检查点到: {save_path}')

    # 训练完成
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳Score: {best_score:.2f}%")
    print("=" * 60)

    writer.close()


if __name__ == '__main__':
    main()
