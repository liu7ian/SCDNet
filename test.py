"""
MTSCDNet测试/推理脚本
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.mtscdnet import MTSCDNet
from utils.dataset import SECONDDataset, get_dataloader
from utils.metrics import SCDMetrics, BCDMetrics


# 类别颜色映射(用于可视化)
CLASS_COLORS = {
    0: (255, 255, 255),  # Unchanged - 白色
    1: (128, 128, 128),  # Ground - 灰色
    2: (0, 128, 0),      # Low-veg - 深绿色
    3: (128, 0, 0),      # Building - 深红色
    4: (0, 255, 0),      # Tree - 亮绿色
    5: (255, 0, 0),      # Playground - 红色
    6: (0, 0, 255),      # Water - 蓝色
}

CLASS_NAMES = ['Unchanged', 'Ground', 'Low-veg', 'Building', 'Tree', 'Playground', 'Water']


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='测试MTSCDNet')

    # 数据集参数
    parser.add_argument('--data_root', type=str, default='./data/SECOND',
                       help='数据集根目录')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='测试哪个数据集划分')
    parser.add_argument('--num_classes', type=int, default=7,
                       help='类别数(包括未变化类)')
    parser.add_argument('--img_size', type=int, default=512,
                       help='输入图像尺寸')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='变化检测阈值')

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存预测结果')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='结果保存目录')

    args = parser.parse_args()
    return args


def label_to_rgb(label, colors=CLASS_COLORS):
    """
    将类别标签转换为RGB图像
    Args:
        label: (H, W) numpy数组
        colors: 类别颜色映射字典
    Returns:
        rgb: (H, W, 3) numpy数组
    """
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        mask = label == class_id
        rgb[mask] = color

    return rgb


def visualize_results(im1, im2, label1, label2, pred_scd, target_scd, change_mask, filename, save_path):
    """
    可视化结果
    Args:
        im1, im2: 输入图像
        label1, label2: 真实标签
        pred_scd: 预测的语义变化检测结果
        target_scd: 真实的语义变化检测结果
        change_mask: 变化掩码
        filename: 文件名
        save_path: 保存路径
    """
    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im1_np = im1.cpu().numpy().transpose(1, 2, 0)
    im2_np = im2.cpu().numpy().transpose(1, 2, 0)
    im1_np = (im1_np * std + mean) * 255
    im2_np = (im2_np * std + mean) * 255
    im1_np = np.clip(im1_np, 0, 255).astype(np.uint8)
    im2_np = np.clip(im2_np, 0, 255).astype(np.uint8)

    # 转换标签为RGB
    label1_rgb = label_to_rgb(label1.cpu().numpy())
    label2_rgb = label_to_rgb(label2.cpu().numpy())
    pred_scd_rgb = label_to_rgb(pred_scd.cpu().numpy())
    target_scd_rgb = label_to_rgb(target_scd.cpu().numpy())

    # 变化掩码可视化
    change_mask_rgb = np.zeros_like(im1_np)
    change_mask_rgb[change_mask.cpu().numpy() == 1] = [255, 0, 0]  # 红色表示变化

    # 创建图像
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(im1_np)
    axes[0, 0].set_title('T1 Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(im2_np)
    axes[0, 1].set_title('T2 Image')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(label1_rgb)
    axes[0, 2].set_title('T1 Label (GT)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(label2_rgb)
    axes[0, 3].set_title('T2 Label (GT)')
    axes[0, 3].axis('off')

    axes[1, 0].imshow(change_mask_rgb)
    axes[1, 0].set_title('Change Mask (Pred)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(target_scd_rgb)
    axes[1, 1].set_title('SCD Result (GT)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pred_scd_rgb)
    axes[1, 2].set_title('SCD Result (Pred)')
    axes[1, 2].axis('off')

    # 对比图:绿色为正确,红色为错误
    diff_rgb = np.zeros_like(im1_np)
    correct = (pred_scd.cpu().numpy() == target_scd.cpu().numpy())
    diff_rgb[correct] = [0, 255, 0]  # 绿色
    diff_rgb[~correct] = [255, 0, 0]  # 红色

    axes[1, 3].imshow(diff_rgb)
    axes[1, 3].set_title('Comparison (Green=Correct, Red=Wrong)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test(model, test_loader, args):
    """测试模型"""
    model.eval()

    scd_metrics = SCDMetrics(num_classes=args.num_classes)
    bcd_metrics = BCDMetrics()

    # 创建结果保存目录
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        pred_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

    print("\n开始测试...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            im1 = batch['im1'].cuda()
            im2 = batch['im2'].cuda()
            label1 = batch['label1'].cuda()
            label2 = batch['label2'].cuda()
            change_label = batch['change_label'].cuda()
            filenames = batch['filename']

            # 推理
            inference_outputs = model.inference(im1, im2, threshold=args.threshold)

            # 更新指标
            scd_pred = inference_outputs['scd_result'].cpu()
            scd_target = label2.cpu()
            scd_metrics.update(scd_pred, scd_target)

            bcd_pred = inference_outputs['change_mask'].cpu()
            bcd_target = change_label.cpu()
            bcd_metrics.update(bcd_pred, bcd_target)

            # 保存结果
            if args.save_results:
                for i in range(im1.size(0)):
                    filename = filenames[i]
                    base_name = os.path.splitext(filename)[0]

                    # 保存可视化结果
                    vis_path = os.path.join(vis_dir, f'{base_name}_result.png')
                    visualize_results(
                        im1[i], im2[i],
                        label1[i], label2[i],
                        scd_pred[i], scd_target[i],
                        bcd_pred[i],
                        filename, vis_path
                    )

                    # 保存预测结果(RGB格式)
                    pred_rgb = label_to_rgb(scd_pred[i].numpy())
                    pred_img = Image.fromarray(pred_rgb)
                    pred_img.save(os.path.join(pred_dir, f'{base_name}_pred.png'))

    # 获取指标
    scd_results = scd_metrics.get_metrics()
    bcd_results = bcd_metrics.get_metrics()

    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    print("\n语义变化检测指标:")
    print(f"  OA:    {scd_results['OA']:.2f}%")
    print(f"  mIoU:  {scd_results['mIoU']:.2f}%")
    print(f"  SeK:   {scd_results['SeK']:.2f}%")
    print(f"  Fscd:  {scd_results['Fscd']:.2f}%")
    print(f"  Score: {scd_results['Score']:.2f}%")

    print("\n每类IoU:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {scd_results['IoU_per_class'][i]:.2f}%")

    print("\n二元变化检测指标:")
    print(f"  Precision: {bcd_results['Precision']:.2f}%")
    print(f"  Recall:    {bcd_results['Recall']:.2f}%")
    print(f"  F1:        {bcd_results['F1']:.2f}%")
    print(f"  IoU:       {bcd_results['IoU']:.2f}%")

    print("\n" + "=" * 60)

    # 保存结果到文件
    if args.save_results:
        result_file = os.path.join(args.output_dir, 'test_results.txt')
        with open(result_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("测试结果\n")
            f.write("=" * 60 + "\n\n")
            f.write("语义变化检测指标:\n")
            f.write(f"  OA:    {scd_results['OA']:.2f}%\n")
            f.write(f"  mIoU:  {scd_results['mIoU']:.2f}%\n")
            f.write(f"  SeK:   {scd_results['SeK']:.2f}%\n")
            f.write(f"  Fscd:  {scd_results['Fscd']:.2f}%\n")
            f.write(f"  Score: {scd_results['Score']:.2f}%\n\n")
            f.write("每类IoU:\n")
            for i, name in enumerate(CLASS_NAMES):
                f.write(f"  {name:12s}: {scd_results['IoU_per_class'][i]:.2f}%\n")
            f.write("\n二元变化检测指标:\n")
            f.write(f"  Precision: {bcd_results['Precision']:.2f}%\n")
            f.write(f"  Recall:    {bcd_results['Recall']:.2f}%\n")
            f.write(f"  F1:        {bcd_results['F1']:.2f}%\n")
            f.write(f"  IoU:       {bcd_results['IoU']:.2f}%\n")
            f.write("=" * 60 + "\n")

        print(f"\n结果已保存到: {args.output_dir}")

    return scd_results, bcd_results


def main():
    args = get_args()

    print("=" * 60)
    print("MTSCDNet测试")
    print("=" * 60)
    print(f"数据集: {args.data_root}")
    print(f"划分: {args.split}")
    print(f"检查点: {args.checkpoint}")
    print(f"阈值: {args.threshold}")
    print("=" * 60)

    # 创建数据加载器
    print("\n加载数据集...")
    test_loader = get_dataloader(
        args.data_root, split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    print(f"测试集样本数: {len(test_loader.dataset)}")

    # 创建模型
    print("\n加载模型...")
    model = MTSCDNet(img_size=args.img_size, num_classes=args.num_classes)
    model = model.cuda()

    # 加载检查点
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"成功加载检查点: {args.checkpoint}")
    else:
        print(f"错误: 找不到检查点文件: {args.checkpoint}")
        return

    # 测试
    scd_results, bcd_results = test(model, test_loader, args)


if __name__ == '__main__':
    main()
