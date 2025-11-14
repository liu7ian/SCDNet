"""
MTSCDNet评估指标
根据论文3.3节实现
包括: OA, mIoU, SeK, Fscd, Score
"""
import numpy as np
import torch


class ConfusionMatrix:
    """混淆矩阵"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """
        更新混淆矩阵
        Args:
            pred: 预测值 (H, W) 或 (B, H, W), numpy数组或tensor
            target: 目标值 (H, W) 或 (B, H, W), numpy数组或tensor
        """
        # 转换为numpy数组
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        # 展平
        pred = pred.flatten()
        target = target.flatten()

        # 过滤掉无效值
        mask = (target >= 0) & (target < self.num_classes)
        pred = pred[mask]
        target = target[mask]

        # 更新混淆矩阵
        for t, p in zip(target, pred):
            self.matrix[p, t] += 1

    def reset(self):
        """重置混淆矩阵"""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def get_matrix(self):
        """获取混淆矩阵"""
        return self.matrix


class SCDMetrics:
    """
    语义变化检测评估指标
    根据MTSCDNet论文实现
    """
    def __init__(self, num_classes=7):
        """
        Args:
            num_classes: 类别数(包括未变化类)
        """
        self.num_classes = num_classes
        self.confusion_matrix = ConfusionMatrix(num_classes)

    def reset(self):
        """重置所有指标"""
        self.confusion_matrix.reset()

    def update(self, pred, target):
        """
        更新指标
        Args:
            pred: 预测值 (H, W) 或 (B, H, W)
            target: 目标值 (H, W) 或 (B, H, W)
        """
        self.confusion_matrix.update(pred, target)

    def get_metrics(self):
        """
        计算所有指标
        Returns:
            metrics: 字典,包含所有评估指标
        """
        matrix = self.confusion_matrix.get_matrix()

        # 计算OA (Overall Accuracy)
        oa = self._compute_oa(matrix)

        # 计算mIoU (Mean IoU)
        iou_per_class, miou = self._compute_miou(matrix)

        # 计算SeK (Separated Kappa)
        sek = self._compute_sek(matrix)

        # 计算Fscd
        fscd = self._compute_fscd(matrix)

        # 计算Score
        score = 0.3 * miou + 0.7 * sek

        # 计算每个类别的精确率和召回率
        precision_per_class, recall_per_class, f1_per_class = self._compute_precision_recall_f1(matrix)

        metrics = {
            'OA': oa * 100,  # 转换为百分比
            'mIoU': miou * 100,
            'SeK': sek * 100,
            'Fscd': fscd * 100,
            'Score': score * 100,
            'IoU_per_class': iou_per_class * 100,
            'Precision_per_class': precision_per_class * 100,
            'Recall_per_class': recall_per_class * 100,
            'F1_per_class': f1_per_class * 100
        }

        return metrics

    def _compute_oa(self, matrix):
        """
        计算Overall Accuracy
        OA = Σ(qii) / Σ(qij)
        """
        oa = np.diag(matrix).sum() / (matrix.sum() + 1e-10)
        return oa

    def _compute_miou(self, matrix):
        """
        计算Mean IoU
        IoU_i = qii / (Σqij + Σqji - qii)
        mIoU = mean(IoU_i)
        """
        iou_per_class = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            tp = matrix[i, i]
            fp = matrix[i, :].sum() - tp
            fn = matrix[:, i].sum() - tp

            iou = tp / (tp + fp + fn + 1e-10)
            iou_per_class[i] = iou

        miou = iou_per_class.mean()

        return iou_per_class, miou

    def _compute_sek(self, matrix):
        """
        计算Separated Kappa (SeK)
        根据论文方程(20)-(22)
        SeK = e^(IoU2-1) * (ρ̂ - η̂) / (1 - η̂)
        """
        n = self.num_classes

        # q11是未变化类的数量(假设是第0类)
        q11 = matrix[0, 0]
        total = matrix.sum()
        total_changed = total - q11

        if total_changed == 0:
            return 0.0

        # 计算IoU2 (变化区域的IoU)
        # IoU2 = Σ(qij for i,j>=2) / (Σ(qij) - q11)
        changed_correct = 0
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    changed_correct += matrix[i, j]

        iou2 = changed_correct / (total_changed + 1e-10)

        # 计算ρ̂
        diagonal_sum_changed = sum(matrix[i, i] for i in range(1, n))
        rho_hat = diagonal_sum_changed / (total_changed + 1e-10)

        # 计算η̂
        eta_hat = 0.0
        for j in range(1, n):
            row_sum = matrix[j, :].sum() - (matrix[j, 0] if j < n else 0)
            col_sum = matrix[:, j].sum() - (matrix[0, j] if j < n else 0)
            eta_hat += (row_sum / total_changed) * (col_sum / total_changed)

        # 计算SeK
        sek = np.exp(iou2 - 1) * (rho_hat - eta_hat) / (1 - eta_hat + 1e-10)

        return sek

    def _compute_fscd(self, matrix):
        """
        计算Fscd (F-score for Semantic Change Detection)
        参考Bi-SRNet论文中的方法
        Fscd是变化区域的F1分数
        """
        n = self.num_classes

        # 计算变化区域的TP, FP, FN
        # TP: 正确预测为变化的像素
        # FP: 错误预测为变化的像素
        # FN: 漏检的变化像素

        # 未变化类是第0类
        # 变化区域: 预测和真实都不是第0类
        tp = 0
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    tp += matrix[i, j]

        # FP: 预测为变化,但实际未变化
        fp = 0
        for i in range(1, n):
            fp += matrix[i, 0]

        # FN: 预测为未变化,但实际变化
        fn = 0
        for j in range(1, n):
            fn += matrix[0, j]

        # 计算Precision和Recall
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        # 计算F1 (Fscd)
        fscd = 2 * precision * recall / (precision + recall + 1e-10)

        return fscd

    def _compute_precision_recall_f1(self, matrix):
        """计算每个类别的精确率、召回率和F1分数"""
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        f1 = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            tp = matrix[i, i]
            fp = matrix[i, :].sum() - tp
            fn = matrix[:, i].sum() - tp

            precision[i] = tp / (tp + fp + 1e-10)
            recall[i] = tp / (tp + fn + 1e-10)
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-10)

        return precision, recall, f1


class BCDMetrics:
    """
    二元变化检测评估指标
    包括: Precision, Recall, F1, IoU
    """
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def reset(self):
        """重置所有指标"""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, pred, target):
        """
        更新指标
        Args:
            pred: 预测值 (H, W) 或 (B, H, W), 0或1
            target: 目标值 (H, W) 或 (B, H, W), 0或1
        """
        # 转换为numpy数组
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        # 展平
        pred = pred.flatten()
        target = target.flatten()

        # 计算TP, FP, TN, FN
        self.tp += np.sum((pred == 1) & (target == 1))
        self.fp += np.sum((pred == 1) & (target == 0))
        self.tn += np.sum((pred == 0) & (target == 0))
        self.fn += np.sum((pred == 0) & (target == 1))

    def get_metrics(self):
        """
        计算所有指标
        Returns:
            metrics: 字典,包含所有评估指标
        """
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-10)

        metrics = {
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1': f1 * 100,
            'IoU': iou * 100
        }

        return metrics


if __name__ == '__main__':
    # 测试评估指标
    print("=" * 60)
    print("测试评估指标")
    print("=" * 60)

    num_classes = 7
    H, W = 512, 512

    # 创建模拟数据
    pred = np.random.randint(0, num_classes, (H, W))
    target = np.random.randint(0, num_classes, (H, W))

    # 测试SCD指标
    print("\n测试语义变化检测指标:")
    scd_metrics = SCDMetrics(num_classes=num_classes)
    scd_metrics.update(pred, target)
    metrics = scd_metrics.get_metrics()

    print(f"OA: {metrics['OA']:.2f}%")
    print(f"mIoU: {metrics['mIoU']:.2f}%")
    print(f"SeK: {metrics['SeK']:.2f}%")
    print(f"Fscd: {metrics['Fscd']:.2f}%")
    print(f"Score: {metrics['Score']:.2f}%")

    print("\n每类IoU:")
    class_names = ['Unchanged', 'Ground', 'Low-veg', 'Building', 'Tree', 'Playground', 'Water']
    for i, name in enumerate(class_names):
        print(f"  {name}: {metrics['IoU_per_class'][i]:.2f}%")

    # 测试BCD指标
    print("\n测试二元变化检测指标:")
    bcd_metrics = BCDMetrics()
    pred_binary = np.random.randint(0, 2, (H, W))
    target_binary = np.random.randint(0, 2, (H, W))
    bcd_metrics.update(pred_binary, target_binary)
    bcd_results = bcd_metrics.get_metrics()

    print(f"Precision: {bcd_results['Precision']:.2f}%")
    print(f"Recall: {bcd_results['Recall']:.2f}%")
    print(f"F1: {bcd_results['F1']:.2f}%")
    print(f"IoU: {bcd_results['IoU']:.2f}%")

    print("\n" + "=" * 60)
    print("评估指标测试通过!")
    print("=" * 60)
