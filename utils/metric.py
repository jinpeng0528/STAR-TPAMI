import numpy as np
import torch


class Evaluator:
    def __init__(self, num_class, old_classes_idx=None, new_classes_idx=None):
        self.num_class = num_class
        self.old_classes_idx = old_classes_idx
        self.new_classes_idx = new_classes_idx
        self.total_classes_idx = self.old_classes_idx + self.new_classes_idx
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum() * 100
        if self.old_classes_idx and self.new_classes_idx:
            Acc_old = (
                np.diag(self.confusion_matrix)[self.old_classes_idx].sum()
                / self.confusion_matrix[self.old_classes_idx, :].sum()
            ) * 100
            Acc_new = (
                np.diag(self.confusion_matrix)[self.new_classes_idx].sum()
                / self.confusion_matrix[self.new_classes_idx, :].sum()
            ) * 100
            return {'harmonic': 2 * Acc_old * Acc_new / (Acc_old + Acc_new),
                    'old': Acc_old, 'new': Acc_new, 'overall': Acc}
        else:
            return {'overall': Acc}

    def Pixel_Accuracy_Class(self):
        Acc_by_class = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1) * 100
        Acc = np.nanmean(np.nan_to_num(Acc_by_class)[self.total_classes_idx])
        if self.old_classes_idx and self.new_classes_idx:
            Acc_old = np.nanmean(np.nan_to_num(Acc_by_class[self.old_classes_idx]))
            Acc_new = np.nanmean(np.nan_to_num(Acc_by_class[self.new_classes_idx]))
            return {'harmonic': 2 * Acc_old * Acc_new / (Acc_old + Acc_new),
                    'by_class': Acc_by_class, 'old': Acc_old, 'new': Acc_new, 'overall': Acc}
        else:
            return {'overall': Acc, 'by_class': Acc_by_class}

    def Mean_Intersection_over_Union(self):
        MIoU_by_class = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix + 1e-6)
        ) * 100
        MIoU = np.nanmean(np.nan_to_num(MIoU_by_class)[self.total_classes_idx])
        if self.old_classes_idx and self.new_classes_idx:
            MIoU_old = np.nanmean(np.nan_to_num(MIoU_by_class[self.old_classes_idx]))
            MIoU_new = np.nanmean(np.nan_to_num(MIoU_by_class[self.new_classes_idx]))
            return {'harmonic': 2 * MIoU_old * MIoU_new / (MIoU_old + MIoU_new), 'by_class': MIoU_by_class, 'old': MIoU_old, 'new': MIoU_new, 'overall': MIoU}
        else:
            return {'overall': MIoU, 'by_class': MIoU_by_class}

    def Precision(self):
        num_classes = self.confusion_matrix.shape[0]
        precision = []
        for i in range(num_classes):
            TP = self.confusion_matrix[i, i]
            FP = np.sum(self.confusion_matrix[:, i]) - TP
            precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0
            precision.append(precision_i * 100)
        precision = np.array(precision)
        if self.old_classes_idx and self.new_classes_idx:
            precision_old = np.nanmean(np.nan_to_num(precision[self.old_classes_idx]))
            precision_new = np.nanmean(np.nan_to_num(precision[self.new_classes_idx]))
            return {'harmonic': 2 * precision_old * precision_new / (precision_old + precision_new),
                    'by_class': precision, 'old': precision_old, 'new': precision_new, 'overall': np.nanmean(precision)}

    # def False_Positive_Rate(self):
    #     FPR = np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
    #     FPR = FPR / (np.sum(self.confusion_matrix, axis=0) + 1e-6) * 100
    #     if self.old_classes_idx and self.new_classes_idx:
    #         FPR_old = np.nanmean(np.nan_to_num(FPR[self.old_classes_idx]))
    #         FPR_new = np.nanmean(np.nan_to_num(FPR[self.new_classes_idx]))
    #         return {'harmonic': 2 * FPR_old * FPR_new / (FPR_old + FPR_new), 'by_class': FPR, 'old': FPR_old, 'new': FPR_new, 'overall': np.nanmean(FPR)}
    #     else:
    #         return {'overall': np.nanmean(FPR), 'by_class': FPR}

    def False_Positive(self):
        num_classes = self.confusion_matrix.shape[0]
        all_FP = []

        total = np.sum(self.confusion_matrix)
        for i in range(num_classes):
            TP = self.confusion_matrix[i, i]
            FP = np.sum(self.confusion_matrix[:, i]) - TP
            all_FP.append(FP)
        all_FP = np.array(all_FP)
        if self.old_classes_idx and self.new_classes_idx:
            FP_old = np.nanmean(np.nan_to_num(all_FP[self.old_classes_idx]))
            FP_new = np.nanmean(np.nan_to_num(all_FP[self.new_classes_idx]))
            return {'harmonic': 2 * FP_old * FP_new / (FP_old + FP_new), 'by_class': all_FP, 'old': FP_old, 'new': FP_new, 'overall': np.nanmean(all_FP)}
        else:
            return {'overall': np.nanmean(all_FP), 'by_class': all_FP}

    def False_Positive_Rate(self):
        num_classes = self.confusion_matrix.shape[0]
        FPR = []

        total = np.sum(self.confusion_matrix)
        for i in range(num_classes):
            TP = self.confusion_matrix[i, i]
            FP = np.sum(self.confusion_matrix[:, i]) - TP
            TN = total - np.sum(self.confusion_matrix[i, :]) - np.sum(self.confusion_matrix[:, i]) + TP
            FPR_i = FP / (FP + TN) if (FP + TN) > 0 else 0
            FPR.append(FPR_i * 10000)
        FPR = np.array(FPR)
        if self.old_classes_idx and self.new_classes_idx:
            FPR_old = np.nanmean(np.nan_to_num(FPR[self.old_classes_idx]))
            FPR_new = np.nanmean(np.nan_to_num(FPR[self.new_classes_idx]))
            return {'harmonic': 2 * FPR_old * FPR_new / (FPR_old + FPR_new), 'by_class': FPR, 'old': FPR_old, 'new': FPR_new, 'overall': np.nanmean(FPR)}
        else:
            return {'overall': np.nanmean(FPR), 'by_class': FPR}

    def False_Negative(self):
        num_classes = self.confusion_matrix.shape[0]
        all_FN = []

        total = np.sum(self.confusion_matrix)
        for i in range(num_classes):
            TP = self.confusion_matrix[i, i]
            FN = np.sum(self.confusion_matrix[i, :]) - TP
            all_FN.append(FN)
        all_FN = np.array(all_FN)
        if self.old_classes_idx and self.new_classes_idx:
            FN_old = np.nanmean(np.nan_to_num(all_FN[self.old_classes_idx]))
            FN_new = np.nanmean(np.nan_to_num(all_FN[self.new_classes_idx]))
            return {'harmonic': 2 * FN_old * FN_new / (FN_old + FN_new), 'by_class': all_FN, 'old': FN_old, 'new': FN_new, 'overall': np.nanmean(all_FN)}
        else:
            return {'overall': np.nanmean(all_FN), 'by_class': all_FN}

    def False_Negative_Rate(self):
        num_classes = self.confusion_matrix.shape[0]
        FNR = []

        for i in range(num_classes):
            TP = self.confusion_matrix[i, i]
            FN = np.sum(self.confusion_matrix[i, :]) - TP
            FNR_i = FN / (FN + TP) if (FN + TP) > 0 else 0
            FNR.append(FNR_i * 10000)
        FNR = np.array(FNR)
        if self.old_classes_idx and self.new_classes_idx:
            FNR_old = np.nanmean(np.nan_to_num(FNR[self.old_classes_idx]))
            FNR_new = np.nanmean(np.nan_to_num(FNR[self.new_classes_idx]))
            return {'harmonic': 2 * FNR_old * FNR_new / (FNR_old + FNR_new), 'by_class': FNR, 'old': FNR_old, 'new': FNR_new, 'overall': np.nanmean(FNR)}
        else:
            return {'overall': np.nanmean(FNR), 'by_class': FNR}

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def sync(self, device):
        # Collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
