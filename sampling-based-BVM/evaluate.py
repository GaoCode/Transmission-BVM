import numpy as np


class Evaluator():

    def segmentation_iou(self, pred, gt, label=255):
        print(type(pred), type(gt))
        print(np.shape(pred), np.shape(gt))
        print(np.unique(pred), np.unique(gt))

        TP = np.sum((gt == label) & (pred == label))
        FP = np.sum((gt != label) & (pred == label))
        FN = np.sum((gt == label) & (pred != label))

        print(np.sum(gt == label), np.sum(gt == 0), np.sum(gt != label),
              np.sum(pred == label))
        print(np.max(gt))

        print(TP, FP, FN)

        n = TP
        d = float(TP + FP + FN + 1e-12)

        iou = np.divide(n, d)

        return iou
