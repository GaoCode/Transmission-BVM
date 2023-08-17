import numpy as np


class Evaluator():

    def segmentation_iou(self, pred, gt):
        print(type(pred), type(gt))
        print(np.shape(pred), np.shape(gt))
        print(np.unique(pred), np.unique(gt))

        TP = np.sum((gt == 255) & (pred == 255))
        FP = np.sum((gt != 255) & (pred == 255))
        FN = np.sum((gt == 255) & (pred != 255))

        print(np.sum(gt == 255), np.sum(gt == 0), np.sum(gt != 255),
              np.sum(pred == 255))
        print(np.max(gt))

        print(TP, FP, FN)

        n = TP
        d = float(TP + FP + FN + 1e-12)

        iou = np.divide(n, d)

        return iou
