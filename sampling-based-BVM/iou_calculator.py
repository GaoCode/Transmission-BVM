class IoUCalculator:

    @staticmethod
    def main():
        # Create a ground truth mask and a predicted mask
        gtMask = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
        predMask = [[1, 1, 0], [0, 1, 1], [0, 0, 0]]

        # Calculate IoU
        iou = IoUCalculator.calculateIoU(gtMask, predMask)

        # Print the result
        print('IoU: ', iou)

    @staticmethod
    def calculateIoU(gtMask, predMask):
        # Calculate the true positives,
        # false positives, and false negatives
        tp = 0
        fp = 0
        fn = 0

        for i in range(len(gtMask)):
            for j in range(len(gtMask[0])):
                if gtMask[i][j] == 1 and predMask[i][j] == 1:
                    tp += 1
                elif gtMask[i][j] == 0 and predMask[i][j] == 1:
                    fp += 1
                elif gtMask[i][j] == 1 and predMask[i][j] == 0:
                    fn += 1

        # Calculate IoU
        iou = tp / (tp + fp + fn)

        return iou


if __name__ == '__main__':
    IoUCalculator.main()
