import os

# import cv2
import numpy as np
import torch
from data import test_dataset
from evaluate import Evaluator
from sampling_bvm import SamplingBVM

MOUNT_POINT = '/home/users/u5155914/bushfire'
MODEL_PATH = 'u5155914/models/sampling-transmission-bvm/' + \
    'baseline/models_final/Model_50_gen.pth'


def test_single_image():
    test_loader = test_dataset('../test_data/mini_set/img/',
                               '../test_data/mini_set/gt/', 480)
    image, gt, HH, WW, name = test_loader.load_data()
    print(type(image), image.shape, type(gt), gt.shape, HH, WW, name)
    # print(torch.unique(gt))

    sampling_bvm = SamplingBVM(
        model_path=os.path.join(MOUNT_POINT, MODEL_PATH))
    pred = sampling_bvm.process_one_image(image, WW, HH)
    # print(np.unique(pred))

    print(torch.min(gt), torch.max(gt), gt.shape)
    # print(torch.min(pred), torch.max(pred), pred.shape)

    # pred_save = '../test_data/mini_set/pred.jpg'
    # cv2.imwrite(pred_save, pred)

    gt_img_path = '../test_data/mini_set/gt/1530903781_+02100.png'
    gt_img = test_loader.binary_loader(gt_img_path)
    gt_img = np.array(gt_img)
    evaluator = Evaluator()
    iou = evaluator.segmentation_iou(pred, gt_img)

    print(iou)
    assert iou > 0.5
