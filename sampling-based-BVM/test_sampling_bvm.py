import os

import cv2
import torch
from data import test_dataset
from PIL import Image
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

    gt_np = 255 * gt.data.cpu().numpy().squeeze()
    gt_save = '../test_data/mini_set/gt.jpg'
    cv2.imwrite(gt_save, gt_np)

    sampling_bvm = SamplingBVM(
        model_path=os.path.join(MOUNT_POINT, MODEL_PATH))
    pred = sampling_bvm.process_one_image(image, WW, HH)
    # print(np.unique(pred))

    print(torch.min(gt), torch.max(gt), gt.shape)
    # print(torch.min(pred), torch.max(pred), pred.shape)

    pred_save = '../test_data/mini_set/pred.jpg'
    cv2.imwrite(pred_save, pred)

    with open('../test_data/mini_set/gt/1530903781_+02100.png', 'rb') as f:
        img = Image.open(f)
        img = img.convert('L')
        # print(np.unique(img))
