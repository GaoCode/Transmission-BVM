
import os
import cv2
from data import test_dataset
from sampling_bvm import SamplingBVM

MOUNT_POINT = "/home/users/u5155914/bushfire"
MODEL_PATH = "u5155914/models/sampling-transmission-bvm/baseline/models_final/Model_50_gen.pth"

def test_single_image():
    test_loader = test_dataset("../test_data/mini_set/img/","../test_data/mini_set/gt/", 480)
    image, gt, HH, WW, name = test_loader.load_data()
    print(image.shape, gt.shape, HH, WW, name)

    sampling_bvm = SamplingBVM(model_path=os.path.join(MOUNT_POINT, MODEL_PATH))
    pred = sampling_bvm.process_one_image(image, WW, HH)

    pred_save = "../test_data/mini_set/pred.jpg"
    cv2.imwrite(pred_save, pred)

