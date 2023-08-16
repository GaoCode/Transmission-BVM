
from data import test_dataset
from sampling_bvm import SamplingBVM

def test_single_image():
    test_loader = test_dataset("../test_data/mini_set/img/","../test_data/mini_set/gt/", 480)
    image, gt, HH, WW, name = test_loader.load_data()
    print(image, gt, HH, WW, name)