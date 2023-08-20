import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from numpy import array, exp

# several data augumentation strategies
def cv_random_flip(img_r, label_r):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img_r = img_r.transpose(Image.FLIP_LEFT_RIGHT)
        label_r = label_r.transpose(Image.FLIP_LEFT_RIGHT)

    return img_r, label_r


def randomCrop(img_r, label_r):
    border = 30
    image_width = img_r.size[0]
    image_height = img_r.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = ((image_width - crop_win_width) >> 1,
                     (image_height - crop_win_height) >> 1,
                     (image_width + crop_win_width) >> 1,
                     (image_height + crop_win_height) >> 1)
    return img_r.crop(random_region), label_r.crop(random_region)


def randomRotation(img_r, label_r):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img_r = img_r.rotate(random_angle, mode)
        label_r = label_r.rotate(random_angle, mode)
    return img_r, label_r


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0, sigma=0.15):

    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomGaussian1(image, mean=0.1, sigma=0.35):

    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and
# test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.),
# the performance will be further improved.
class SalObjDataset(data.Dataset):

    def __init__(self, right_image_root, right_gt_root, trainsize):
        self.trainsize = trainsize
        self.right_images = [
            right_image_root + f for f in os.listdir(right_image_root)
            if f.endswith('.jpg')
        ]
        self.right_gts = [
            right_gt_root + f for f in os.listdir(right_gt_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]

        self.right_images = sorted(self.right_images)
        self.right_gts = sorted(self.right_gts)

        self.filter_files()
        self.size = len(self.right_images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        right_image = self.rgb_loader(self.right_images[index])
        right_gt = self.binary_loader(self.right_gts[index])
        right_image, right_gt = cv_random_flip(right_image, right_gt)
        right_image, right_gt = randomCrop(right_image, right_gt)
        right_image, right_gt = randomRotation(right_image, right_gt)
        right_image = colorEnhance(right_image)
        # gt=randomGaussian(gt)
        right_gt = randomPeper(right_gt)
        right_image = self.img_transform(right_image)
        right_gt = self.gt_transform(right_gt)

        return right_image, right_gt

    def filter_files(self):
        assert len(self.right_images) == len(self.right_gts)
        right_images = []
        right_gts = []

        for right_img_path, right_gt_path in zip(self.right_images,
                                                 self.right_gts):
            right_img = Image.open(right_img_path)
            right_gt = Image.open(right_gt_path)

            if right_img.size == right_gt.size:
                right_images.append(right_img_path)
                right_gts.append(right_gt_path)

        self.right_images = right_images
        self.right_gts = right_gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h),
                                                                 Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(right_image_root,
               right_gt_root,
               batchsize,
               trainsize,
               shuffle=True,
               num_workers=12,
               pin_memory=True):
    dataset = SalObjDataset(right_image_root, right_gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader


class test_dataset:

    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.gts = [
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomVerticalFlip(p=0)
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(gt).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

class ImageLoader:
    def __init__(self, input_seg_size=480):
        self.transform = transforms.Compose(
            [
                transforms.Resize((input_seg_size, input_seg_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def process_input_segmentation(self, img_path):
        image = self.rgb_loader(img_path)
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        return image, HH, WW

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")



class DetectorPostProcessor:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def post_process_segmentation(self, pred):
        pred = DetectorPostProcessor.sigmoid(pred)
        print(
            np.max(pred), 
            np.min(pred), 
            pred.size, 
            pred.shape,
            np.isnan(pred).any())

        pred = pred.squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred.astype(int)
        print(
            np.max(pred), 
            np.min(pred), 
            pred.size, 
            pred.shape,
            np.isnan(pred).any())
        return pred
