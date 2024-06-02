from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,Dataset
import cv2
import numpy as np
import torch
import imgaug.augmenters as iaa
import random
#读取训练图片类
class Mydataset(Dataset):
    def __init__(self, lines, train=True):
        super(Mydataset, self).__init__()
        #储存图像所有路径
        self.lines = lines
        self.train = train


    def __getitem__(self, item):
        """读取图像,并转换成rgb格式"""
        # 图片路径
        img_path = self.lines[item].strip().split()[0]
        # 图片标签
        img_lab = self.lines[item].strip().split()[1]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image at {img_path}")
        img = img[..., ::-1]

        #img = cv2.imread(img_path)[..., ::-1]
        # 图像标签转换成整数
        img_lab = int(img_lab)
        # 数据增强
        if self.train:
            img = self.get_random_data(img)
        else:
            img = cv2.resize(img, (64, 64))
        # 灰度化
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 进行二值化
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        img = 255-img
        """数据归一化，并在添加一个维度"""
        img = np.expand_dims(img, axis=0)/255
        img = img.astype('float32')


        return img,img_lab


    def __len__(self):
        #返回训练图片数量
        return len(self.lines)

    def get_random_data(self, img):
        """随机增强图像"""
        seq = iaa.Sequential([
            # 更改亮度，不影响边界框
            iaa.Multiply((0.8, 1.5)),  # change brightness, doesn't affect BBs(bounding boxes)
            # 高斯扰动
            iaa.GaussianBlur(sigma=(0, 1.0)),  # 标准差为0到3之间的值
            # 截取 按比例来crop
            iaa.Crop(percent=(0, 0.06)),
            # 变成灰度图 alpha: 覆盖旧的颜色空间时，新颜色空间的Alpha值
            iaa.Grayscale(alpha=(0, 1)),
            # 仿射变换
            iaa.Affine(
                # scale: 图像缩放因子。1表示不缩放,0.5表示缩小到原来的50%，
                # 两个key:x, y,每个x或y的值都可以是float, float tuple,此时x-axis和y-axis的缩放比例不一样
                scale=(0.9, 1.),  # 尺度变换
                # 旋转
                rotate=(-20, 20),
                # cval: 当平移后使用常量填充的时候指定填充的常量值
                cval=(250),
                # mode: 采用一个常量填充经过变换后空白的像素点
                mode='constant'),
            #
            iaa.Resize(64)
        ])
        # 增强图像
        img = seq.augment(image=img)
        return img



