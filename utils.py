import torch as t
import torchvision as tv
import numpy as np


def gram_matrix(y):
    b, c, h, w = y.size()
    feature = y.view(b, c, h * w)
    gram = t.bmm(feature, feature.transpose(1, 2))
    return gram


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
#  vgg16 是在imagenet 上训练的
# 而imagenet的图像已被归一化为 mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]


def get_image_data(path, image_size):
    image_transforms = tv.transforms.Compose([
        tv.transforms.Resize(image_size),
        tv.transforms.CenterCrop(image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    image = tv.datasets.folder.default_loader(path)
    img_tensor = image_transforms(image).unsqueeze(0)
    return img_tensor
