import random
import torch
from torchvision import transforms

class ApplyOneof(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        transform = random.choice(self.transforms)
        return transform(img)

class Identity(object):
    def __call__(self, img):
        return img


def random_hflip(img, p = 0.5):
    if torch.rand(1) > p:
        return img
    else:
        if len(img.shape) == 5:
            b, n_slice, c, h, w = img.shape
            img = img.reshape(b * n_slice, c, h, w)
            img_t = transforms.functional.hflip(img).reshape(b, n_slice, c, h, w)
            return img_t

        else:
            img_t = transforms.functional.hflip(img)
            return img_t

def random_rotate(img, max_angle = 45, p = 0.5):
    # img: tensor value in [-1, 1]
    # angle: rotate angle value in degrees, counter-clockwise
    if torch.rand(1) > p:
        return img
    else:
        random_angle = (2 * random.random() - 1) * max_angle # (-angle, angle]
        if len(img.shape) == 5:
            b, n_slice, c, h, w = img.shape
            img = img.reshape(b * n_slice, c, h, w)
            img_t = transforms.functional.rotate(img + 1.0, random_angle) - 1.0
            return img_t.reshape(b, n_slice, c, h, w)
        else:
            img_t = transforms.functional.rotate(img + 1.0, random_angle) - 1.0
            return img_t

def random_aug_g(img):
    transforms = [random_hflip, random_rotate]
    for t in transforms:
        img = t(img)
    return img


class RandomAugG(object):
    def __init__(self, rotate = 45, p = 0.5):
        self.transforms = [random_hflip, random_rotate]
        self.p = p
        
    def __call__(self, img):
        for t in self.transforms:
            img = t(img, p = float(self.p))
        return img