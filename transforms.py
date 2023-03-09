import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, images, target):
        image1 = F.to_tensor(images[0])
        image2 = F.to_tensor(images[1])
        images = [image1, image2]
        return images, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images, target):
        if random.random() < self.prob:
            image = images[0]
            image_ul = images[1]
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            image_ul = image_ul.flip(-1)  # 水平翻转图片

            images = [image, image_ul]

            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return images, target
