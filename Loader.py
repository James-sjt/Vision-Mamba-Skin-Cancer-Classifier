import os
from PIL import Image
import torch
from torchvision import transforms
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loader_train(batch_size, idx_helper, cls_1, cls_2):
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(60),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])
    images = []
    label = torch.zeros(batch_size, 2).to(device)
    current_time = time.time()
    random.seed(int(current_time))
    d_types = [cls_1, cls_2]
    for i in range(batch_size):
        flag = random.randint(0, 1)
        path = 'Skin Cancer/Skin Cancer/' + d_types[flag]
        imgs = os.listdir(path)
        idx = idx_helper[flag]
        idx_helper[flag] += 1
        img_path = os.path.join(path, imgs[idx % len(imgs)])
        image = Image.open(img_path).convert('RGB')
        image = transform_train(image)
        images.append(image)
        label[i, flag] = 1
    images = torch.stack(images, dim=0).to(device)

    return images, label, idx_helper


def loader_test(batch_size, flag, cls_1, cls_2):
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    images = []
    label = torch.zeros(batch_size, 2).to(device)
    nums = {'akiec': 60, 'mel': 220, 'bkl': 220, 'nv': 1340, 'bcc': 100, 'vasc': 30, 'df': 20}
    flags = {'akiec': 60, 'mel': 280, 'bkl': 500, 'nv': 1840, 'bcc': 1940, 'vasc': 1970, 'df': 1990}
    for i in range(batch_size):
        path = 'Skin Cancer/Skin Cancer/test'
        imgs = os.listdir(path)
        img_path = os.path.join(path, imgs[i + flag])
        image = Image.open(img_path).convert('RGB')
        image = transform_test(image)
        images.append(image)

        if flag + i + 1 <= flags[cls_1] and flag + i + 1 <= flags[cls_2] - nums[cls_2]:  # cls_1
            label[i, 0] = 1
        else:  # cls_2
            label[i, 1] = 1

    images = torch.stack(images, dim=0).to(device)

    return images, label
