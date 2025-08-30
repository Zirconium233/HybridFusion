import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms as transforms
from collections import Counter

import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF', 'tiff'])

def is_text_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.txt', '.TXT'])

def transform():
    return Compose([
        ToTensor(), # to [0,1]
        Normalize([0.5], [0.5])
    ])


def load_img(filepath, is_gray=False, is_ycbcr=False):
    img = Image.open(filepath).convert('RGB')
    if is_gray:
        img = img.convert('L')
    if is_ycbcr:
        img = img.convert('YCbCr').split()[0]
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(A_image, B_image, patch_size, scale=1, ix=-1, iy=-1):
    # PIL size: (width, height)
    (bw, bh) = B_image.size
    tp = scale * patch_size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, bw - ip + 1)  # left (x)
    if iy == -1:
        iy = random.randrange(0, bh - ip + 1)  # top (y)

    # scaled coords for A (assuming A is scale times B)
    tx = scale * ix
    ty = scale * iy

    # crop takes (left, upper, right, lower)
    B_image = B_image.crop((ix, iy, ix + ip, iy + ip))
    A_image = A_image.crop((tx, ty, tx + tp, ty + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return A_image, B_image, info_patch


def augment(A_image, B_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        A_image = ImageOps.flip(A_image)
        B_image = ImageOps.flip(B_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            A_image = ImageOps.mirror(A_image)
            B_image = ImageOps.mirror(B_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            A_image = A_image.rotate(180)
            B_image = B_image.rotate(180)

            info_aug['trans'] = True

    return A_image, B_image, info_aug

def load_text(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    return text

class ImageFusionDataset(data.Dataset):
    """
    Minimal image fusion dataset (visible A + infrared B).
    - Defaults: A kept RGB, B converted to grayscale.
    - Args:
        dir_A, dir_B: folders with paired images (sorted order used).
        dir_C: optional folder with labels (treated like A when provided).
        is_train: if True, augmentation allowed when `augment=True`.
        is_getpatch: if True, randomly sample patches of size `patch_size`.
        patch_size: patch size for B (and scaled A if scale != 1).
        augment: enable simple augmentation (uses existing augment()).
        is_ycbcrA / is_ycbcrB / is_ycbcrC: convert A/B/C to Y channel when True.
        transform: default transform() (ToTensor).
        scale: scale factor between A and B (default 1).
    """
    def __init__(self,
                 dir_A,
                 dir_B,
                 dir_C=None,
                 is_train=True,
                 is_getpatch=False,
                 patch_size=128,
                 augment=False,
                 is_ycbcrA=False,
                 is_ycbcrB=False,
                 is_ycbcrC=False,
                 transform=None,
                  scale=1):
        super(ImageFusionDataset, self).__init__()
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.dir_C = dir_C
        self.is_train = is_train
        self.is_getpatch = is_getpatch
        self.patch_size = patch_size
        self.augment = augment if is_train else False
        self.is_ycbcrA = is_ycbcrA
        self.is_ycbcrB = is_ycbcrB
        self.is_ycbcrC = is_ycbcrC
        self.transform = transform
        self.scale = scale

        self.A_image_filenames = sorted([join(dir_A, x) for x in listdir(dir_A) if is_image_file(x)])
        self.B_image_filenames = sorted([join(dir_B, x) for x in listdir(dir_B) if is_image_file(x)])

        if self.dir_C is not None:
            self.C_image_filenames = sorted([join(dir_C, x) for x in listdir(dir_C) if is_image_file(x)])
        else:
            self.C_image_filenames = None

        # trim to smallest common length
        if self.C_image_filenames is not None:
            n = min(len(self.A_image_filenames), len(self.B_image_filenames), len(self.C_image_filenames))
            self.A_image_filenames = self.A_image_filenames[:n]
            self.B_image_filenames = self.B_image_filenames[:n]
            self.C_image_filenames = self.C_image_filenames[:n]
        else:
            n = min(len(self.A_image_filenames), len(self.B_image_filenames))
            self.A_image_filenames = self.A_image_filenames[:n]
            self.B_image_filenames = self.B_image_filenames[:n]

        # 如果没有提供 transform，自动统计 A/B 中最常见分辨率并在原 transform 前加 Resize
        if self.transform is None:
            def _most_common_size(paths):
                sizes = []
                for p in paths:
                    try:
                        with Image.open(p) as im:
                            sizes.append(im.size)  # PIL: (width, height)
                    except:
                        continue
                if not sizes:
                    return None
                return Counter(sizes).most_common(1)[0][0]

            combined = self.A_image_filenames + self.B_image_filenames
            most = _most_common_size(combined)
            if most is not None:
                w, h = most
                resize_size = (h, w)  # torchvision.Resize expects (H, W)
                self.transform = Compose([transforms.Resize(resize_size), ToTensor(), Normalize([0.5], [0.5])])
            else:
                self.transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.A_image_filenames)

    def __getitem__(self, index):
        a_path = self.A_image_filenames[index]
        b_path = self.B_image_filenames[index]
        c_path = None
        if self.C_image_filenames is not None:
            c_path = self.C_image_filenames[index]

        # default behaviour: A kept RGB (is_gray=False), B -> gray, C like A
        A_image = load_img(a_path, is_gray=False, is_ycbcr=self.is_ycbcrA)
        B_image = load_img(b_path, is_gray=True, is_ycbcr=self.is_ycbcrB)
        C_image = None
        if c_path is not None:
            C_image = load_img(c_path, is_gray=False, is_ycbcr=self.is_ycbcrC)

        if self.is_getpatch:
            A_image, B_image, info_patch = get_patch(A_image, B_image, self.patch_size, scale=self.scale)
            if C_image is not None:
                # apply same crop to C (use same coords as A)
                tx = info_patch['tx']
                ty = info_patch['ty']
                tp = info_patch['tp']
                C_image = C_image.crop((tx, ty, tx + tp, ty + tp))

        if self.augment:
            A_image, B_image, info_aug = augment(A_image, B_image)
            if C_image is not None:
                # apply same augmentation to C using info flags
                if info_aug.get('flip_h', False):
                    C_image = ImageOps.flip(C_image)
                if info_aug.get('flip_v', False):
                    C_image = ImageOps.mirror(C_image)
                if info_aug.get('trans', False):
                    C_image = C_image.rotate(180)

        if self.transform:
            real_A = self.transform(A_image)
            real_B = self.transform(B_image)
            if C_image is not None:
                real_C = self.transform(C_image)
                return real_A, real_B, real_C
            else:
                return real_A, real_B
        else:
            if C_image is not None:
                return A_image, B_image, C_image
            else:
                return A_image, B_image

if __name__ == "__main__":
    # minimal test loop using paths referenced in ref.yml
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='use train paths (default: test if not set)')
    args = parser.parse_args()

    default_train_A = './data/MSRS-main/MSRS-main/train/vi'
    default_train_B = './data/MSRS-main/MSRS-main/train/ir'
    default_train_C = './data/MSRS-main/MSRS-main/train/label'
    # M3FD:
    # "dir_A": "./data/M3FD_Fusion/Vis",
    # "dir_B": "./data/M3FD_Fusion/Ir",
    default_test_A = './data/M3FD_Fusion/Vis'
    default_test_B = './data/M3FD_Fusion/Ir'
    # RS:


    if args.train:
        dirA, dirB, dirC = default_train_A, default_train_B, default_train_C
    else:
        dirA, dirB = default_test_A, default_test_B
        dirC = None

    ds = ImageFusionDataset(
        dir_A=dirA,
        dir_B=dirB,
        dir_C=dirC,
        is_train=args.train,
        is_getpatch=False,
        patch_size=128,
        augment=True,
        is_ycbcrA=False,
        is_ycbcrB=False,
        scale=1
    )

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, drop_last=False)
    print(f"Dataset size: {len(ds)}, running one epoch test...")
    total = 0
    for batch_idx, (A, B) in enumerate(loader):
        total += A.size(0)
        print(f"Batch {batch_idx}: A {tuple(A.shape)}, B {tuple(B.shape)}")
    print(f"Finished epoch, total samples: {total}")