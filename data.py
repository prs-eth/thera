import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
import torch.nn.functional as f
import torchvision.transforms.functional as tf
from torchvision.transforms import RandomCrop, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image
from jax.tree_util import tree_map

from utils import make_grid, pil_resize, RandomRotate

INTERP_MODE = 'bicubic'
TARGET_SIZE = 48


def to_uint8_tensor(image: Image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)


def open_img(path):
    with open(str(path), "rb") as f:
        img = Image.open(f).convert("RGB")
        img = to_uint8_tensor(img)
    return img


class ImageFolder(Dataset):

    def __init__(self, root: Path, in_memory=True):
        self.in_memory = in_memory
        self.items = []

        for path in sorted(list(root.rglob('*'))):
            if path.suffix in IMG_EXTENSIONS:
                self.items.append(open_img(path) if in_memory else path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i] if self.in_memory else open_img(self.items[i])


class ArbitraryScaleWrapper(Dataset):

    def __init__(
            self,
            dataset,
            source_size: int,
            scale_range: tuple[float, float],
            augment_scale_range: tuple[float, float],
            augment_scale_prob: float,
            transforms):
        self.dataset = dataset
        self.source_size = source_size
        self.scale_range = scale_range
        self.augment_scale_range = augment_scale_range
        self.augment_scale_prob = augment_scale_prob
        self.transforms = transforms

        assert augment_scale_range[0] >= 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = self.dataset[item]

        scale = random.uniform(*self.scale_range)
        min_shape = min(image.shape[-2:])
        assert self.source_size * scale <= min_shape
        augment_scale_upper = min(self.augment_scale_range[1],
                                  min_shape / (self.source_size * scale))
        augment_scale_lower = min(self.augment_scale_range[0], augment_scale_upper)

        augment_scale = random.uniform(augment_scale_lower, augment_scale_upper) \
            if random.random() < self.augment_scale_prob else 1

        # source will always be self.source_size**2, target size is determined by scale factor
        # TODO the `min` should not be necessary
        target = RandomCrop(min(int(self.source_size * scale * augment_scale), min_shape))(image)
        target = target.float() / 255.
        target_size = max(int(target.shape[-1] / augment_scale), self.source_size)
        if augment_scale != 1:
            target = pil_resize(target, (target_size, target_size))

        target = self.transforms(target)
        source = pil_resize(target, (self.source_size, self.source_size))

        source_up = f.interpolate(
            source.unsqueeze(0), (target_size, target_size), mode='nearest-exact')[0]
        effective_scale = target_size / self.source_size

        source, source_up, target = \
            source.permute(1, 2, 0), source_up.permute(1, 2, 0), target.permute(1, 2, 0)

        # get random pixel samples
        target_coords = torch.from_numpy(make_grid(target_size)).view(-1, 2)
        target_flat = target.view(-1, 3)
        source_up_flat = source_up.view(-1, 3)
        # np.random.choice doesnt work properly with multiprocessing
        sample_idc = torch.randperm(target_size**2)[:TARGET_SIZE**2]
        target_coords, target_flat, source_up_flat = \
            target_coords[sample_idc], target_flat[sample_idc], source_up_flat[sample_idc]

        return {
            'source': source,
            'target_coords': target_coords.view(TARGET_SIZE, TARGET_SIZE, -1),
            'target': target_flat.view(TARGET_SIZE, TARGET_SIZE, -1),
            'source_nearest': source_up_flat.view(TARGET_SIZE, TARGET_SIZE, -1),
            'scale': effective_scale
        }


class COZImageFolder(Dataset):

    def __init__(self, root: Path):
        self.hr, self.lr = [], []

        for hr_path in sorted(list((root / 'HR').rglob('*'))):
            if hr_path.suffix.lower() in IMG_EXTENSIONS:
                self.hr.append(open_img(hr_path))
            lr = []
            for lr_path in sorted(list((root / 'LR' / hr_path.stem).rglob('*'))):
                if lr_path.suffix.lower() in IMG_EXTENSIONS:
                    lr.append(open_img(lr_path))
            self.lr.append(lr)

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, i):
        return self.hr[i], self.lr[i]


class COZArbitraryScaleWrapper(Dataset):
    def __init__(
            self,
            dataset,
            source_size: int,
            scale_range: tuple[float, float],
            transforms):
        self.dataset = dataset
        self.source_size = source_size
        self.scale_range = scale_range
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        hr, lr = self.dataset[item]
        lr = [i for i in lr if self.scale_range[0] <= hr.shape[-1] / i.shape[-1] <= self.scale_range[1]]
        lr = random.choice(lr)
        scale = hr.shape[-1] / lr.shape[-1]
        s2 = hr.shape[-2] / lr.shape[-2]

        pad = 0
        if s2 < scale:
            new_lr_size = int(hr.shape[-1] / s2)
            pad = new_lr_size - lr.shape[-1] + 1

        x0 = random.randint(0, lr.shape[-2] - self.source_size - pad)
        y0 = random.randint(0, lr.shape[-1] - self.source_size)
        crop_lr = lr[:, x0: x0 + self.source_size, y0: y0 + self.source_size]
        crop_lr = crop_lr.float() / 255.

        target_size = round(self.source_size * scale)
        x1 = round(x0 * scale)
        y1 = round(y0 * scale)
        crop_hr = hr[:, x1: x1 + target_size, y1: y1 + target_size]
        crop_hr = crop_hr.float() / 255.

        if random.random() < 0.5:
            crop_lr = tf.hflip(crop_lr)
            crop_hr = tf.hflip(crop_hr)

        if random.random() < 0.5:
            crop_lr = tf.vflip(crop_lr)
            crop_hr = tf.vflip(crop_hr)

        deg = random.choice([0, 90, 180, 270])
        crop_lr = tf.rotate(crop_lr, deg)
        crop_hr = tf.rotate(crop_hr, deg)

        # save as PIL image
        #transforms.ToPILImage()(crop_lr).resize(crop_hr.shape[-2:], Image.NEAREST).save('crop_lr.png')
        #transforms.ToPILImage()(crop_hr).save('crop_hr.png')

        source_up = f.interpolate(
            crop_lr.unsqueeze(0), (target_size, target_size), mode='nearest-exact')[0]
        #assert scale == target_size / self.source_size

        source, source_up, target = \
            crop_lr.permute(1, 2, 0), source_up.permute(1, 2, 0), crop_hr.permute(1, 2, 0)

        # get random pixel samples
        target_coords = torch.from_numpy(make_grid(target_size)).view(-1, 2)
        target_flat = target.view(-1, 3)
        source_up_flat = source_up.view(-1, 3)
        # np.random.choice doesnt work properly with multiprocessing
        sample_idc = torch.randperm(target_size**2)[:TARGET_SIZE**2]
        try:
        	target_coords, target_flat, source_up_flat = \
            	target_coords[sample_idc], target_flat[sample_idc], source_up_flat[sample_idc]
        except:
            print(pad)
            print(scale, target_size, hr.shape, lr.shape)
            print(x0, y0, x1, y1)
            print(crop_lr.shape, crop_hr.shape)
            exit(1)

        return {
            'source': source,
            'target_coords': target_coords.view(TARGET_SIZE, TARGET_SIZE, -1),
            'target': target_flat.view(TARGET_SIZE, TARGET_SIZE, -1),
            'source_nearest': source_up_flat.view(TARGET_SIZE, TARGET_SIZE, -1),
            'scale': scale
        }


def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))


def make_data_loaders(args: Namespace):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRotate([0, 90, 180, 270])
    ])
    if not args.coz:
        data_sets = [ImageFolder(Path(args.data_dir) / s) for s in (args.train_set, args.val_set)]
        data_sets = [ArbitraryScaleWrapper(ds, args.patch_size, args.scale_range, args.augment_scale_range,
                                           args.augment_scale_prob, transform) for ds in data_sets]
    else:
        data_sets = [COZImageFolder(Path(args.data_dir) / s) for s in (args.train_set, args.val_set)]
        data_sets = [COZArbitraryScaleWrapper(ds, args.patch_size, args.scale_range, transform) for ds in data_sets]

    data_loaders = [DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=numpy_collate
    ) for ds in data_sets]

    return data_loaders
