import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
import torch.nn.functional as f
from torchvision.transforms import RandomCrop
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image
from jax.tree_util import tree_map

from utils import make_grid, pil_resize

TARGET_SIZE = 48


def to_uint8_tensor(image: Image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)


def open_img(path):
    with open(str(path), "rb") as f:
        img = Image.open(f).convert("RGB")
        img = to_uint8_tensor(img)
    return img


class ImageFolder(Dataset):

    def __init__(self, root: Path, *, in_memory=True):
        self.in_memory = in_memory
        self.items = []

        for path in sorted(list(root.rglob('*'))):
            if path.suffix in IMG_EXTENSIONS:
                self.items.append(open_img(path) if in_memory else path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item_uint8 = self.items[i] if self.in_memory else open_img(self.items[i])
        return item_uint8 / 255.


class ArbitraryScaleWrapper(Dataset):

    def __init__(
            self,
            dataset,
            source_size: int,
            scale_range: tuple[float, float],
            augment_scale_range: tuple[float, float],
            augment_scale_prob: float,
            transforms
    ):
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
        augment_scale = random.uniform(*self.augment_scale_range) \
             if random.random() < self.augment_scale_prob else 1

        # source will always be self.source_size**2, target size is determined by scale factor
        target = RandomCrop(round(self.source_size * scale * augment_scale))(image)

        target_size = int(target.shape[-1] / augment_scale)
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


def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))
