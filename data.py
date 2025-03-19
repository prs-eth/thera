import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
import torch.nn.functional as f
from torchvision.transforms import RandomCrop, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image
from jax.tree_util import tree_map

from utils import make_grid, pil_resize

TARGET_SIZE = 48


class ImageFolder(Dataset):

    def open(self, path):
        with open(str(path), "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __init__(self, root: Path, transform, in_memory=True):
        self.transform = transform
        self.in_memory = in_memory
        self.items = []

        for path in sorted(list(root.rglob('*'))):
            if path.suffix in IMG_EXTENSIONS:
                self.items.append(self.open(path) if in_memory else path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i] if self.in_memory else self.open(self.items[i])
