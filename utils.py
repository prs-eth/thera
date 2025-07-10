import random
from functools import partial

import jax
import jax.numpy as jnp
import torch
from torchvision.transforms import functional as f
from torchvision import transforms
import numpy as np
from PIL import Image


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split(arr, n_devices):
    """
    Splits the first axis of `arr` evenly across the number of devices.
    https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html
    """
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


def repeat_vmap(fun, in_axes=[0]):
    for axes in in_axes:
        fun = jax.vmap(fun, in_axes=axes)
    return fun


def make_grid(patch_size: int | tuple[int, int]):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    offset_h, offset_w = 1 / (2 * np.array(patch_size))
    space_h = np.linspace(-0.5 + offset_h, 0.5 - offset_h, patch_size[0])
    space_w = np.linspace(-0.5 + offset_w, 0.5 - offset_w, patch_size[1])
    return np.stack(np.meshgrid(space_h, space_w, indexing='ij'), axis=-1)  # [h, w]


def interpolate_grid(coords, grid, order=0):
    """
    args:
        coords: Tensor of shape (B, H, W, 2) with coordinates in [-0.5, 0.5]
        grid: Tensor of shape (B, H', W', C)
    returns:
        Tensor of shape (B, H, W, C) with interpolated values
    """
    # convert [-0.5, 0.5] -> [0, size], where pixel centers are expected at
    # [-0.5 + 1 / (2*size), ..., 0.5 - 1 / (2*size)]
    coords = coords.transpose((0, 3, 1, 2))
    coords = coords.at[:, 0].set(coords[:, 0] * grid.shape[-3] + (grid.shape[-3] - 1) / 2)
    coords = coords.at[:, 1].set(coords[:, 1] * grid.shape[-2] + (grid.shape[-2] - 1) / 2)
    map_coordinates = partial(jax.scipy.ndimage.map_coordinates, order=order, mode='nearest')
    return jax.vmap(jax.vmap(map_coordinates, in_axes=(2, None), out_axes=2))(grid, coords)


class RandomRotate:
    """https://pytorch.org/vision/main/transforms.html"""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return f.rotate(x, angle)


def pil_resize(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


def compute_metrics(out, target, jacobian=None, y_only=False):
    diff = out - target
    if y_only:
        # the / 256 was transferred from previous implementations
        gray_coeffs = np.array([65.738, 129.057, 25.064])[None, None, None] / 256.
        diff = (diff * gray_coeffs).sum(axis=-1)
    mse = jnp.mean(diff ** 2)
    mae = jnp.mean(jnp.abs(diff))
    psnr = -10 * jnp.log10(mse)
    metrics = {'PSNR': psnr, 'MAE': mae, 'MSE': mse}

    if jacobian is not None:
        metrics['TV'] = jnp.mean(jnp.abs(jacobian))  # L1 TV

    return metrics
