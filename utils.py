import random
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import torch
from torchvision.transforms import functional as f
from torchvision import transforms
import numpy as np
from PIL import Image


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


def pil_resize(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


def compute_metrics(out, target, jacobian=None, compute_ssim=False, y_only=False):
    diff = out - target
    if y_only:
        gray_coeffs = np.array([65.738, 129.057, 25.064])[None, None, None] / 256.
        diff = (diff * gray_coeffs).sum(axis=-1)
    mse = jnp.mean(diff ** 2)
    mae = jnp.mean(jnp.abs(diff))
    psnr = -10 * jnp.log10(mse)
    metrics = {'PSNR': psnr, 'MAE': mae, 'MSE': mse}

    if jacobian is not None:
        metrics['TV'] = jnp.mean(jnp.abs(jacobian))  # L1 TV

    if compute_ssim:
        # analogous to torchmetrics.functional.image.ssim
        data_range = max(out.max() - out.min(), target.max() - target.min())
        metrics['SSIM'] = ssim(out, target, data_range).mean()

    return metrics


def ssim(
        img0,
        img1,
        max_val,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        return_map=False
):
    """
    Taken from https://github.com/google/mipnerf/blob/main/internal/math.py
    """
    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = jnp.exp(-0.5 * f_i)
    filt /= jnp.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return jsp.signal.convolve2d(z, f, mode='valid', precision=jax.lax.Precision.HIGHEST)

    filt_fn1 = lambda z: convolve2d(z, filt[:, None])
    filt_fn2 = lambda z: convolve2d(z, filt[None, :])

    # Vmap the blurs to the tensor size, and then compose them.
    num_dims = len(img0.shape)
    map_axes = tuple(list(range(num_dims - 3)) + [num_dims - 1])
    for d in map_axes:
        filt_fn1 = jax.vmap(filt_fn1, in_axes=d, out_axes=d)
        filt_fn2 = jax.vmap(filt_fn2, in_axes=d, out_axes=d)
        filt_fn = lambda z: filt_fn1(filt_fn2(z))

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = jnp.maximum(0., sigma00)
    sigma11 = jnp.maximum(0., sigma11)
    sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01))

    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = jnp.mean(ssim_map, list(range(num_dims - 3, num_dims)))
    return ssim_map if return_map else ssim
