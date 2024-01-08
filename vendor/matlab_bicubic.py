# JAX wrapper around https://github.com/fatheral/matlab_imresize/blob/master/imresize.py

from functools import partial
import numpy as np
from math import ceil
import jax
import jax.numpy as jnp


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def cubic(x):
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + \
        np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (1 < absx) & (absx <= 2))
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1)  # .astype(np.float64)
    u = (x / scale + 0.5 * (1 - 1 / scale))
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store].astype(np.float32)
    indices = indices[:, ind2store]
    return weights, indices


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = jnp.sum(weights * ((inimg[indices].squeeze(axis=1))), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = jnp.sum(weights * ((inimg[:, indices].squeeze(axis=2))), axis=2)
    if inimg.dtype == jnp.uint8:
        outimg = jnp.clip(outimg, 0, 255)
        return jnp.around(outimg).astype(jnp.uint8)
    else:
        return outimg


@partial(jax.jit, static_argnums=(1, 2))
def imresize(I, scalar_scale=None, output_shape=None):
    assert I.dtype == np.float32

    I = (I * 255).astype(np.uint8)

    if scalar_scale is not None and output_shape is not None:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('either scalar_scale OR output_shape should be defined')

    kernel = cubic
    kernel_width = 4.0

    scale_np = np.array(scale, dtype=np.float32)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)

    B = I

    for k in range(2):
        dim = order[k]
        B = imresizevec(B, weights[dim], indices[dim], dim)

    B = (B / 255.).astype(np.float32)
    return B
