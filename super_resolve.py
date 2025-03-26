#!/usr/bin/env python

from argparse import ArgumentParser, Namespace
import pickle

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from PIL import Image

from model import build_thera
from utils import make_grid, interpolate_grid

MEAN = jnp.array([.4488, .4371, .4040])
VAR = jnp.array([.25, .25, .25])
PATCH_SIZE = 256


def process_single(source, apply_encoder, apply_decoder, params, target_shape):
    t = jnp.float32((target_shape[0] / source.shape[1])**-2)[None]
    coords_nearest = jnp.asarray(make_grid(target_shape)[None])
    source_up = interpolate_grid(coords_nearest, source[None])
    source = jax.nn.standardize(source, mean=MEAN, variance=VAR)[None]

    encoding = apply_encoder(params, source)
    coords = jnp.asarray(make_grid(source_up.shape[1:3])[None])  # global sampling coords
    out = jnp.full_like(source_up, jnp.nan, dtype=jnp.float32)

    for h_min in range(0, coords.shape[1], PATCH_SIZE):
        h_max = min(h_min + PATCH_SIZE, coords.shape[1])
        for w_min in range(0, coords.shape[2], PATCH_SIZE):
            # apply decoder with one patch of coordinates
            w_max = min(w_min + PATCH_SIZE, coords.shape[2])
            coords_patch = coords[:, h_min:h_max, w_min:w_max]
            out_patch = apply_decoder(params, encoding, coords_patch, t)
            out = out.at[:, h_min:h_max, w_min:w_max].set(out_patch)

    out = out * jnp.sqrt(VAR)[None, None, None] + MEAN[None, None, None]
    out += source_up
    return out


def process(source, model, params, target_shape, do_ensemble=True):
    apply_encoder = jit(model.apply_encoder)
    apply_decoder = jit(model.apply_decoder)

    outs = []
    for i_rot in range(4 if do_ensemble else 1):
        source_ = jnp.rot90(source, k=i_rot, axes=(-3, -2))
        target_shape_ = tuple(reversed(target_shape)) if i_rot % 2 else target_shape
        out = process_single(source_, apply_encoder, apply_decoder, params, target_shape_)
        outs.append(jnp.rot90(out, k=i_rot, axes=(-2, -3)))

    out = jnp.stack(outs).mean(0).clip(0., 1.)
    return jnp.rint(out[0] * 255).astype(jnp.uint8)


def main(args: Namespace):
    source = np.asarray(Image.open(args.in_file).convert('RGB')) / 255.

    if args.scale is not None:
        if args.size is not None:
            raise ValueError('Cannot specify both size and scale')
        target_shape = (
            round(source.shape[0] * args.scale),
            round(source.shape[1] * args.scale),
        )
    elif args.size is not None:
        target_shape = args.size
    else:
        raise ValueError('Must specify either size or scale')

    with open(args.checkpoint, 'rb') as fh:
        check = pickle.load(fh)
        params, backbone, size = check['model'], check['backbone'], check['size']

    model = build_thera(3, backbone, size)

    out = process(source, model, params, target_shape, not args.no_ensemble)

    Image.fromarray(np.asarray(out)).save(args.out_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('in_file')
    parser.add_argument('out_file')
    parser.add_argument('--scale', type=float, help='Scale factor for super-resolution')
    parser.add_argument('--size', type=int, nargs=2,
                        help='Target size (h, w), mutually exclusive with --scale')
    parser.add_argument('--checkpoint', help='Path to checkpoint file')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable geo-ensemble')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
