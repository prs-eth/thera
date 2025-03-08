# from https://github.com/isaaccorley/jax-enhance

from functools import partial
from typing import Any, Sequence, Callable

import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import freeze
import einops


class PixelShuffle(nn.Module):
    scale_factor: int

    def setup(self):
        self.layer = partial(
            einops.rearrange,
            pattern="b h w (c h2 w2) -> b (h h2) (w w2) c",
            h2=self.scale_factor,
            w2=self.scale_factor
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layer(x)


class ResidualBlock(nn.Module):
    channels: int
    kernel_size: Sequence[int]
    res_scale: float
    activation: Callable
    dtype: Any = jnp.float32

    def setup(self):
        self.body = nn.Sequential([
            nn.Conv(features=self.channels, kernel_size=self.kernel_size, dtype=self.dtype),
            self.activation,
            nn.Conv(features=self.channels, kernel_size=self.kernel_size, dtype=self.dtype),
        ])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.body(x)


class UpsampleBlock(nn.Module):
    num_upsamples: int
    channels: int
    kernel_size: Sequence[int]
    dtype: Any = jnp.float32

    def setup(self):
        layers = []
        for _ in range(self.num_upsamples):
            layers.extend([
                nn.Conv(features=self.channels * 2 ** 2, kernel_size=self.kernel_size, dtype=self.dtype),
                PixelShuffle(scale_factor=2),
            ])
        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class EDSR(nn.Module):
    """Enhanced Deep Residual Networks for Single Image Super-Resolution https://arxiv.org/pdf/1707.02921v1.pdf"""
    scale_factor: int
    channels: int = 3
    num_blocks: int = 32
    num_feats: int = 256
    dtype: Any = jnp.float32

    def setup(self):
        # pre res blocks layer
        self.head = nn.Sequential([nn.Conv(features=self.num_feats, kernel_size=(3, 3), dtype=self.dtype)])

        # res blocks
        res_blocks = [
            ResidualBlock(channels=self.num_feats, kernel_size=(3, 3), res_scale=0.1, activation=nn.relu, dtype=self.dtype)
            for i in range(self.num_blocks)
        ]
        res_blocks.append(nn.Conv(features=self.num_feats, kernel_size=(3, 3), dtype=self.dtype))
        self.body = nn.Sequential(res_blocks)

    def __call__(self, x: jnp.ndarray, _=None) -> jnp.ndarray:
        x = self.head(x)
        x = x + self.body(x)
        return x


def convert_edsr_checkpoint(torch_dict, no_upsampling=True):
    def convert(in_dict):
        top_keys = set([k.split('.')[0] for k in in_dict.keys()])
        leaves = set([k for k in in_dict.keys() if '.' not in k])

        # convert leaves
        out_dict = {}
        for l in leaves:
            if l == 'weight':
                out_dict['kernel'] = jnp.asarray(in_dict[l]).transpose((2, 3, 1, 0))
            elif l == 'bias':
                out_dict[l] = jnp.asarray(in_dict[l])
            else:
                out_dict[l] = in_dict[l]

        for top_key in top_keys.difference(leaves):
            new_top_key = 'layers_' + top_key if top_key.isdigit() else top_key
            out_dict[new_top_key] = convert(
                {k[len(top_key) + 1:]: v for k, v in in_dict.items() if k.startswith(top_key)})
        return out_dict

    converted = convert(torch_dict)

    # remove unwanted keys
    if no_upsampling:
        del converted['tail']

    for k in ('add_mean', 'sub_mean'):
        del converted[k]

    return freeze(converted)
