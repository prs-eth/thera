import math

import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, ArrayLike, PyTreeDef
import numpy as np

from utils import interpolate_grid


class ConvNeXtBlock(nn.Module):
    """ConvNext block. See Fig.4 in "A ConvNet for the 2020s" by Liu et al.

    https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf
    """
    n_dims: int = 64
    kernel_size: int = 3  # 7 in the paper's version
    group_features: bool = False

    def setup(self) -> None:
        self.residual = nn.Sequential([
            nn.Conv(self.n_dims, kernel_size=(self.kernel_size, self.kernel_size), use_bias=False,
                    feature_group_count=self.n_dims if self.group_features else 1),
            nn.LayerNorm(),
            nn.Conv(4 * self.n_dims, kernel_size=(1, 1)),
            nn.gelu,
            nn.Conv(self.n_dims, kernel_size=(1, 1)),
        ])

    def __call__(self, x: ArrayLike) -> Array:
        return x + self.residual(x)


class Projection(nn.Module):
    n_dims: int

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        x = nn.LayerNorm()(x)
        x = nn.Conv(self.n_dims, (1, 1))(x)
        return x


class Hypernetwork(nn.Module):
    encoder: nn.Module
    tail_blocks: list
    output_params_shape: list[tuple]  # e.g. [(16,), (32, 32), ...]
    tree_def: PyTreeDef  # used to reconstruct the parameter sets

    def setup(self):
        # setup the tail
        refine_layers = []
        if len(self.tail_blocks):
            current_size = self.tail_blocks[0][0]
            for block_def in self.tail_blocks:
                if block_def[0] != current_size:
                    refine_layers.append(Projection(block_def[0]))
                refine_layers.append(ConvNeXtBlock(*block_def))
                current_size = block_def[0]

        self.refine = nn.Sequential(refine_layers) if len(refine_layers) else lambda x: x

        # one layer 1x1 conv to calculate field params, as in SIREN paper
        output_size = sum(math.prod(s) for s in self.output_params_shape)
        self.out_conv = nn.Conv(output_size, kernel_size=(1, 1), use_bias=True)

    def get_encoding(self, source: ArrayLike, training=False) -> Array:
        """Convenience method for whole-image evaluation"""
        return self.refine(self.encoder(source, training))

    def get_params_at_coords(self, encoding: ArrayLike, coords: ArrayLike) -> Array:
        encoding = interpolate_grid(coords, encoding)
        phi_params = self.out_conv(encoding)

        # reshape to output params shape
        phi_params = jnp.split(
            phi_params, np.cumsum([math.prod(s) for s in self.output_params_shape[:-1]]), axis=-1)
        phi_params = [jnp.reshape(p, p.shape[:-1] + s) for p, s in
                      zip(phi_params, self.output_params_shape)]

        return jax.tree_util.tree_unflatten(self.tree_def, phi_params)

    def __call__(self, source: ArrayLike, target_coords: ArrayLike, training=False) -> Array:
        encoding = self.get_encoding(source, training)
        return self.get_params_at_coords(encoding, target_coords)
