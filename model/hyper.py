import math

import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, ArrayLike, PyTreeDef
import numpy as np

from utils import interpolate_grid


class Hypernetwork(nn.Module):
    encoder: nn.Module
    refine: nn.Module
    output_params_shape: list[tuple]  # e.g. [(16,), (32, 32), ...]
    tree_def: PyTreeDef  # used to reconstruct the parameter sets

    def setup(self):
        # one layer 1x1 conv to calculate field params, as in SIREN paper
        output_size = sum(math.prod(s) for s in self.output_params_shape)
        self.out_conv = nn.Conv(output_size, kernel_size=(1, 1), use_bias=True)

    def get_encoding(self, source: ArrayLike, training=False) -> Array:
        """Convenience method for whole-image evaluation"""
        return self.refine(self.encoder(source, training), training)

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
