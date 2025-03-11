import math

import jax
from flax.core import unfreeze, freeze
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, ArrayLike, PyTree

from .edsr import EDSR
from .rdn import RDN
from .hyper import Hypernetwork
from .tail import build_tail
from .init import uniform_between, linear_up
from utils import make_grid, interpolate_grid, repeat_vmap


class Thermal(nn.Module):
    w0_scale: float = 1.

    @nn.compact
    def __call__(self, x: ArrayLike, t, norm, k) -> Array:
        phase = self.param('phase', nn.initializers.uniform(.5), x.shape[-1:])
        return jnp.sin(self.w0_scale * x + phase) * jnp.exp(-(self.w0_scale * norm)**2 * k * t)


class TheraField(nn.Module):
    dim_hidden: int
    dim_out: int
    w0: float = 1.
    c: float = 6.

    @nn.compact
    def __call__(self, x: ArrayLike, t: ArrayLike, k: ArrayLike, components: ArrayLike) -> Array:
        # coordinate projection according to shared components ("first layer")
        x = x @ components

        # thermal activations
        norm = jnp.linalg.norm(components, axis=-2)
        x = Thermal(self.w0)(x, t, norm, k)

        # linear projection from hidden to output space ("second layer")
        w_std = math.sqrt(self.c / self.dim_hidden) / self.w0
        dense_init_fn = uniform_between(-w_std, w_std)
        x = nn.Dense(self.dim_out, kernel_init=dense_init_fn, use_bias=False)(x)

        return x


class Thera:

    def __init__(
            self,
            hidden_dim: int,
            out_dim: int,
            backbone: nn.Module,
            tail: nn.Module,
            k_init: float = None,
            components_init_scale: float = None
    ):
        self.hidden_dim = hidden_dim
        self.k_init = k_init
        self.components_init_scale = components_init_scale

        # single TheraField object whose `apply` method is used for all grid cells
        self.field = TheraField(hidden_dim, out_dim)

        # infer output size of the hypernetwork from a sample pass through the field;
        # key doesnt matter as field params are only used for size inference
        sample_params = self.field.init(jax.random.PRNGKey(0),
            jnp.zeros((2,)), 0., 0., jnp.zeros((2, hidden_dim)))
        sample_params_flat, tree_def = jax.tree_util.tree_flatten(sample_params)
        param_shapes = [p.shape for p in sample_params_flat]

        self.hypernet = Hypernetwork(backbone, tail, param_shapes, tree_def)

    def init(self, key, sample_source) -> PyTree:
        keys = jax.random.split(key, 2)
        sample_coords = jnp.zeros(sample_source.shape[:-1] + (2,))
        params = unfreeze(self.hypernet.init(keys[0], sample_source, sample_coords))

        params['params']['k'] = jnp.array(self.k_init)
        params['params']['components'] = \
            linear_up(self.components_init_scale)(keys[1], (2, self.hidden_dim))

        return freeze(params)

    def apply_encoder(self, params: PyTree, source: ArrayLike, **kwargs) -> Array:
        """
        Performs a forward pass through the hypernetwork to obtain an encoding.
        """
        return self.hypernet.apply(
            params, source, method=self.hypernet.get_encoding, **kwargs)

    def apply_decoder(
        self,
        params: PyTree,
        encoding: ArrayLike,
        coords: ArrayLike,
        t: ArrayLike,
        return_jac: bool = False
    ) -> Array | tuple[Array, Array]:
        """
        Performs a forward prediction through a grid of HxW Thera fields,
        informed by `encoding`, at spatial and temporal coordinates
        `coords` and `t`, respectively.
        args:
            params: Field parameters, shape (B, H, W, N)
            encoding: Encoding tensor, shape (B, H, W, C)
            coords: Spatial coordinates in [-0.5, 0.5], shape (B, H, W, 2)
            t: Temporal coordinates, shape (B, 1)
        """
        phi_params: PyTree = self.hypernet.apply(
            params, encoding, coords, method=self.hypernet.get_params_at_coords)

        # create local coordinate systems
        source_grid = jnp.asarray(make_grid(encoding.shape[-3:-1]))
        source_coords = jnp.tile(source_grid, (encoding.shape[0], 1, 1, 1))
        interp_coords = interpolate_grid(coords, source_coords)
        rel_coords = (coords - interp_coords)
        rel_coords = rel_coords.at[..., 0].set(rel_coords[..., 0] * encoding.shape[-3])
        rel_coords = rel_coords.at[..., 1].set(rel_coords[..., 1] * encoding.shape[-2])

        # three maps over params, coords; one over t; dont map k and components
        in_axes = [(0, 0, None, None, None), (0, 0, None, None, None), (0, 0, 0, None, None)]
        apply_field = repeat_vmap(self.field.apply, in_axes)
        out = apply_field(phi_params, rel_coords, t, params['params']['k'],
            params['params']['components'])

        if return_jac:
            apply_jac = repeat_vmap(jax.jacrev(self.field.apply, argnums=1), in_axes)
            jac = apply_jac(phi_params, rel_coords, jnp.zeros_like(t), params['params']['k'],
                params['params']['components'])
            return out, jac

        return out

    def apply(
        self,
        params: ArrayLike,
        source: ArrayLike,
        coords: ArrayLike,
        t: ArrayLike,
        return_jac: bool = False,
        **kwargs
    ) -> Array:
        """
        Performs a forward pass through the Thera model.
        """
        encoding = self.apply_encoder(params, source, **kwargs)
        out = self.apply_decoder(params, encoding, coords, t, return_jac=return_jac)
        return out


def build_thera(
    out_dim: int,
    backbone: str,
    size: str,
    k_init: float = None,
    components_init_scale: float = None
):
    """
    Convenience function for building the three Thera sizes described in the paper.
    """
    hidden_dim = 32 if size == 'air' else 512

    if backbone == 'edsr-baseline':
        backbone_module = EDSR(None, num_blocks=16, num_feats=64)
    elif backbone == 'rdn':
        backbone_module = RDN()
    else:
        raise NotImplementedError(backbone)

    tail_module = build_tail(size)

    return Thera(hidden_dim, out_dim, backbone_module, tail_module, k_init, components_init_scale)
