import math
from typing import Callable, Optional, Iterable

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array


def trunc_normal(mean=0., std=1., a=-2., b=2., dtype=jnp.float32) -> Callable:
    """Truncated normal initialization function"""

    def init(key, shape, dtype=dtype) -> Array:
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        out = jax.random.uniform(key, shape, dtype=dtype, minval=2 * l - 1, maxval=2 * u - 1)
        out = jax.scipy.special.erfinv(out) * std * math.sqrt(2.) + mean
        return jnp.clip(out, a, b)

    return init


def Dense(features, use_bias=True, kernel_init=trunc_normal(std=.02), bias_init=nn.initializers.zeros):
    return nn.Dense(features, use_bias=use_bias, kernel_init=kernel_init, bias_init=bias_init)


def LayerNorm():
    """torch LayerNorm uses larger epsilon by default"""
    return nn.LayerNorm(epsilon=1e-05)


class Mlp(nn.Module):

    in_features: int
    hidden_features: int = None
    out_features: int = None
    act_layer: Callable = nn.gelu
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.hidden_features or self.in_features)(x)
        x = self.act_layer(x)
        x = nn.Dropout(self.drop, deterministic=not training)(x)
        x = nn.Dense(self.out_features or self.in_features)(x)
        x = nn.Dropout(self.drop, deterministic=not training)(x)
        return x


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape((B, H // window_size, window_size, W // window_size, window_size, C))
    windows = x.transpose((0, 1, 3, 2, 4, 5)).reshape((-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape((B, H // window_size, W // window_size, window_size, window_size, -1))
    x = x.transpose((0, 1, 3, 2, 4, 5)).reshape((B, H, W, -1))
    return x


class DropPath(nn.Module):
    """
    Implementation referred from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    dropout_prob: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, input, training):
        if not training:
            return input
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        rng = self.make_rng("dropout")
        random_tensor = keep_prob + jax.random.uniform(rng, shape)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(input, keep_prob) * random_tensor


class WindowAttention(nn.Module):
    dim: int
    window_size: Iterable[int]
    num_heads: int
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    att_drop: float = 0.0
    proj_drop: float = 0.0

    def make_rel_pos_index(self):
        h_indices = np.arange(0, self.window_size[0])
        w_indices = np.arange(0, self.window_size[1])
        indices = np.stack(np.meshgrid(w_indices, h_indices, indexing="ij"))
        flatten_indices = np.reshape(indices, (2, -1))
        relative_indices = flatten_indices[:, :, None] - flatten_indices[:, None, :]
        relative_indices = np.transpose(relative_indices, (1, 2, 0))
        relative_indices[:, :, 0] += self.window_size[0] - 1
        relative_indices[:, :, 1] += self.window_size[1] - 1
        relative_indices[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_pos_index = np.sum(relative_indices, -1)
        return relative_pos_index

    @nn.compact
    def __call__(self, inputs, mask, training):
        rpbt = self.param(
            "relative_position_bias_table",
            trunc_normal(std=.02),
            (
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ),
        )

        #relative_pos_index = self.variable(
        #    "variables", "relative_position_index", self.get_rel_pos_index
        #)

        batch, n, channels = inputs.shape
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")(inputs)
        qkv = qkv.reshape(batch, n, 3, self.num_heads, channels // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.qk_scale or (self.dim // self.num_heads) ** -0.5
        q = q * scale
        att = q @ jnp.swapaxes(k, -2, -1)

        rel_pos_bias = jnp.reshape(
            rpbt[np.reshape(self.make_rel_pos_index(), (-1))],
            (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ),
        )
        rel_pos_bias = jnp.transpose(rel_pos_bias, (2, 0, 1))
        att += jnp.expand_dims(rel_pos_bias, 0)

        if mask is not None:
            att = jnp.reshape(
                att, (batch // mask.shape[0], mask.shape[0], self.num_heads, n, n)
            )
            att = att + jnp.expand_dims(jnp.expand_dims(mask, 1), 0)
            att = jnp.reshape(att, (-1, self.num_heads, n, n))
            att = jax.nn.softmax(att)

        else:
            att = jax.nn.softmax(att)

        att = nn.Dropout(self.att_drop)(att, deterministic=not training)

        x = jnp.reshape(jnp.swapaxes(att @ v, 1, 2), (batch, n, channels))
        x = nn.Dense(self.dim, name="proj")(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic=not training)
        return x


class SwinTransformerBlock(nn.Module):

    dim: int
    input_resolution: tuple[int]
    num_heads: int
    window_size: int = 7
    shift_size: int = 0
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.
    act_layer: Callable = nn.activation.gelu
    norm_layer: Callable = LayerNorm

    @staticmethod
    def make_att_mask(shift_size, window_size, height, width):
        if shift_size > 0:
            mask = jnp.zeros([1, height, width, 1])
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )

            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask = mask.at[:, h, w, :].set(count)
                    count += 1

            mask_windows = window_partition(mask, window_size)
            mask_windows = jnp.reshape(mask_windows, (-1, window_size * window_size))
            att_mask = jnp.expand_dims(mask_windows, 1) - jnp.expand_dims(mask_windows, 2)
            att_mask = jnp.where(att_mask != 0.0, float(-100.0), att_mask)
            att_mask = jnp.where(att_mask == 0.0, float(0.0), att_mask)
        else:
            att_mask = None

        return att_mask

    @nn.compact
    def __call__(self, x, x_size, training):
        H, W = x_size
        B, L, C = x.shape

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        shortcut = x
        x = self.norm_layer()(x)
        x = x.reshape((B, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = jnp.roll(x, (-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape((-1, self.window_size * self.window_size, C))  # nW*B, window_size*window_size, C

        #attn_mask = self.variable(
        #    "variables",
        #    "attn_mask",
        #    self.get_att_mask,
        #    self.shift_size,
        #    self.window_size,
        #    self.input_resolution[0],
        #    self.input_resolution[1]
        #)

        attn_mask = self.make_att_mask(self.shift_size, self.window_size, *self.input_resolution)

        attn = WindowAttention(self.dim, (self.window_size, self.window_size), self.num_heads,
                               self.qkv_bias, self.qk_scale, self.attn_drop, self.drop)
        if self.input_resolution == x_size:
            attn_windows = attn(x_windows, attn_mask, training)  # nW*B, window_size*window_size, C
        else:
            # test time
            assert not training
            test_mask = self.make_att_mask(self.shift_size, self.window_size, *x_size)
            attn_windows = attn(x_windows, test_mask, training=False)

        # merge windows
        attn_windows = attn_windows.reshape((-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = jnp.roll(shifted_x, (self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        x = x.reshape((B, H * W, C))

        # FFN
        x = shortcut + DropPath(self.drop_path)(x, training)

        norm = self.norm_layer()(x)
        mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim * self.mlp_ratio),
                  act_layer=self.act_layer, drop=self.drop)(norm, training)
        x = x + DropPath(self.drop_path)(mlp, training)

        return x


class PatchMerging(nn.Module):
    inp_res: Iterable[int]
    dim: int
    norm_layer: Callable = LayerNorm

    @nn.compact
    def __call__(self, inputs):
        batch, n, channels = inputs.shape
        height, width = self.inp_res[0], self.inp_res[1]
        x = jnp.reshape(inputs, (batch, height, width, channels))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = jnp.concatenate([x0, x1, x2, x3], axis=-1)
        x = jnp.reshape(x, (batch, -1, 4 * channels))
        x = self.norm_layer()(x)
        x = nn.Dense(2 * self.dim, use_bias=False)(x)
        return x


class BasicLayer(nn.Module):

    dim: int
    input_resolution: int
    depth: int
    num_heads: int
    window_size: int
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.
    norm_layer: Callable = LayerNorm
    downsample: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, x_size, training):
        for i in range(self.depth):
            x = SwinTransformerBlock(
                self.dim,
                self.input_resolution,
                self.num_heads,
                self.window_size,
                0 if (i % 2 == 0) else self.window_size // 2,
                self.mlp_ratio,
                self.qkv_bias,
                self.qk_scale,
                self.drop,
                self.attn_drop,
                self.drop_path[i] if isinstance(self.drop_path, (list, tuple)) else self.drop_path,
                norm_layer=self.norm_layer
            )(x, x_size, training)

        if self.downsample is not None:
            x = self.downsample(self.input_resolution, dim=self.dim, norm_layer=self.norm_layer)(x)

        return x


class RSTB(nn.Module):

    dim: int
    input_resolution: int
    depth: int
    num_heads: int
    window_size: int
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.
    norm_layer: Callable = LayerNorm
    downsample: Optional[Callable] = None
    img_size: int = 224,
    patch_size: int = 4,
    resi_connection: str = '1conv'

    @nn.compact
    def __call__(self, x, x_size, training):
        res = x
        x = BasicLayer(dim=self.dim,
                       input_resolution=self.input_resolution,
                       depth=self.depth,
                       num_heads=self.num_heads,
                       window_size=self.window_size,
                       mlp_ratio=self.mlp_ratio,
                       qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                       drop=self.drop, attn_drop=self.attn_drop,
                       drop_path=self.drop_path,
                       norm_layer=self.norm_layer,
                       downsample=self.downsample)(x, x_size, training)

        x = PatchUnEmbed(embed_dim=self.dim)(x, x_size)

        # resi_connection == '1conv':
        x = nn.Conv(self.dim, (3, 3))(x)

        x = PatchEmbed()(x)

        return x + res


class PatchEmbed(nn.Module):
    norm_layer: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1, x.shape[-1]))  # B Ph Pw C -> B Ph*Pw C
        if self.norm_layer is not None:
            x = self.norm_layer()(x)
        return x


class PatchUnEmbed(nn.Module):
    embed_dim: int = 96

    @nn.compact
    def __call__(self, x, x_size):
        B, HW, C = x.shape
        x = x.reshape((B, x_size[0], x_size[1], self.embed_dim))
        return x


class SwinIR(nn.Module):
    r""" SwinIR JAX implementation
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 25I think5.
    """

    img_size: int = 48
    patch_size: int = 1
    in_chans: int = 3
    embed_dim: int = 180
    depths: tuple = (6, 6, 6, 6, 6, 6)
    num_heads: tuple = (6, 6, 6, 6, 6, 6)
    window_size: int = 8
    mlp_ratio: float = 2.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.1
    norm_layer: Callable = LayerNorm
    ape: bool = False
    patch_norm: bool = True
    upscale: int = 2
    img_range: float = 1.
    num_feat: int = 64

    def pad(self, x):
        _, h, w, _ = x.shape
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = jnp.pad(x, ((0, 0), (0, mod_pad_h), (0, mod_pad_w), (0, 0)), 'reflect')
        return x

    @nn.compact
    def __call__(self, x, training):
        _, h_before, w_before, _ = x.shape
        x = self.pad(x)
        _, h, w, _ = x.shape
        patches_resolution = [self.img_size // self.patch_size] * 2
        num_patches = patches_resolution[0] * patches_resolution[1]

        # conv_first
        x = nn.Conv(self.embed_dim, (3, 3))(x)
        res = x

        # feature extraction
        x_size = (h, w)
        x = PatchEmbed(self.norm_layer if self.patch_norm else None)(x)

        if self.ape:
            absolute_pos_embed = \
                self.param('ape', trunc_normal(std=.02), (1, num_patches, self.embed_dim))
            x = x + absolute_pos_embed

        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)

        dpr = [x.item() for x in np.linspace(0, self.drop_path_rate, sum(self.depths))]
        for i_layer in range(len(self.depths)):
            x = RSTB(
                dim=self.embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=None,
                img_size=self.img_size,
                patch_size=self.patch_size)(x, x_size, training)

        x = self.norm_layer()(x)  # B L C
        x = PatchUnEmbed(self.embed_dim)(x, x_size)

        # conv_after_body
        x = nn.Conv(self.embed_dim, (3, 3))(x)
        x = x + res

        # conv_before_upsample
        x = nn.activation.leaky_relu(nn.Conv(self.num_feat, (3, 3))(x))

        # revert padding
        x = x[:, :-(h - h_before) or None, :-(w - w_before) or None]
        return x
