import flax.linen as nn
from jaxtyping import Array, ArrayLike


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


class ConvNeXt(nn.Module):
    block_defs: list[tuple]

    def setup(self) -> None:
        layers = []
        current_size = self.block_defs[0][0]
        for block_def in self.block_defs:
            if block_def[0] != current_size:
                layers.append(Projection(block_def[0]))
            layers.append(ConvNeXtBlock(*block_def))
            current_size = block_def[0]
        self.layers = layers

    def __call__(self, x: ArrayLike, _: bool) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x

