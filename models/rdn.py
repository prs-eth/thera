# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import jax.numpy as jnp
import flax.linen as nn


class RDB_Conv(nn.Module):
    growRate: int
    kSize: int = 3

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential([
            nn.Conv(self.growRate, (self.kSize, self.kSize), padding=(self.kSize-1)//2),
            nn.activation.relu
        ])(x)
        return jnp.concatenate((x, out), -1)


class RDB(nn.Module):
    growRate0: int
    growRate: int
    nConvLayers: int

    @nn.compact
    def __call__(self, x):
        res = x

        for c in range(self.nConvLayers):
            x = RDB_Conv(self.growRate)(x)

        x = nn.Conv(self.growRate0, (1, 1))(x)

        return x + res


class RDN(nn.Module):
    G0: int = 64
    RDNkSize: int = 3
    RDNconfig: str = 'B'
    scale: int = 2
    n_colors: int = 3

    @nn.compact
    def __call__(self, x, _=None):
        D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[self.RDNconfig]

        # Shallow feature extraction
        f_1 = nn.Conv(self.G0, (self.RDNkSize, self.RDNkSize))(x)
        x = nn.Conv(self.G0, (self.RDNkSize, self.RDNkSize))(f_1)

        # Redidual dense blocks and dense feature fusion
        RDBs_out = []
        for i in range(D):
            x = RDB(self.G0, G, C)(x)
            RDBs_out.append(x)

        x = jnp.concatenate(RDBs_out, -1)

        # Global Feature Fusion
        x = nn.Sequential([
            nn.Conv(self.G0, (1, 1)),
            nn.Conv(self.G0, (self.RDNkSize, self.RDNkSize))
        ])(x)

        x = x + f_1
        return x
