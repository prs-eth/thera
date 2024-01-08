from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array


def uniform_between(a: float, b: float, dtype=jnp.float32) -> Callable:
    def init(key, shape, dtype=dtype) -> Array:
        return jax.random.uniform(key, shape, dtype=dtype, minval=a, maxval=b)
    return init


def linear_up(scale: float) -> Callable:
    def init(key, shape, dtype=jnp.float32) -> Array:
        assert shape[-2] == 2
        keys = jax.random.split(key, 2)
        norm = jnp.pi * scale * (
                jax.random.uniform(keys[0], shape=(1, shape[-1])) ** .5)
        theta = 2 * jnp.pi * jax.random.uniform(keys[1], shape=(1, shape[-1]))
        x = norm * jnp.cos(theta)
        y = norm * jnp.sin(theta)
        return jnp.concatenate([x, y], axis=-2).astype(dtype)
    return init
