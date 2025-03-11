import flax.linen as nn

from .convnext import ConvNeXt 
from .swin_ir import SwinIR


def build_tail(size: str):
    """ Convenience function to build the three tails described in the paper. """
    if size == 'air':
        return lambda x, _: x
    elif size == 'plus':
        blocks = [(64, 3, True)] * 6 + [(96, 3, True)] * 7 + [(128, 3, True)] * 3
        return ConvNeXt(blocks)
    elif size == 'pro':
        return SwinIR(depths=[7, 6], num_heads=[6, 6])
    else:
        raise NotImplementedError('size: ' + size)

