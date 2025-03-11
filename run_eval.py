import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from jax.image import resize
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from args import parser
from data import ImageFolder
from model import build_thera
from utils import make_grid, compute_metrics
from vendor.matlab_bicubic import imresize as matlab_imresize

MEAN = np.array([.4488, .4371, .4040])
VAR = np.array([.25, .25, .25])
MAX_PATCH_SIZE = 256


def prepare_batch(target, scale):
    target = jnp.asarray(target)
    target = target.transpose((0, 2, 3, 1))

    source_h, source_w = int(target.shape[1] / scale), int(target.shape[2] / scale)
    target = target[:, :source_h * scale, :source_w * scale]
    target_t = jnp.float32(scale**(-2))[None]

    source = matlab_imresize(target[0], output_shape=(source_h, source_w))[None]
    source_up = resize(source, target.shape, 'nearest')
    source = jax.nn.standardize(source, mean=MEAN, variance=VAR)

    return source, source_up, target_t, target


def evaluate(val_loader, model, params, scale, border_crop,
             do_ensemble, save_dir: Optional[Path] = None, y_only=False):

    apply_encoder = jit(model.apply_encoder)
    apply_decoder = jit(model.apply_decoder)

    metrics = defaultdict(list)
    for i_img, target in enumerate(tqdm(val_loader)):
        source, source_up, target_t, target = prepare_batch(target, scale)

        # memory scales in patch_size * scale, so we keep that factor constant
        patch_size = MAX_PATCH_SIZE // scale
        if patch_size > min(source.shape[1:3]):
            patch_size = min(source.shape[1:3])

        target_coords = jnp.tile(make_grid(patch_size * scale), (target.shape[0], 1, 1, 1))

        outs = []

        for i_rot in range(4 if do_ensemble else 1):
            source_ = jnp.rot90(source, k=i_rot, axes=(-3, -2))
            source_up_ = jnp.rot90(source_up, k=i_rot, axes=(-3, -2))
            encoding = apply_encoder(params, source_)
            assert encoding.shape[:-1] == source_.shape[:-1]

            num_patches_h = (source_.shape[1] // patch_size) + 1
            num_patches_w = (source_.shape[2] // patch_size) + 1
            out = np.full_like(source_up_, np.nan, dtype=np.float32)

            for i, j in product(range(num_patches_h), range(num_patches_w)):
                h_min = min(i * patch_size, source_.shape[1] - patch_size)
                h_max = min((i + 1) * patch_size, source_.shape[1])
                w_min = min(j * patch_size, source_.shape[2] - patch_size)
                w_max = min((j + 1) * patch_size, source_.shape[2])
                encoding_p = encoding[:, h_min:h_max, w_min:w_max, :]
                out_p = apply_decoder(params, encoding_p, target_coords, target_t)
                out[:, scale * h_min:scale * h_max, scale * w_min:scale * w_max, :] = out_p

            assert not np.isnan(out).any()
            out = out * np.sqrt(VAR)[None, None, None] + MEAN[None, None, None]
            out += source_up_
            outs.append(np.rot90(out, k=i_rot, axes=(-2, -3)))

        out = np.stack(outs).mean(0).clip(0., 1.)

        if save_dir is not None:
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(np.array(out[0] * 255)).astype(np.uint8))\
                .save(save_dir / f'{i_img}.png')

        s = border_crop
        batch_metrics = compute_metrics(
            out[:, s:-s, s:-s], target[:, s:-s, s:-s], compute_ssim=True, y_only=y_only)
        for k, v in batch_metrics.items():
            metrics[k] += [v.item()]

    return {k: np.mean(v) for k, v in metrics.items()}


def main(args):
    data_sets = [ImageFolder(Path(args.data_dir) / s, transforms.ToTensor(), in_memory=False)
                 for s in args.eval_sets]
    data_loaders = [DataLoader(s, batch_size=1, num_workers=0, shuffle=False) for s in data_sets]

    with open(args.checkpoint, 'rb') as fh:
        check = pickle.load(fh)
        params, backbone, size = check['model'], check['backbone'], check['size']

    model = build_thera(3, backbone, size)

    for eval_set, data_loader in zip(args.eval_sets, data_loaders):
        for scale in args.eval_scales:
            border_crop = scale + 6 if 'DIV2K' in eval_set else scale
            save_dir = (Path(args.save_dir) / ('ours_' + eval_set + '_' + args.backbone) / str(scale)) \
                if args.save_dir else None

            metrics = evaluate(data_loader, model, params, scale, border_crop,
                not args.no_geo_ensemble, save_dir, args.y_only)

            metrics = {k: np.round(v, 5) for k, v in metrics.items()}
            print(f'[{eval_set} x{scale}] ' + ' '.join([f'{k}: {v}' for k, v in metrics.items()]))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
