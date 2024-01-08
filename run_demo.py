from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle

from torch.utils.data import DataLoader
from torchvision import transforms
import jax
import jax.numpy as jnp
from jax.image import resize
import numpy as np
from PIL import Image
from tqdm import tqdm

from data import ImageFolder
from model import Hypernetwork, TheraNet
from utils import make_grid, interpolate_grid

MEAN = np.array([.4488, .4371, .4040])
VAR = np.array([.25, .25, .25])
PATCH_SIZE = 16


def preprocess(source, scale):
    source = jnp.asarray(source).transpose((0, 2, 3, 1))

    target_coords = jnp.tile(make_grid(PATCH_SIZE * scale), (1, 1, 1, 1))
    target_t = jnp.float32(scale**(-2))

    target_shape = (source.shape[0], source.shape[1] * scale,
                    source.shape[2] * scale, source.shape[3])
    source_up = resize(source, target_shape, 'nearest')
    source = jax.nn.standardize(source, mean=MEAN, variance=VAR)

    return source, source_up, target_coords, target_t


def build_models(key, args):
    phi = TheraNet(args.thera_dim, 3, 1)

    key0, key1 = jax.random.split(key, num=2)

    # use sample parameter set to infer sizes of phi's parameters
    sample_params = phi.init(key0, np.ones((2,)), 1., 1., np.ones((1, 1, 2, args.thera_dim)))
    sample_params_flat, tree_def = jax.tree_util.tree_flatten(sample_params)
    param_sizes = [p.shape for p in sample_params_flat]

    hyper_net = Hypernetwork(args.backbone, param_sizes, tree_def, big_tail=args.big_tail, small_tail=args.small_tail)
    with open(args.checkpoint_path, 'rb') as fh:
        unpickled = pickle.load(fh)
        params = unpickled['model'] if 'model' in unpickled else unpickled

    return hyper_net.bind(params), phi


def process(data_loader, hyper_model, phi, scale, do_ensemble, include_flips, save_dir):
    hyper_model, params = hyper_model.unbind()

    @jax.jit
    def forward_encoder(params, source):
        return hyper_model.apply(params, source, method=hyper_model.get_encoding)

    @jax.jit
    def forward_decoder(params, encoding, target_coords, target_t):
        phi_params = hyper_model.apply(params, encoding, target_coords,
                                       method=hyper_model.get_params_at_coords)

        # create local coordinate system
        source_coords = jnp.tile(make_grid(encoding.shape[-2]), (encoding.shape[0], 1, 1, 1))
        interp_coords = interpolate_grid(target_coords, source_coords)
        rel_coords = (target_coords - interp_coords) * encoding.shape[-2]

        # vectorizing map over params and inputs, appending (N, H, W) as batch dims
        in_axes = (0, 0, 0, None, None)  # don't map k
        apply_phi_batched = jax.vmap(jax.vmap(jax.vmap(phi.apply, in_axes), in_axes), in_axes)
        out = apply_phi_batched(phi_params, rel_coords, jnp.tile(target_t, target_coords.shape[:-1]),
                                params['params']['k'], params['params']['components'])
        out = out * np.sqrt(VAR)[None, None, None] + MEAN[None, None, None]
        return out

    for i_img, source in enumerate(tqdm(data_loader)):
        source, source_up, target_coords, target_t = preprocess(source, scale)

        # TODO choose this adaptively
        assert PATCH_SIZE <= min(source.shape[1], source.shape[2])
        outs = []

        for i_rot in range(4 if do_ensemble else 1):
            for i_flip in range(2 if include_flips else 1):
                if i_flip == 1:
                    source_ = jnp.rot90(jnp.flip(source, axis=1), k=i_rot, axes=(-3, -2))
                    source_up_ = jnp.rot90(jnp.flip(source_up, axis=1), k=i_rot, axes=(-3, -2))
                else:
                    source_ = jnp.rot90(source, k=i_rot, axes=(-3, -2))
                    source_up_ = jnp.rot90(source_up, k=i_rot, axes=(-3, -2))
                encoding = forward_encoder(params, source_)
                assert encoding.shape[:-1] == source_.shape[:-1]

                num_patches_h = (source_.shape[1] // PATCH_SIZE) + 1
                num_patches_w = (source_.shape[2] // PATCH_SIZE) + 1
                out = np.full_like(source_up_, np.nan, dtype=np.float32)

                for i in range(num_patches_h):
                    for j in range(num_patches_w):
                        h_min = min(i * PATCH_SIZE, source_.shape[1] - PATCH_SIZE)
                        h_max = min((i + 1) * PATCH_SIZE, source_.shape[1])
                        w_min = min(j * PATCH_SIZE, source_.shape[2] - PATCH_SIZE)
                        w_max = min((j + 1) * PATCH_SIZE, source_.shape[2])
                        encoding_p = encoding[:, h_min:h_max, w_min:w_max, :]
                        out_p = forward_decoder(params, encoding_p, target_coords, target_t)
                        out[:, scale * h_min:scale * h_max, scale * w_min:scale * w_max, :] = out_p

                assert (not np.isnan(out).any())
                out += source_up_
                if i_flip == 1:
                    outs.append(np.flip(np.rot90(out, k=i_rot, axes=(-2, -3)), axis=1))
                else:
                    outs.append(np.rot90(out, k=i_rot, axes=(-2, -3)))

        out = np.stack(outs).mean(0).clip(0., 1.)

        if save_dir is not None:
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(np.array(out[0] * 255)).astype(np.uint8)) \
                .save(save_dir / f'{i_img}.png')


def main(args: Namespace):
    data_set = ImageFolder(Path(args.data_dir), transforms.ToTensor(), in_memory=False)
    data_loader = DataLoader(data_set, batch_size=1, num_workers=1, shuffle=False)

    key = jax.random.PRNGKey(42)
    hyper_model, phi = build_models(key, args)

    process(data_loader, hyper_model, phi, args.scale,
            args.geo_ensemble, args.geo_flip, Path(args.save_dir))


if __name__ == '__main__':
    parser = ArgumentParser()
    # TODO add single image option
    parser.add_argument('--data-dir', type=str, help='Directory of images to super-resolve')
    parser.add_argument('--scale', type=int, help='Scale factor for super-resolution')
    parser.add_argument('--save-dir', type=str, help='Directory of images to super-resolve')
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--backbone', type=str, default='edsr-baseline')
    parser.add_argument('--thera-dim', type=int, default=512)
    parser.add_argument('--big-tail', action='store_true')
    parser.add_argument('--small-tail', action='store_true')
    parser.add_argument('--geo-ensemble', action='store_true')
    parser.add_argument('--geo-flip', action='store_true', help='Add flips to geo-ensemble')
    args = parser.parse_args()

    main(args)
