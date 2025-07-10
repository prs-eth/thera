import pickle
from collections import defaultdict
from functools import partial
from itertools import islice
from pathlib import Path
from datetime import datetime

import numpy as np
import jax
from jax import value_and_grad, pmap
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb

from args.train import parser
from model import build_thera
from utils import seed_all, RandomRotate, split, compute_metrics
from data import ArbitraryScaleWrapper, ImageFolder, numpy_collate

MEAN = np.array([.4488, .4371, .4040])
VAR = np.array([.25, .25, .25])


def train(model, params, data_loaders, optimizer, args):
    opt_state = optimizer.init(params)

    def forward(params, batch, **kwargs):
        source = jax.nn.standardize(batch['source'], mean=MEAN, variance=VAR)
        inputs = (params, source, batch['target_coords'], batch['scale']**-2)
        out = model.apply(*inputs, **kwargs, return_jac=args.tv_weight > 0.)
        out, jac = out if isinstance(out, tuple) else (out, None)
        out = out * jnp.sqrt(VAR) + MEAN
        return out + batch['source_nearest'], jac

    @partial(pmap, axis_name='num_devices')
    def train_step(batch, params, opt_state, key):
        def get_loss_and_metrics(params):
            out, jac = forward(params, batch, training=True, rngs={'dropout': key})
            metrics = compute_metrics(out, batch['target'], jac)
            loss = metrics[args.loss] + args.tv_weight * metrics.get('TV', 0.)
            return loss, metrics

        (_, metrics), grads = value_and_grad(get_loss_and_metrics, has_aux=True)(params)

        # combine gradients and metrics from all devices
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        metrics = jax.lax.pmean(metrics, axis_name='num_devices')

        # parameter updates happen on each device individually
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return metrics, params, opt_state

    i_start = 0
    if args.resume:
        print(f'Resuming from checkpoint {args.resume}')
        with open(args.resume, 'rb') as fh:
            checkpoint = pickle.load(fh)
            params = checkpoint['model']
            opt_state = checkpoint['optimizer']
            i_start = int(opt_state[1][2].count)

    # replicate params, opt_state accross devices
    params = tree_map(lambda x: jnp.array([x] * args.n_devices), params)
    opt_state = tree_map(lambda x: jnp.array([x] * args.n_devices), opt_state)

    train_loader, val_loader = data_loaders
    train_metrics = defaultdict(list)

    def inf_train_loader():
        while True:
            yield from train_loader

    # main training loop
    for i, batch in (pbar := tqdm(zip(range(i_start, args.n_iter), inf_train_loader()),
                                  total=args.n_iter, initial=i_start)):
        batch = tree_map(partial(split, n_devices=args.n_devices), batch)
        keys = jnp.stack(jax.random.split(jax.random.PRNGKey(i), args.n_devices))
        batch_metrics, params, opt_state = train_step(batch, params, opt_state, keys)

        for k, v in batch_metrics.items():
            train_metrics[k] += [v[0].item()]

        if i % args.val_every == 0 and i > 0:
            # do validation
            val_metrics = defaultdict(list)
            for batch in islice(val_loader, args.val_samples // args.batch_size):
                batch = tree_map(partial(split, n_devices=args.n_devices), batch)
                out, jac = pmap(forward)(params, batch)
                batch_metrics = compute_metrics(out, batch['target'], jac)
                for k, v in batch_metrics.items():
                    val_metrics[k] += [v.item()]

            if not args.no_wandb:
                wandb.log({k + '/train': np.mean(v) for k, v in train_metrics.items()}, i)
                wandb.log({k + '/val': np.mean(v) for k, v in val_metrics.items()}, i)

            pbar.set_postfix_str(f'Val {args.loss}: {np.mean(val_metrics[args.loss]).round(3)}')
            train_metrics = defaultdict(list)

        if (i % args.save_every == 0 and i > 0) or i == args.n_iter - 1:
            with open(args.checkpoint, 'wb') as fh:
                pickle.dump({
                    'model': jax.device_get(tree_map(lambda x: x[0], params)),
                    'optimizer': jax.device_get(tree_map(lambda x: x[0], opt_state))
                }, fh)


def main(args):
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, dir=args.wandb_dir)
        wandb.config.update(args)

    seed_all(args.seed)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRotate([0, 90, 180, 270])
    ])
    data_sets = [ImageFolder(Path(args.data_dir) / s) for s in (args.train_set, args.val_set)]
    data_sets = [ArbitraryScaleWrapper(ds, args.patch_size, args.scale_range, args.augment_scale_range,
                 args.augment_scale_prob, transform) for ds in data_sets]

    data_loaders = [
        DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                   drop_last=True, persistent_workers=args.num_workers > 0, collate_fn=numpy_collate)
        for ds in data_sets
    ]

    sample_source = next(iter(data_loaders[0]))['source']
    model = build_thera(3, args.backbone, args.size, args.init_k, args.init_scale)
    params = model.init(jax.random.PRNGKey(args.seed), sample_source)
    print('# params: ', sum(x.size for x in jax.tree_util.tree_leaves(params)))

    schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=args.n_iter)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adamw(schedule))

    train(model, params, data_loaders, optimizer, args)


if __name__ == '__main__':
    args = parser.parse_args()
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    args.checkpoint = (Path(args.wandb_dir) /
        (f'params_latest_{timestamp}' + (f'-{args.tag}.pkl' if args.tag else '.pkl')))
    print(parser.format_values())

    main(args)
