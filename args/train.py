import configargparse
import numpy as np
import jax

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, type=str)

# training
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--n-devices', type=int, default=jax.local_device_count())
parser.add_argument('--n-iter', type=int, default=5_000_000)
parser.add_argument('--val-every', type=int, default=1_000)
parser.add_argument('--val-samples', type=int, default=256)
parser.add_argument('--patch-size', type=int, default=48, help='Image size at t=1.0')
parser.add_argument('--scale-range', type=float, nargs='+', default=(1.2, 4.))
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--loss', type=str, default='MAE')
parser.add_argument('--tv-weight', type=float, default=1e-4, help='Set to zero for no TV')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save-every', type=int, default=100_000)
parser.add_argument('--max-grad-norm', type=float, default=10.)
parser.add_argument('--augment-scale-range', type=float, nargs='+', default=(1., 2.0))
parser.add_argument('--augment-scale-prob', type=float, default=0.5)

# model
parser.add_argument('--backbone', type=str, default='edsr-baseline')
parser.add_argument('--init-k', type=float, default=np.sqrt(np.log(4)) / (np.pi ** 2 * 2))
parser.add_argument('--init-scale', type=float, default=16.0)
parser.add_argument('--size', type=str, default='pro')

# data
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--train-set', type=str, default='DIV2K_train')
parser.add_argument('--val-set', type=str, default='DIV2K_dev_val')
parser.add_argument('--no-wandb', action='store_true')
parser.add_argument('--wandb-project', type=str, default='thera')
parser.add_argument('--wandb-dir', type=str, default='../logs')
parser.add_argument('--tag', type=str, default='', help='Tag to append to checkpoint file name')
