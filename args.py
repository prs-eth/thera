import configargparse
import numpy as np
import jax

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, type=str)

parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--save-dir', type=str, default=None)
parser.add_argument('--eval-sets', type=str, nargs='+', default=['DIV2K_val'])
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--eval-scales', type=int, nargs='+', default=[2, 3, 4, 6, 12, 18, 24, 30])
parser.add_argument('--no-geo-ensemble', action='store_true')
parser.add_argument('--y-only', action='store_true', help='Only evaluate Y channel of YCbCr image')
