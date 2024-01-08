from datetime import datetime
from pathlib import Path

from args import parser
from run_train import main as run_train
from run_eval import main as run_eval


if __name__ == '__main__':
    args = parser.parse_args()
    # append checkpoint path to config values
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    args.checkpoint_path = Path(args.wandb_dir) / (f'params_latest_{timestamp}' +
                                                   (f'-{args.tag}.pkl' if args.tag else '.pkl'))
    print(parser.format_values())

    run_train(args)

    print('Training finished, running evaluation now.')

    # checkpoint path was automatically set
    run_eval(args)
