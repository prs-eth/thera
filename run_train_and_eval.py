from datetime import datetime
from pathlib import Path

from args.train import parser as train_parser
from args.eval import parser as eval_parser
from run_train import main as run_train
from run_eval import main as run_eval


if __name__ == '__main__':
    args = train_parser.parse_args()
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    checkpoint_path = Path(args.wandb_dir) / \
        (f'params_latest_{timestamp}' + (f'-{args.tag}.pkl' if args.tag else '.pkl'))
    args.checkpoint = checkpoint_path
    print('Training args:\n', train_parser.format_values())

    run_train(args)

    print('Training finished, running evaluation now.')

    args = eval_parser.parse_args()
    args.checkpoint = checkpoint_path
    print('Eval args:\n', eval_parser.format_values())

    run_eval(args)
