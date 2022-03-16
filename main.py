from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import (UMPDataModule, df_to_input_feat, df_to_input_id,
                         load_data)
from litmodule import UMPLitModule, UMPLitModuleMem


def get_name(args):
    return '-'.join(filter(None, [  # Remove empty string by filtering
        'x'.join(str(sz) for sz in args.szs),
        'x'.join(str(mha) for mha in args.mhas),
        f'n_mem{args.n_mem}',
        f'epch{args.max_epochs}',
        f'btch{args.batch_size}x{args.accumulate_grad_batches}',
        f'{args.optimizer}',
        f'drop{args.dropout}',
        f'schd{args.lr_scheduler}',
        f'loss{args.loss}',
        f'lr{args.lr}',
        f'wd{args.weight_decay}',
        f'swa{args.swa}',
        f'emb{args.emb_dim}',
    ])).replace(' ', '')


def get_litmodule_cls(args):
    if args.n_mem > 0:
        return UMPLitModuleMem
    return UMPLitModule


def get_litmodule(args):
    cls = get_litmodule_cls(args)
    return cls(args)


def submit(args, ckpts):

    litmodels = [
        get_litmodule_cls(args).load_from_checkpoint(ckpt, args=args).eval()
        for ckpt in ckpts
    ]

    import ubiquant
    env = ubiquant.make_env()   # initialize the environment

    for test_df, submit_df in env.iter_test():
        input_ids = df_to_input_id(test_df).unsqueeze(0)
        input_feats = df_to_input_feat(test_df).unsqueeze(0)

        with torch.no_grad():
            submit_df['target'] = torch.cat([
                litmodel.forward(input_ids, input_feats)
                for litmodel in litmodels
            ]).mean(dim=0)

        env.predict(submit_df)   # register your predictions


def test(args):
    seed_everything(args.seed)

    litmodel = get_litmodule_cls(args).load_from_checkpoint(args.checkpoint,
                                                            args=args)
    dm = UMPDataModule(args)

    Trainer.from_argparse_args(args).test(litmodel, datamodule=dm)


def train_single(args, seed):
    seed_everything(seed)

    litmodel = get_litmodule(args)
    dm = UMPDataModule(args)

    name = get_name(args)
    logger = TensorBoardLogger(save_dir='logs', name=name)

    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         deterministic=True,
                                         precision=16)
    trainer.fit(litmodel, dm)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_result = trainer.test(ckpt_path=best_ckpt,
                               datamodule=dm)

    return {
        'ckpt_path': best_ckpt,
        'test_pearson': test_result[0]['test_pearson']
    }


def train(args):
    return [
        train_single(args, seed)
        for seed in range(args.seed, args.seed + args.n_fold)
    ]


def parse_args(is_kaggle=False):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--workers', type=int, default=2,
                        help='# of workers')
    parser.add_argument(
        '--input', default='../input/ubiquant-parquet/train_low_mem.parquet',
        help='path to train data')

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam', 'adamw'],
                        help='optimizer')
    parser.add_argument('--lr_scheduler', default=None,
                        choices=['plateau'],
                        help='learning rate scheduler')
    parser.add_argument('--loss', default='pcc', choices=['mse', 'pcc'],
                        help='loss function')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='investment embedding dimension')
    parser.add_argument('--n_fold', type=int, default=1,
                        help='Number of folds')
    parser.add_argument('--split_ratios', type=float, nargs='+',
                        default=[0.7, 0.15, 0.15],
                        help='train, val, and test set (optional) split ratio')
    parser.add_argument('--early_stop', action='store_true',
                        help='whether to early stop')
    parser.add_argument('--swa', action='store_true',
                        help='whether to perform Stochastic Weight Averaging')

    # Model structure
    parser.add_argument('--szs', type=int, nargs='+',
                        default=[512, 256, 128, 64],
                        help='sizes of each layer')
    parser.add_argument(
        '--mhas', type=int, nargs='+', default=[],
        help=('Insert MHA layer (BertLayer) at the i-th layer (start from 0). '
              'Every element should be 0 <= * < len(szs)'))
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate, set to 0.0 to disable')
    parser.add_argument('--n_mem', type=int, default=0,
                        help='# of memory tokens for mha, set to 0 to disable')

    # Test
    parser.add_argument('--test', action='store_true',
                        help='whether to test')

    # Checkpoint
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')

    # Handle kaggle platform
    args, unknown = parser.parse_known_args()

    if not is_kaggle:
        assert not unknown, f'unknown args: {unknown}'

    assert all(0 <= i < len(args.szs) for i in args.mhas)

    args.with_memory = args.n_mem > 0
    if args.with_memory:
        assert len(args.mhas) == 1, 'Currently support one mha with memory'
    return args


def run_local():
    args = parse_args()

    if args.test:
        test(args)
        return

    best_results = train(args)
    test_pearsons = [res['test_pearson'] for res in best_results]
    print(f'mean={sum(test_pearsons)/len(test_pearsons)}, {test_pearsons}')
    print(best_results)


def kaggle():
    args = parse_args(True)
    # On kaggle mode, we are using only the args with default value
    # To changle arguments, please hard code it below, e.g.:
    # args.loss = 'mse'
    # args.szs = [512, 128, 64, 64, 64]

    args.max_epochs = 20
    args.gpus = 1

    do_submit = False
    train_on_kaggle = False

    if train_on_kaggle:
        assert args.n_fold == 1, (
            'Otherwise it will meet out of memory problem. '
            'Probably memory leak problem on kaggle.'
        )
        best_results = train(args)
        best_ckpts = [ckpt for ckpt, _ in best_results]

        test_pearsons = [res['test_pearson'] for res in best_results]
        print(f'mean={sum(test_pearsons)/len(test_pearsons)}, {test_pearsons}')
    else:
        # TODO fill in the ckpt paths
        best_ckpts = []

    assert best_ckpts

    if do_submit:
        submit(args, best_ckpts)


if __name__ == '__main__':
    is_kaggle = False
    if is_kaggle:
        kaggle()
    else:
        run_local()
