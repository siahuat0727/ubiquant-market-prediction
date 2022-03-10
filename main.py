from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import (UMPDataModule, df_to_input_feat, df_to_input_id,
                         load_data)
from litmodule import UMPLitModule


def get_name(args):
    return '-'.join(filter(None, [  # Remove empty string by filtering
        'x'.join(str(sz) for sz in args.szs),
        'x'.join(str(mha) for mha in args.mhas),
        f'epch{args.epochs}',
        f'opt{args.optimizer}',
        f'schd{args.lr_scheduler}',
        f'loss{args.loss}',
        f'lr{args.lr}',
        f'wd{args.weight_decay}',
        f'emb{args.emb_dim}',
    ])).replace(' ', '')


def do_submit(args, ckpts):

    litmodels = [
        UMPLitModule.load_from_checkpoint(ckpt_path, args=args).eval()
        for ckpt_path in ckpts
    ]

    import ubiquant
    env = ubiquant.make_env()   # initialize the environment

    for test_df, submit_df in env.iter_test():
        input_ids = df_to_input_id(test_df)
        input_feats = df_to_input_feat(test_df)

        with torch.no_grad():
            submit_df['target'] = torch.stack([
                litmodel.forward(input_ids, input_feats)
                for litmodel in litmodels
            ]).mean(dim=0)

        env.predict(submit_df)   # register your predictions


def run(args, seed, dataset):
    seed_everything(seed)

    # Model
    litmodel = (
        UMPLitModule(args)
        if args.checkpoint is None else
        UMPLitModule.load_from_checkpoint(args.checkpoint, args=args)
    )
    dm = UMPDataModule(args, dataset)
    if args.test:
        Trainer(gpus=args.n_gpu).test(litmodel, datamodule=dm)
        return

    name = get_name(args)
    logger = TensorBoardLogger(save_dir='tensorboard_logs', name=name)

    trainer = Trainer(gpus=args.n_gpu,
                      max_epochs=args.epochs,
                      deterministic=True,
                      logger=logger,
                      # precision=16,
                      )

    trainer.fit(litmodel, dm)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_result = trainer.test(ckpt_path=best_ckpt,
                               datamodule=dm)

    return best_ckpt, test_result[0]['test_pearson']


def parse_args(is_kaggle):
    parser = ArgumentParser()

    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument(
        '--input', default='../input/ubiquant-parquet/train_low_mem.parquet')

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr_scheduler', default=None)
    parser.add_argument('--loss', default='pcc', choices=['mse', 'pcc'])
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--n_fold', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Set to 0.0 to disable')
    parser.add_argument('--split_ratios', type=float,
                        default=[0.7, 0.15, 0.15],
                        help='train, val, and test set (optional) split ratio')

    # Model structure
    parser.add_argument('--n_emb', type=int, default=4000)  # TODO tight
    parser.add_argument('--szs', type=int, nargs='+',
                        default=[512, 256, 128, 64])
    parser.add_argument(
        '--mhas', type=int, nargs='+', default=[],
        help=('Insert MHA layer (BertLayer) at the i-th layer (start from 1). '
              'Every element should be <= len(szs)'))

    # Test
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--submit', action='store_true')

    # Checkpoint
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')

    # Handle kaggle platform
    args, unknown = parser.parse_known_args()

    if not is_kaggle:
        assert not unknown, f'unknown args: {unknown}'

    assert all(0 < i <= len(args.szs) for i in args.mhas)
    return args


def main():
    kaggle = False

    args = parse_args(kaggle)
    # On kaggle mode, we are using only the args with default value
    # To changle arguments, please hard code it below, e.g.:
    # args.loss = 'mse'
    # args.szs = [512, 128, 64, 64, 64]


    data = load_data(args.input)
    best_results = [
        run(args, seed, data)
        for seed in range(args.seed, args.seed + args.n_fold)
    ]
    del data

    test_pearsons = [pearson for _, pearson in best_results]
    print(f'{test_pearsons}, mean={sum(test_pearsons)/len(test_pearsons)}')

    submit = False
    if submit:
        best_ckpts = [ckpt for ckpt, _ in best_results]
        do_submit(args, best_ckpts)
    else:
        test_pearsons = [pearson for _, pearson in best_results]
        print(f'{test_pearsons}, mean={sum(test_pearsons)/len(test_pearsons)}')



if __name__ == '__main__':
    main()
