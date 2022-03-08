from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import UMPDataModule
from litmodule import UMPLitModule


def get_name(args):
    return '-'.join(filter(None, [  # Remove empty string by filtering
        'x'.join(str(sz) for sz in args.szs),
        f'epch{args.epochs}',
        f'opt{args.optimizer}',
        f'lrschd{args.lr_scheduler}',
        f'lr{args.lr}',
        f'wd{args.weight_decay}',
        f'emb{args.emb_dim}',
    ])).replace(' ', '')


def main(args):
    seed_everything(args.seed)

    # Model
    litmodel = (
        UMPLitModule(args)
        if args.checkpoint is None else
        UMPLitModule.load_from_checkpoint(args.checkpoint, args=args)
    )

    name = get_name(args)
    logger = TensorBoardLogger(save_dir='tensorboard_logs', name=name)

    kwargs = {}


    dm = UMPDataModule(args)
    if not args.test:
        trainer = Trainer(gpus=args.n_gpu,
                          max_epochs=args.epochs,
                          deterministic=True,
                          logger=logger,
                          **kwargs,
                          )
        trainer.fit(litmodel, dm)
        trainer.test(litmodel, datamodule=dm)
    else:
        Trainer(gpus=args.n_gpu).test(litmodel, datamodule=dm)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--input', default='../input/ubiquant-parquet/train_low_mem.parquet')

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr_scheduler', default=None)

    # Model structure
    parser.add_argument('--n_emb', type=int, default=4000)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--szs', type=int, nargs='+',
                        default=[512, 256, 128, 64])

    # Test
    parser.add_argument('--test', action='store_true')

    # Checkpoint
    parser.add_argument('--checkpoint', help='path to checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
