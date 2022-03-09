from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import UMPDataModule, df_to_input_feat, df_to_input_id
from litmodule import UMPLitModule


def get_name(args):
    return '-'.join(filter(None, [  # Remove empty string by filtering
        'x'.join(str(sz) for sz in args.szs),
        f'epch{args.epochs}',
        f'opt{args.optimizer}',
        f'lrschd{args.lr_scheduler}',
        f'loss{args.loss}',
        f'lr{args.lr}',
        f'wd{args.weight_decay}',
        f'emb{args.emb_dim}',
    ])).replace(' ', '')


def submit(litmodel):
    litmodel.eval()
    assert not litmodel.model.training

    import ubiquant
    env = ubiquant.make_env()   # initialize the environment

    for test_df, submit_df in env.iter_test():
        input_ids = df_to_input_id(test_df)
        input_feats = df_to_input_feat(test_df)

        with torch.no_grad():
            submit_df['target'] = litmodel.forward(input_ids, input_feats)

        env.predict(submit_df)   # register your predictions


def run(args):
    seed_everything(args.seed)

    # Model
    litmodel = (
        UMPLitModule(args)
        if args.checkpoint is None else
        UMPLitModule.load_from_checkpoint(args.checkpoint, args=args)
    )

    if args.submit:
        submit(litmodel)
        return

    dm = UMPDataModule(args)
    if args.test:
        Trainer(gpus=args.n_gpu).test(litmodel, datamodule=dm)
        return

    name = get_name(args)
    logger = TensorBoardLogger(save_dir='tensorboard_logs', name=name)

    kwargs = {}

    trainer = Trainer(gpus=args.n_gpu,
                      max_epochs=args.epochs,
                      deterministic=True,
                      logger=logger,
                      **kwargs,
                      )
    trainer.fit(litmodel, dm)
    trainer.test(litmodel, datamodule=dm)
    # TODO get checkpoint path
    # trainer.test(ckpt_path='best')  probably pl bug, can't get best ckpt


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

    # Model structure
    parser.add_argument('--n_emb', type=int, default=4000)  # TODO tight
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--szs', type=int, nargs='+',
                        default=[512, 256, 128, 64])

    # Test
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--submit', action='store_true')

    # Checkpoint
    parser.add_argument('--checkpoint', help='path to checkpoint')

    if is_kaggle:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def main():
    kaggle = True
    args = parse_args(kaggle)
    # On kaggle mode, we are using only the args with default value
    # To changle arguments, please hard code it below, e.g.:
    # args.loss = 'mse'
    # args.szs = [256, 256, 256, 256, 256]

    # args.submit = True
    # args.checkpoint = '../input/pretrained/epoch11-step10163.ckpt'

    run(args)


if __name__ == '__main__':
    main()
