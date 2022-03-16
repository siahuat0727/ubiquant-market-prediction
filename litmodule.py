import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from torch import nn
from torchmetrics import PearsonCorrCoef

from constants import FEATURES, N_INVESTMENT
from model import Net


def get_loss_fn(loss):
    def mse(preds, y):
        return nn.MSELoss()(preds, y)

    def pcc(preds, y):
        assert preds.dim() == 2, preds.size()
        assert preds.size() == y.size(), (preds.size(), y.size())

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return -cos(preds - preds.mean(dim=1, keepdim=True),
                    y - y.mean(dim=1, keepdim=True)).mean()

    return {
        'mse': mse,
        'pcc': pcc,
    }[loss]


class UMPLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Net(args, n_embed=N_INVESTMENT, n_feature=len(FEATURES))
        self.test_pearson = PearsonCorrCoef()
        self.loss_fn = get_loss_fn(args.loss)

    def forward(self, *args):
        return self.model(*args)

    def predict(self, *args):
        return self.forward(*args)

    def training_step(self, batch, batch_idx):
        x_id, x_feat, y = batch

        preds = self.forward(x_id, x_feat)
        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def _evaluate_step(self, batch, batch_idx, stage):
        x_id, x_feat, y = batch

        preds = self.forward(x_id, x_feat)
        self.test_pearson(preds, y)
        self.log(f'{stage}_pearson', self.test_pearson, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        kwargs = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay,
        }

        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]

        optim_config = {
            'optimizer': optimizer,
            'monitor': 'val_pearson',
        }

        if self.args.lr_scheduler is not None:
            optim_config['lr_scheduler'] = {
                'step_lr': torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.8),
                'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.95, patience=10)
            }[self.args.lr_scheduler]

        return optim_config

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_pearson', mode='max', save_last=True,
                            filename='{epoch}-{val_pearson:.4f}'),
        ]
        if self.args.swa:
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=0.7,
                                                       device='cpu'))
        if self.args.early_stop:
            callbacks.append(EarlyStopping(monitor='val_pearson',
                                           mode='max', patience=35))
        return callbacks


class UMPLitModuleMem(UMPLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem = None

    def predict(self, *args):
        preds, self.mem = self.forward(*args)
        return preds

    def on_train_epoch_start(self):
        self.mem = None

    def on_validation_epoch_start(self):
        self.mem = None

    def training_step(self, batch, batch_idx):
        x_id, x_feat, y = batch

        preds, self.mem = self.forward(x_id, x_feat, self.mem)
        if (batch_idx + 1) % self.args.accumulate_grad_batches == 0:
            self.mem = self.mem.detach()
            # Random reset mem
            if torch.rand(1).le(0.05):
                self.mem = None
        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def _evaluate_step(self, batch, batch_idx, stage):
        x_id, x_feat, y = batch

        # TODO no need detach, reset at epoch start
        preds, self.mem = self.forward(x_id, x_feat, self.mem)
        self.test_pearson(preds, y)
        self.log(f'{stage}_pearson', self.test_pearson, prog_bar=True)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        return loss.backward(*args, **kwargs, retain_graph=True)
