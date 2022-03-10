import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from constants import FEATURES


def collate_fn(datas):
    prems = [torch.randperm(data[0].size(0)) for data in datas]
    length = min(data[0].size(0) for data in datas)
    return [
        torch.stack([d[i][perm][:length] for d, perm in zip(datas, prems)])
        for i in range(3)
    ]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *tensor_lists) -> None:
        assert all(len(tensor_lists[0]) == len(
            t) for t in tensor_lists), "Size mismatch between tensor_lists"
        self.tensor_lists = tensor_lists

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensor_lists)

    def __len__(self):
        return len(self.tensor_lists[0])


def df_to_input_id(df):
    return torch.tensor(df['investment_id'].to_numpy(dtype=np.int16),
                        dtype=torch.int)


def df_to_input_feat(df):
    return torch.tensor(df[FEATURES].to_numpy(),
                        dtype=torch.float16)


def load_dataset(args):

    def get_df_group():
        df = pd.read_parquet(args.input)
        groups = df.groupby('time_id')
        return [
            groups.get_group(v)
            for v in df.time_id.unique()
        ]

    df_groupby_time = get_df_group()

    X_id = [df_to_input_id(df) for df in df_groupby_time]
    X_feat = [df_to_input_feat(df) for df in df_groupby_time]

    y = [
        torch.tensor(df['target'].to_numpy(), dtype=torch.float16)
        for df in df_groupby_time
    ]

    dataset = MyDataset(X_id, X_feat, y)

    n_train = int(len(dataset)*0.7)
    n_val = int(len(dataset)*0.1)
    n_test = len(dataset) - n_train - n_val

    tr, val, test = random_split(dataset, [n_train, n_val, n_test])
    return tr, val, test


class UMPDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.tr, self.val, self.test = load_dataset(args)
        self.args = args

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.args.batch_size,
                          num_workers=self.args.workers, shuffle=True,
                          collate_fn=collate_fn, drop_last=True,
                          pin_memory=True)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=1,
                          num_workers=self.args.workers, pin_memory=True)

    def val_dataloader(self):
        return self._val_dataloader(self.val)

    def test_dataloader(self):
        return self._val_dataloader(self.test)
