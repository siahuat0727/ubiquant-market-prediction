import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *tensor_lists) -> None:
        assert all(len(tensor_lists[0]) == len(t) for t in tensor_lists), "Size mismatch between tensor_lists"
        self.tensor_lists = tensor_lists

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensor_lists)

    def __len__(self):
        return len(self.tensor_lists[0])


def load_dataset():

    def get_df_group():
        df = pd.read_pickle('train.pkl')
        groups = df.groupby('time_id')
        return [
            groups.get_group(v)
            for v in df.time_id.unique()
        ]

    df_groupby_time = get_df_group()
    features = [f'f_{i}' for i in range(300)]

    X_id = [
        torch.tensor(df['investment_id'].to_numpy(), dtype=torch.int)
        for df in df_groupby_time
    ]

    X_feat = [
        torch.tensor(df[features].to_numpy(), dtype=torch.float32)
        for df in df_groupby_time
    ]

    y = [
        torch.tensor(df['target'].to_numpy(), dtype=torch.float32)
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
        self.tr, self.val, self.test = load_dataset()
        self.args = args

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.args.batch_size,
                          num_workers=self.args.workers, shuffle=True,
                          drop_last=True, pin_memory=True)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.args.batch_size,
                          num_workers=self.args.workers, pin_memory=True)

    def val_dataloader(self):
        return self._val_dataloader(self.val)

    def test_dataloader(self):
        return self._val_dataloader(self.test)
