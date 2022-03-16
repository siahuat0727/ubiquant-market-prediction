from itertools import accumulate

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from constants import FEATURES, N_INVESTMENT
from utils import rand_uniform


def collate_fn(datas):
    perms = [torch.randperm(data[0].size(0)) for data in datas]
    min_len = min(data[0].size(0) for data in datas)
    # Random truncate some
    min_len = int(min_len * rand_uniform(0.8, 1.0))

    ids, _, _ = res = [
        torch.stack([d[i][perm][:min_len] for d, perm in zip(datas, perms)])
        for i in range(3)
    ]
    # Random mask some ids to unknown
    mask = torch.rand(ids.size()).le(0.002)
    ids[mask] = N_INVESTMENT
    return res


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *tensor_lists) -> None:
        assert all(len(tensor_lists[0]) == len(
            t) for t in tensor_lists), "Size mismatch between tensor_lists"
        self.tensor_lists = tensor_lists

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensor_lists)

    def __len__(self):
        return len(self.tensor_lists[0])


class MyTimeDataset(torch.utils.data.Dataset):
    def __init__(self, *tensor_lists, times=None) -> None:
        assert all(len(tensor_lists[0]) == len(
            t) for t in tensor_lists), "Size mismatch between tensor_lists"
        assert times is not None and times.size(0) == tensor_lists[0].size(0)

        self.tensor_lists = tensor_lists
        self.unique_times = times.unique()
        self.times = times

    def __getitem__(self, index):
        mask = self.times.eq(self.unique_times[index])
        return tuple(t[mask] for t in self.tensor_lists)

    def __len__(self):
        return self.unique_times.size(0)


def df_to_time(df):
    return torch.LongTensor(df['time_id'].to_numpy(dtype=np.int))


def df_to_input_id(df):
    return torch.LongTensor(df['investment_id'].to_numpy(dtype=np.int))


def df_to_input_feat(df):
    return torch.FloatTensor(df[FEATURES].to_numpy())


def df_to_target(df):
    return torch.FloatTensor(df['target'].to_numpy())


def load_data(path):
    df = pd.read_parquet(path)
    groups = df.groupby('time_id')
    return [
        groups.get_group(v)
        for v in df.time_id.unique()
    ]


def split(df_groupby_time, split_ratios):
    ids = [df_to_input_id(df) for df in df_groupby_time]
    feats = [df_to_input_feat(df) for df in df_groupby_time]
    targets = [df_to_target(df) for df in df_groupby_time]

    dataset = MyDataset(ids, feats, targets)

    lengths = []
    for ratio in split_ratios[:-1]:
        lengths.append(int(len(dataset)*ratio))
    lengths.append(len(dataset) - sum(lengths))

    return random_split(dataset, lengths)


def get_dataset_shuffled(args):
    return split(load_data(args.input), args.split_ratios)


def get_dataset_through_time(args):
    df = pd.read_parquet(args.input)
    ids = df_to_input_id(df)
    feats = df_to_input_feat(df)
    targets = df_to_target(df)
    times = df_to_time(df)
    unique_times = times.unique()

    lengths = []
    for ratio in args.split_ratios[:-1]:
        lengths.append(int(len(unique_times)*ratio))
    lengths.append(len(unique_times) - sum(lengths))

    accum_lens = list(accumulate(lengths))

    def get_dataset(lo, hi):
        ts = unique_times[lo:hi]
        mask = times.ge(ts.min()) & times.le(ts.max())
        return MyTimeDataset(ids[mask], feats[mask], targets[mask],
                             times=times[mask])

    return [
        get_dataset(lo, hi)
        for lo, hi in zip([0]+accum_lens[:-1], accum_lens)
    ]



class UMPDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # datasets = split(load_data(args.input), args.split_ratios)
        datasets = get_dataset_through_time(args)
        if len(datasets) == 3:
            self.tr, self.val, self.test = datasets
        else:
            self.tr, self.val = datasets
            self.test = self.val

    def train_dataloader(self):
        shuffle = not self.args.with_memory
        return DataLoader(self.tr, batch_size=self.args.batch_size,
                          num_workers=self.args.workers, shuffle=shuffle,
                          collate_fn=collate_fn, drop_last=True,
                          pin_memory=True)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=1,
                          num_workers=self.args.workers, pin_memory=True)

    def val_dataloader(self):
        return self._val_dataloader(self.val)

    def test_dataloader(self):
        return self._val_dataloader(self.test)
