import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_dataset():


    def get_df_group():
        df = pd.read_pickle('train.pkl')
        df = df.pop('investment_id')
        groups = df.group_by('time_id')
        return [
            groups.get_group(v)
            for v in df.time_id.unique()
        ]

    df_groupby_time = get_df_group()
    features = [f'f_{i}' for i in range(300)]

    X = [
        (
            torch.tensor(df['investment_id'].to_numpy(), dtype=torch.uint16),
            torch.tensor(df[features].to_numpy(), dtype=torch.float32),
        )
        for df in df_groupby_time
    ]
    y = [
        torch.tensor(df['target'].to_numpy(), dtype=torch.float32)
        for df in df_groupby_time
    ]
    print(len(X))
    assert len(X) == len(y)

    dataset = TensorDataset(X, y)

    n_train=int(len(dataset)*0.7)
    n_val=int(len(dataset)*0.2)
    n_test=len(dataset) - n_train - n_val

    tr, val, test = random_split(dataset, [n_train, n_val, n_test])
