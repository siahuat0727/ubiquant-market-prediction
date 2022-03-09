import numpy as np
import pandas as pd


def transform_csv2pickle(path, usecols, dtypes):
    train = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes
    )
    train.to_pickle('train.pkl')


def main():
    path = '../input/train.csv'

    basecols = ['row_id', 'time_id', 'investment_id', 'target']
    features = [f'f_{i}' for i in range(300)]

    dtypes = {
        'row_id': 'str',
        'time_id': 'int16',
        'investment_id': 'int16',
        'target': 'float32',
    }
    for col in features:
        dtypes[col] = 'float32'
    transform_csv2pickle(path, basecols+features, dtypes)


main()
