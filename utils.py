import math

import torch


def rand_uniform(lo, hi):
    return torch.FloatTensor(1).uniform_(lo, hi).item()


# Modify from https://github.com/wzlxjtu/PositionalEncoding2D
def pos_encoding(length, n_dim, device='cpu', dtype=torch.float):
    """
    :param n_dim: dimension of the model
    :param length: length of positions
    :return: length*n_dim position matrix
    """
    if n_dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         f"odd dim (got n_dim={n_dim:d})")
    pe = torch.zeros(length, n_dim, device=device)
    position = torch.arange(length, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, n_dim, 2, device=device, dtype=dtype) *
                         -(math.log(10000.0) / n_dim)))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
