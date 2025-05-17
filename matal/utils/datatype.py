import torch
import pandas as pd
import numpy as np


def to_torch(x):
    if isinstance(x, list):
        return torch.tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float64))
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        return torch.from_numpy(x.values.astype(np.float64))
    else:
        return torch.tensor(x)
        # raise ValueError("The input type is not supported. Please provide a list, numpy array, pandas dataframe, or pandas series.")
