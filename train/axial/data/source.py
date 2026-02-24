from pathlib import Path
from torch.utils.data import Dataset

import numpy as np


class NumpyDataSource[T](Dataset):
    """
    A simple data source that loads data from a .npy file.
    Expects the top-level structure to be an array of items.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data = np.load(path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> T:
        return self.data[idx]