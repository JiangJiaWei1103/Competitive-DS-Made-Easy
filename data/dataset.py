"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.
"""
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from metadata import TARGET_COL


class DemoDataset(Dataset):
    """Demo Dataset.

    Args:
        data: processed data
        split: data split

    Attributes:
        _n_samples: number of samples
        _infer: if True, the dataset is constructed for inference
            *Note: Ground truth is not provided.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **dataset_cfg: Any,
    ) -> None:
        self.data = data
        self.dataset_cfg = dataset_cfg

        self.feats = [c for c in self.data.columns if c != TARGET_COL]

        self._set_n_samples()
        self._infer = False

    def _set_n_samples(self) -> None:
        self._n_samples = len(self.data)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if isinstance(self.data, pd.DataFrame):
            data_row = self.data.iloc[idx]
        else:
            data_row = self.data[idx]

        # Construct data sample here...
        data_sample = {"x": torch.tensor(data_row[self.feats].values, dtype=torch.float32)}
        if not self._infer:
            data_sample["y"] = torch.tensor(data_row[TARGET_COL], dtype=torch.float32)

        return data_sample
