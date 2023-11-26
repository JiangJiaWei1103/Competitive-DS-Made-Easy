"""
Data processor.
Author: JiaWei Jiang

This file contains the definition of data processor generating datasets
ready for modeling phase.
"""
import logging
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from metadata import TARGET_COL


class DataProcessor(object):
    """Data processor generating datasets ready for modeling phase.

    Args:
        dp_cfg: hyperparameters of data processor

    Attributes:
        _data_cv: training set
            *Note: Fold columns are expected if pre-splitting is done,
                providing mapping from sample index to fold number.
        _data_test: test set
    """

    # https://stackoverflow.com/questions/59173744
    _data_cv: Union[pd.DataFrame, np.ndarray]
    _data_test: Union[pd.DataFrame, np.ndarray]

    def __init__(self, **dp_cfg: Any) -> None:
        # Setup data processor
        self.dp_cfg = dp_cfg
        self._setup()

        # Load raw data
        self._load_data()

    def _setup(self) -> None:
        """Retrieve hyperparameters for data processing."""
        # Before data splitting

        # After data splitting
        self.scaling = self.dp_cfg["scaling"]

    def _load_data(self) -> None:
        """Load raw data."""
        self._data_cv = pd.read_csv(self.dp_cfg["data_cv_path"])
        self._data_test = None

    def run_before_splitting(self) -> None:
        """Clean and process data before data splitting (i.e., on raw
        static DataFrame).
        """
        logging.info("Run data cleaning and processing before data splitting...")

        # Put processing logic below...

        logging.info("Done.")

    def run_after_splitting(
        self,
        data_tr: Union[pd.DataFrame, np.ndarray],
        data_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Any]:
        """Clean and process data after data splitting to avoid data
        leakage issue.

        Note that data processing is prone to data leakage, such as
        fitting the scaler with the whole dataset.

        Args:
            data_tr: training data
            data_val: validation data

        Returns:
            data_tr: processed training data
            data_val: processed validation data
            scaler: scaling object
        """
        logging.info("Run data cleaning and processing after data splitting...")

        # Scale data
        scaler = None
        if self.scaling is not None:
            data_tr, data_val, scaler = self._scale(data_tr, data_val)

        # Put more processing logic below...

        logging.info("Done.")

        return data_tr, data_val, scaler

    def get_data_cv(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return data for CV iteration."""
        return self._data_cv

    def get_data_test(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return unseen test set for final evaluation."""
        return self._data_test

    def _scale(
        self,
        data_tr: Union[pd.DataFrame, np.ndarray],
        data_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Any]:
        """Scale the data.

        Args:
            data_tr: training data
            data_val: validation data

        Returns:
            data_tr: scaled training data
            data_val: scaled validation data
            scaler: scaling object
        """
        logging.info(f"\t>> Scale data using {self.scaling} scaler...")
        feats_to_scale = [c for c in data_tr.columns if c != TARGET_COL]
        if self.scaling == "standard":
            scaler = StandardScaler()

        # Scale cv data
        data_tr.loc[:, feats_to_scale] = scaler.fit_transform(data_tr[feats_to_scale].values)
        data_val.loc[:, feats_to_scale] = scaler.transform(data_val[feats_to_scale].values)

        # Scale test data...

        return data_tr, data_val, scaler
