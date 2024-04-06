""" List of functions and classes for data preprocessing. """

import logging
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LoadDataPreprocessor(BaseEstimator, TransformerMixin):


    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):

        return self

    def transform(self, X: pd.DataFrame):
        print("transform")
        X.columns = ["date", "hour", "load"]

        X["datetime"] = X.date + pd.to_timedelta(X.hour - 1, unit="h")
        logging.getLogger(self.__class__.__name__).info(f"preprocessed load data")
        return X


class TempDataPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):

        X = (
            X.set_axis(["date", "hour", "station_id", "temperature"], axis=1)
            .assign(datetime=lambda d: d.date + pd.to_timedelta(d.hour - 1, unit="h"))
            .assign(
                dummy_col=np.arange(len(X)) % 2
            )  # to preserve the repeated hours during pivot
            .pivot_table(
                values="temperature",
                index=["date", "hour", "datetime", "dummy_col"],
                columns="station_id",
            )
            .reset_index()
            .drop(labels=["dummy_col", "date", "hour"], axis=1)
            .pipe(
                lambda d: d.rename(
                    columns=dict(
                        [
                            (col, "t" + str(col))
                            for col in d.columns
                            if isinstance(col, int)
                        ]
                    )
                )
            )
        )
        logging.getLogger(self.__class__.__name__).info(
            f"preprocessed temperature data"
        )
        return X
