""" List of objects for data transformation. """

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import logging
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture as GM
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as hol_calendar

from src import utils

logger = logging.getLogger(__name__)


class DerivedColumnTransformer(BaseEstimator, TransformerMixin):
    """
    This custom transformer takes a DataFrame, a column name, and a function
    to create a new derived column.
    """

    def __init__(self, column_name, new_column_name, derive_func, func_kwargs=None):
        """
        Args:
            column_name (str): The name of the existing column to use for derivation.
            derive_func (callable): The function that takes a Series and returns a Series
                containing the derived values.
        """
        self.column_name = column_name
        self.derive_func = derive_func
        self.new_column_name = new_column_name
        self.func_kwargs = func_kwargs

    def fit(self, X, y=None):
        """
        This method does nothing as there is no parameter fitting required.
        """
        return self

    def transform(self, X):
        """
        Args:
            X (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with the new derived column.
        """
        X = X.copy()
        try:
            # Check if the column exists
            # if self.column_name not in X.columns:
            #     raise ValueError(f"Column '{self.column_name}' not found in the DataFrame.")

            # Apply the derive function and add the new column
            if self.func_kwargs is None:
                X[self.new_column_name] = self.derive_func(X[self.column_name])
            else:
                X[self.new_column_name] = self.derive_func(
                    X[self.column_name], **self.func_kwargs
                )
            logging.getLogger(self.__class__.__name__).info(
                f"created {self.new_column_name} column"
            )
            return X

        except Exception as e:
            # Log the error and potentially re-raise for pipeline handling
            logger.error(f"Error during transformation: {e}")
            raise  # Re-raise to be caught by the pipeline


class MultiColumnTransformer(BaseEstimator, TransformerMixin):
    """
    This custom transformer takes a DataFrame, multiple columns , and a function
    to create a new derived column.
    """

    def __init__(self, column_names, new_column_name, derive_func, func_kwargs=None):
        """
        Args:
            column_name (str): The name of the existing column to use for derivation.
            derive_func (callable): The function that takes a Series and returns a Series
                containing the derived values.
        """
        self.column_names = column_names
        self.derive_func = derive_func
        self.new_column_name = new_column_name
        self.func_kwargs = func_kwargs

    def fit(self, X, y=None):
        """
        This method does nothing as there is no parameter fitting required.
        """
        return self

    def transform(self, X):
        """
        Args:
            X (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with the new derived column.
        """
        X = X.copy()
        try:
            # Check if the column exists
            # if self.column_name not in X.columns:
            #     raise ValueError(f"Column '{self.column_name}' not found in the DataFrame.")

            # Apply the derive function and add the new column
            if self.func_kwargs is None:
                X[self.new_column_name] = self.derive_func(X[self.column_names])
            else:
                X[self.new_column_name] = self.derive_func(
                    X[self.column_names], **self.func_kwargs
                )
            logging.getLogger(self.__class__.__name__).info(
                f"created {self.new_column_name} column"
            )
            return X

        except Exception as e:
            # Log the error and potentially re-raise for pipeline handling
            logger.error(f"Error during transformation: {e}")
            raise  # Re-raise to be caught by the pipeline


class ColumsOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    This custom transformer takes a DataFrame, a column name, and a function
    to create a new derived column.
    """

    def __init__(self, categorical_column_names: list):
        """
        Args:
        """
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.categorical_column_names = categorical_column_names

    def fit(self, X, y=None):
        """ """
        categorical_column_names = self.categorical_column_names
        if categorical_column_names:  # check if list is not empty
            data_subset = X[categorical_column_names]
            self.encoder.fit(data_subset)
        return self

    def transform(self, X: pd.DataFrame):
        """
        Args:
            X (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with the new columns.
        """
        if self.categorical_column_names:  # check if list is not empty
            data_subset = X[self.categorical_column_names].copy()
            cols_to_drop = data_subset.columns
            transformed_data = self.encoder.transform(data_subset)

            X.drop(labels=cols_to_drop, axis=1)
            encoded_feature_names = self.encoder.get_feature_names_out()
            X[encoded_feature_names] = transformed_data.toarray()

        # else:
        #     data = X.copy()
        logging.getLogger(self.__class__.__name__).info(
            f"categorical columns transformed: \
                                                        {self.categorical_column_names}"
        )
        # logging.getLogger(self.__class__.__name__).info(f'cat.cols. encoded: \n {X.head(2)}')

        return X


class ColumnsRemover(BaseEstimator, TransformerMixin):
    def __init__(self, column_names) -> None:
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        initial_cols = list(X.columns)
        X = X.drop(self.column_names, axis=1, errors="ignore")
        final_column_names = list(X.columns)
        dropped_columns = list(set(initial_cols) - set(final_column_names))
        logging.getLogger(self.__class__.__name__).info(
            f"dropped colums: {dropped_columns}"
        )
        return X


class OptimalTemperatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        temperature_column_names_path,
        optimal_temperature_column_names_path,
    ) -> None:

        self.optimal_temperature_column_names_path = (
            optimal_temperature_column_names_path
        )
        self.temperature_column_names_path = temperature_column_names_path
        self.optimal_temperature_column_names = None
        self.temperature_column_names = None
        # self.label_col_name = label_col_name

    def fit(self, X: pd.DataFrame, y=None):
        if self.optimal_temperature_column_names_path is not None:
            try:
                optimal_temperature_column_names = utils.load_value(
                    self.optimal_temperature_column_names_path
                )
                temperature_column_names = utils.load_value(
                    self.temperature_column_names_path
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "File for optimal columns not available \
                                        optimal columns should be determined first."
                )
            self.optimal_temperature_column_names = optimal_temperature_column_names
            self.temperature_column_names = temperature_column_names
        return self

    def transform(self, X: pd.DataFrame):

        columns_to_drop = list(
            set(self.temperature_column_names)
            - set(self.optimal_temperature_column_names)
        )
        X = X.drop(labels=columns_to_drop, axis=1)

        logging.getLogger(self.__class__.__name__).info(
            f"selected temperature columns: {self.optimal_temperature_column_names} "
        )
        # print(X.columns)
        return X


class HourlyProfileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column_name, feature_path) -> None:

        self.datetime_column_name = datetime_column_name
        self.feature_path = feature_path
        self.profile_feature_dict = None

    def fit(self, X: pd.DataFrame, y=None):

        self.profile_feature_dict = utils.load_value(self.feature_path)

        # load_data = pd.DataFrame({'datetime':X[self.datetime_column_name].values,
        #                           'load':y.values})
        # # load_data  = X[['datetime','load']].copy()
        # # print(X.head())
        # # print(load_data)
        # load_data['hour'] = load_data['datetime'].dt.hour.values
        # load_data['date'] = load_data['datetime'].dt.date.values
        # load_pivoted = load_data.pivot_table(values='load', index='date', columns='hour')
        # # print(load_pivoted)
        # print(load_pivoted.isnull().sum())
        # X_ = load_pivoted.dropna(axis=0).values  #  because some hours are null for hour 2

        # n_components=2
        # print(X_.shape)
        # X_pca = PCA(n_components=n_components).fit_transform(X_)

        # gmm = GM(n_components=n_components, covariance_type='full', random_state=0)
        # gmm.fit(X_pca)
        # cluster_label = gmm.predict(X_pca)
        # # print(X.head())
        # load_pivoted['cluster'] = cluster_label
        # # print(load_data_pivoted.head())
        # temp_df = load_data.join(load_pivoted['cluster'], on='date')
        # temp_df = temp_df.groupby(['cluster', temp_df.datetime.dt.time]).mean(numeric_only=True)

        # profile_feature_dict = {
        # 'profile_1':dict(zip(temp_df.loc[0]['hour'].values.astype(int), temp_df.loc[0]['load'].values)),
        # 'profile_2':dict(zip(temp_df.loc[1]['hour'].values.astype(int), temp_df.loc[1]['load'].values))
        # }

        # self.profile_feature_dict = profile_feature_dict

        return self

    def transform(self, X: pd.DataFrame):

        for k, v in self.profile_feature_dict.items():
            X[k] = X[self.datetime_column_name].dt.hour.map(v).values
            # X[k] = X[self.hour_column_name].map(v)

        logging.getLogger(self.__class__.__name__).info(
            f"created daily temperature profiles"
        )
        return X


# class LagColumnTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, column_name, n_hours) -> None:
#         self.column_name = column_name

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X: pd.DataFrame):
#         initial_cols = list(X.columns)
#         X = X.drop(self.column_names, axis=1,  errors='ignore')
#         final_column_names = list(X.columns)
#         dropped_columns = list(set(initial_cols) - set(final_column_names))
#         logging.getLogger(self.__class__.__name__).info(f"dropped colums: {dropped_columns}")
#         return X


## pure functions only
def is_weekend(dates: pd.Series):

    is_weekend = np.uint8(dates.dt.day_of_week > 4)
    return is_weekend


def is_holiday(dates: pd.Series):
    cal = hol_calendar()
    holidays = cal.holidays(start=dates.min(), end=dates.max())
    is_holiday = np.uint8(dates.isin(holidays))
    return is_holiday


def get_month(dates: pd.Series):
    months = dates.dt.month
    return months


def get_hour(dates: pd.Series):
    hour = dates.dt.hour
    return hour


def daily_trend(dates: pd.Series):
    pass


def time_sin_transform(dates: pd.Series, period):

    date_sin = np.sin(dates / period * 2 * np.pi)
    return date_sin


def time_cos_transform(dates: pd.Series, period):

    date_cos = np.cos(dates / period * 2 * np.pi)
    return date_cos


def exp_value(data: pd.Series, exp):

    return data**exp


def multiply_columns(df: pd.DataFrame):

    prod = df.iloc[:, 0] * df.iloc[:, 1]

    return prod


# def trend(dates:pd.Series):
#     return np.arange(0, len(dates))


def trend(dates: pd.Series):

    base_date = pd.Timestamp("1990-11-25")
    days_since = (dates - base_date).dt.days

    return days_since


def season_encode(month_int: pd.Series):

    y = np.power(np.abs(6.0 - month_int), 2)
    return y


def map_time_load_profile(dates):

    pass
