"""utilities for training model with training data"""

import pandas as pd
import logging
import pathlib
import optuna
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from src.modelling import models


from src import utils

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        model_class: models.WrappedModel,
        transform_pipeline: Pipeline,
        training_data,
        testing_data,
        label_col_name,
        model_output_path,
    ) -> None:
        self.best_params = None
        self.model_output_path = model_output_path
        self.model_class = model_class
        self.tune_best_score = None
        self.training_data = training_data
        self.testing_data = testing_data
        self.label_col_name = label_col_name
        self.transform_pipeline = transform_pipeline
        self.trained_model = None

    def tune_model(self, n_trials, maximize=True):
        if maximize:
            direction = "maximize"
        else:
            direction = "minimize"
        self.direction = direction

        X_train, y_train = split_Xy(
            self.training_data, label_col_name=self.label_col_name
        )

        study = optuna.create_study(direction=direction)
        study.optimize(
            lambda trial: self.model_class.objective_function(
                trial=trial,
                X=X_train,
                y=y_train,
                transform_pipeline=self.transform_pipeline,
            ),
            n_trials=n_trials,
        )

        self.best_params = study.best_params
        self.tune_best_score = study.best_value
        self.model_class.tuned_params = study.best_params
        # model = self.model_class.init_model(params=self.best_params)
        self.model_class.init_model()
        # self.model_class.model = model
        self.model_class.is_tuned = True
        print(f"best score is: {self.tune_best_score}")

    def train_model(self, save_model=True) -> Pipeline:
        X_train, y_train = split_Xy(
            self.training_data, label_col_name=self.label_col_name
        )

        model_is_tuned = self.model_class.is_tuned
        if not model_is_tuned:
            logging.getLogger(self.__class__.__name__).info("model is not tuned")

        trained_model = self.model_class.train(
            X=X_train, y=y_train, transform_pipeline=self.transform_pipeline
        )

        if save_model:
            utils.save_value(trained_model, fname=self.model_output_path)

        self.trained_model = trained_model

        return trained_model

    def evaluate_model(self):
        X_test, y_test = split_Xy(self.testing_data, label_col_name=self.label_col_name)

        y_pred = self.trained_model.predict(X_test)
        score = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)

        return score

    def inf_model(self):
        X_test, y_test = split_Xy(self.testing_data, label_col_name=self.label_col_name)

        model = utils.load_value(self.model_output_path)
        logging.getLogger(self.__class__.__name__).info(
            f"model loaded: {model.steps[-1][-1]}"
        )

        y_pred = model.predict(X_test)
        score = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)

        return score


def get_features(path: pathlib.Path):
    features_list = utils.load_value(path)

    return features_list


def get_training_data(file_path: pathlib.Path = None):
    """Reaturns the combined load and temperature data, for the training dates"""
    if file_path is None:
        proj_root = utils.get_proj_root()
        config = utils.get_config("config/config.ini")
        file_path = proj_root.joinpath(config["data_paths"]["preprocessed_data"])

    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"].values)
    # print(df.head(2))
    df = df[df.datetime.dt.year < 2008]

    return df


# def train_test_split(df: pd.DataFrame, test_size_frac):


#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size_frac, shuffle=False
#     )

# logger.info(
#     f"data split into: training ({training_data.shape}) and test ({testing_data.shape}) sets "
# )

# return training_data, testing_data


def split_Xy(df: pd.DataFrame, label_col_name: str):
    X = df.drop(label_col_name, axis=1)
    y = df[label_col_name]

    return X, y


def get_model_class(model_name: str):
    if model_name == "linear_model":
        model = models.WrappedRidgeRegression
    elif model_name == "random_forest":
        model = models.RandomForestWrapper
    elif model_name == "xgboost":
        model = models.WrappedXgboost

    return model


def get_categorical_cols(data: pd.DataFrame, raw_data_cat_col_names=None):

    if raw_data_cat_col_names is None:

        raw_data_cat_col_names = ["year", "industry", "symbol"]

    data_col_names = data.columns.values
    cat_cols = set(data_col_names) & set(raw_data_cat_col_names)

    return list(cat_cols)


def get_temperature_column_names():
    config = utils.get_config("config/config.ini")
    proj_root = utils.get_proj_root()
    temperature_column_names_path = proj_root.joinpath(
        config["modelling_paths"]["temperature_columns"]
    )
    temperature_column_names = utils.load_value(temperature_column_names_path)
    return temperature_column_names


config = utils.get_config("config/config.ini")


def save_data(data: pd.DataFrame, path: pathlib.Path):
    data.to_csv(path_or_buf=path, index=False)
