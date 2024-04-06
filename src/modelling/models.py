"""Scripts for constituting models"""

import sklearn
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import numpy as np
from abc import ABC, abstractmethod
import logging


class WrappedModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train, transform_pipeline: Pipeline = None):
        """
        Train the model on the given training data.

        Parameters:
        - X_train: Input features for training.
        - y_train: Target labels for training.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Input features for prediction.

        Returns:
        - Predicted labels.
        """
        pass

    @abstractmethod
    def objective_function(self, trial, X, y, transform_pipeline: Pipeline = None):
        """
        Objective function for optimization tasks.

        Parameters:
        - X: Input features for objective evaluation.
        - y: Target labels for objective evaluation.

        Returns:
        - Objective value (e.g., accuracy, loss).
        """
        pass

    @abstractmethod
    def init_model(self, params):
        pass


ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=0,
    max_train_size=None,
    test_size=None,
)


class WrappedRidgeRegression(WrappedModel):
    def __init__(self, alpha, solver=None) -> None:
        self.alpha = float(alpha)
        self.model = Ridge
        self.is_tuned = False
        self.tuned_params = None

    def objective_function(self, trial, X, y, transform_pipeline: Pipeline):
        alpha = trial.suggest_float("alpha", 0.1, 5, log=True)

        model = self.model(alpha=alpha)

        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=model
        )
        scores = cross_val_score(
            model_pipeline,
            X,
            y,
            cv=ts_cv,
            scoring="neg_mean_absolute_percentage_error",
            error_score="raise",
        )
        mean_score = np.mean(scores)

        print(f"Trial {trial.number}, alpha: {alpha},  mape: {-mean_score}")
        return mean_score

    def init_model(self):
        if self.tuned_params is not None:
            model = self.model(**self.tuned_params)
        else:
            model = self.model(alpha=self.alpha, verbose=1)

        self.model = model

    def train(self, X, y, transform_pipeline: Pipeline = None):
        if not self.is_tuned:
            self.init_model()

        logging.getLogger(self.__class__.__name__).info(
            f"training with {type(self.model)}"
        )
        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=self.model
        )
        model_pipeline.fit(X, y)

        return model_pipeline

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class WrappedXgboost(WrappedModel):

    def __init__(self, n_jobs) -> None:
        self.n_jobs = int(n_jobs)
        self.model = XGBRegressor
        self.is_tuned = False
        self.tuned_params = None

    def objective_function(self, trial, X, y, transform_pipeline: Pipeline):
        n_estimators = trial.suggest_int("n_estimators", 2, 150)
        max_depth = trial.suggest_int("max_depth", 1, 50)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.9, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        gamma = trial.suggest_float("gamma", 0, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0, 1)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 1)

        model = self.model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=self.n_jobs,
        )
        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=model
        )
        # Using cross_val_score to get the average ROC-AUC score for each fold
        scores = cross_val_score(
            model_pipeline,
            X,
            y,
            cv=ts_cv,
            scoring="neg_mean_absolute_percentage_error",
            error_score="raise",
        )
        mean_score = np.mean(scores)
        # Printing intermediate results
        print(
            f"Trial {trial.number}, n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate},"
            f"min_child_weight: {min_child_weight}, subsample: {subsample} "
            f"gamma: {gamma}, reg_alpha: {reg_alpha}, reg_lambda: {reg_lambda}, mape: {-mean_score}"
        )
        return mean_score

    def init_model(self):
        if self.tuned_params is not None:
            model = self.model(**self.tuned_params, n_jobs=self.n_jobs)
        else:
            model = self.model(n_jobs=self.n_jobs)

        self.model = model

    def train(self, X, y, transform_pipeline: Pipeline = None):
        if not self.is_tuned:
            self.init_model()

        logging.getLogger(self.__class__.__name__).info(
            f"training with {type(self.model)}"
        )

        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=self.model
        )
        model_pipeline.fit(X, y)
        return model_pipeline

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


def _make_model_pipeline(transform_pipeline, model):
    model_pipeline = sklearn.clone(transform_pipeline)
    model_pipeline.steps.append(["model", model])
    return model_pipeline
