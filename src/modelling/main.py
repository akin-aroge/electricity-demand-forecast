"""main script for modelling"""

import logging
import argparse
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src import utils
from src.modelling import transforms
from src.modelling import training


warnings.filterwarnings("ignore")

proj_root = utils.get_proj_root()

config = utils.get_config("config/config.ini")


def make_pipeline():

    categorical_column_names = ["hour", "month"]
    columns_to_drop = ["date", "hour", "datetime", "mean_temp"]

    temperature_column_names_path = proj_root.joinpath(
        config["modelling_paths"]["temperature_columns"]
    )
    optimal_temperature_column_names_path = proj_root.joinpath(
        config["modelling_paths"]["optimal_temperature_columns"]
    )
    optimal_temperature_column_names = utils.load_value(
        optimal_temperature_column_names_path
    )
    daily_load_profile_path = proj_root.joinpath(
        config["modelling_paths"]["daily_load_profile_feature"]
    )

    pipeline = Pipeline(
        steps=[
            (
                "select_optimal_temperature_columns",
                transforms.OptimalTemperatureSelector(
                    temperature_column_names_path=temperature_column_names_path,
                    optimal_temperature_column_names_path=optimal_temperature_column_names_path,
                ),
            ),
            (
                "create_weekend_col",
                transforms.DerivedColumnTransformer(
                    column_name="datetime",
                    new_column_name="is_weekend",
                    derive_func=transforms.is_weekend,
                ),
            ),
            (
                "create_month_col",
                transforms.DerivedColumnTransformer(
                    column_name="datetime",
                    new_column_name="month",
                    derive_func=transforms.get_month,
                ),
            ),
            (
                "create_holiday_col",
                transforms.DerivedColumnTransformer(
                    column_name="datetime",
                    new_column_name="is_holiday",
                    derive_func=transforms.is_holiday,
                ),
            ),
            (
                "create_hour_col",
                transforms.DerivedColumnTransformer(
                    column_name="datetime",
                    new_column_name="hour",
                    derive_func=transforms.get_hour,
                ),
            ),
            (
                "create_mean_temperature",
                transforms.DerivedColumnTransformer(
                    column_name=optimal_temperature_column_names,
                    new_column_name="mean_temp",
                    derive_func=np.median,
                    func_kwargs={"axis": 1},
                ),
            ),
            (
                "create_temperature_squared",
                transforms.DerivedColumnTransformer(
                    column_name="mean_temp",
                    new_column_name="temp_sq",
                    derive_func=np.square,
                ),
            ),
            (
                "create_temperature_cube",
                transforms.DerivedColumnTransformer(
                    column_name="mean_temp",
                    new_column_name="temp_cube",
                    derive_func=transforms.exp_value,
                    func_kwargs={"exp": 3},
                ),
            ),

            # ("create_interaction_temp_hour_profile_1",transforms.MultiColumnTransformer(column_names=['mean_temp', 'profile_1'],
            #                                             new_column_name='temp_hour_p1',
            #                                             derive_func=transforms.multiply_columns)
            #                                             ),
            (
                "create_interaction_temp_hour_profile_2",
                transforms.MultiColumnTransformer(
                    column_names=["mean_temp", "hour"],
                    new_column_name="temp_hour_p2",
                    derive_func=transforms.multiply_columns,
                ),
            ),
            (
                "create_interaction_temp_month",
                transforms.MultiColumnTransformer(
                    column_names=["mean_temp", "month"],
                    new_column_name="temp_month",
                    derive_func=transforms.multiply_columns,
                ),
            ),

            (
                "create_interaction_month_hour",
                transforms.MultiColumnTransformer(
                    column_names=["month", "hour"],
                    new_column_name="month_hour",
                    derive_func=transforms.multiply_columns,
                ),
            ),

            (
                "create_non_linear_hour_features",
                transforms.HourlyProfileTransformer(
                    datetime_column_name="datetime",
                    feature_path=daily_load_profile_path,
                ),
            ),
            (
                "create_interaction_month_hour_profile_1",
                transforms.MultiColumnTransformer(
                    column_names=["month", "profile_1"],
                    new_column_name="month_hour_p1",
                    derive_func=transforms.multiply_columns,
                ),
            ),
            (
                "create_trend",
                transforms.DerivedColumnTransformer(
                    column_name="datetime",
                    new_column_name="trend",
                    derive_func=transforms.trend,
                ),
            ),
            (
                "one_hot_categorical_column",
                transforms.ColumsOneHotEncoder(
                    categorical_column_names=categorical_column_names
                ),
            ),
            ("drop_columns", transforms.ColumnsRemover(column_names=columns_to_drop)),
        ]
    )
    return pipeline


def main(model_name: str, n_tuning_trials=1):
    logger = logging.getLogger(__name__)

    model_output_dir = proj_root.joinpath(config["modelling_paths"]["model_output"])
    preprocessed_data_path = proj_root.joinpath(
        config["data_paths"]["preprocessed_data"]
    )
    test_size = float(config["modelling_paths"]["test_size"])

    label_col_name = "load"
    model_params = config._sections[model_name]
    model_class = training.get_model_class(model_name=model_name)
    model = model_class(**model_params)

    preprocessed_data = training.get_training_data(file_path=preprocessed_data_path)

    training_data, testing_data = train_test_split(
        preprocessed_data, test_size=test_size, shuffle=False
    )

    pipeline = make_pipeline()

    model_output_path = model_output_dir.joinpath(model_name + ".pkl")
    trainer = training.ModelTrainer(
        model_class=model,
        transform_pipeline=pipeline,
        training_data=training_data,
        testing_data=testing_data,
        label_col_name=label_col_name,
        model_output_path=model_output_path,
    )

    logger.info("==============tuning started=============")
    trainer.tune_model(n_trials=n_tuning_trials)
    logger.info("==============training started====================")
    model = trainer.train_model(save_model=True)
    logger.info("==========training completed===============")
    logger.info("==========evaluation===============")
    score = trainer.evaluate_model()
    print(f"test score:{score}")
    logger.info(f"test score:{score}")
    score = trainer.inf_model()
    print(f"test score:{score}")
    logger.info(f"inf score:{score}")
    logger.info("==========evaluation completed===============")

    return model, score


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.ERROR, format=log_fmt)

    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--n_tuning_trials", type=int, default=1)
    args = parser.parse_args()

    main(model_name=args.model_name, n_tuning_trials=args.n_tuning_trials)
