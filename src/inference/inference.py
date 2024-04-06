""" components for inference """

from src import utils
import logging
import pandas as pd

from src.preprocessing import preprocessing as prep

proj_root = utils.get_proj_root()
config = utils.get_config("config/config.ini")

logger = logging.getLogger(__name__)


def load_model(model_name: str):

    project_root = utils.get_proj_root()

    model_full_path = project_root.joinpath(f"models/{model_name}.pkl")
    try:
        loaded_model = utils.load_value(model_full_path)
        return loaded_model
    except FileNotFoundError as f:
        logger.error(f"Error during transformation: {f}")


def generate_hr_dates_from_days(n_days, start_date="2008-01-01"):
    """generate hourly datetime for `n_days` starting from `start_date`"""

    end_date = pd.to_datetime(start_date) + pd.DateOffset(hours=n_days * 24)
    dates = pd.date_range(start=start_date, end=end_date, freq="h", inclusive="left")
    return dates


def get_datetime_temperature(datetimes):
    raw_temperature_data_path = utils.get_full_path(
        config["data_paths"]["raw_temp_data"]
    )
    raw_temperature_data = pd.read_csv(raw_temperature_data_path, parse_dates=[0])
    preprocessed_temperature_data = (
        prep.TempDataPreprocessor()
        .fit(raw_temperature_data)
        .transform(raw_temperature_data)
    )
    select_temperatures = preprocessed_temperature_data[
        preprocessed_temperature_data["datetime"].isin(datetimes)
    ].iloc[:, 1:]
    return select_temperatures


def make_inference_data(dates, temperature):
    temperature["datetime"] = dates

    return temperature


def get_actual_load_data(dates):

    eval_data_path = config["data_paths"]["evaluation_data"]
    actual_load = pd.read_csv(proj_root.joinpath(eval_data_path), parse_dates=[0])
    preprocessed_load = (
        prep.LoadDataPreprocessor().fit(actual_load).transform(actual_load)
    )
    select_load = preprocessed_load[
        preprocessed_load["datetime"].isin(dates)
    ] 
    return select_load
