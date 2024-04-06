""" data preprocssing script """

import pandas as pd
import logging
import argparse

from src import utils
from src.preprocessing import preprocessing as prep


proj_root = utils.get_proj_root()
config = utils.get_config(config_rel_path="config/config.ini")


def main(save_data: bool = True):
    logger = logging.getLogger(__name__)

    raw_load_data_path = utils.get_full_path(config["data_paths"]["raw_load_data"])
    raw_temp_data_path = utils.get_full_path(config["data_paths"]["raw_temp_data"])

    raw_load_data = pd.read_csv(raw_load_data_path, parse_dates=[0])
    raw_temp_data = pd.read_csv(raw_temp_data_path, parse_dates=[0])

    load_data = prep.LoadDataPreprocessor().fit_transform(raw_load_data)
    temp_data = prep.TempDataPreprocessor().fit(raw_temp_data).transform(raw_temp_data)

    load_data.drop(
        labels=["datetime", "hour", "date"], axis=1, inplace=True, errors="ignore"
    )
    load_n_temp_data = pd.concat([load_data, temp_data], axis=1)

    if save_data:
        preprocessed_data_path = proj_root.joinpath(
            config["data_paths"]["preprocessed_data"]
        )
        load_n_temp_data.to_csv(preprocessed_data_path, index=False)
        logger.info(f"preprocessed data saved:{load_n_temp_data.head(3)}")

    return load_n_temp_data


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="preprocessing parser")
    parser.add_argument(
        "--save_data",
        type=str,
        default=True,
    )
    args = parser.parse_args()

    main(save_data=args.save_data)
