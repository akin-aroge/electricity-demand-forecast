"""main inference script"""

import pandas as pd
import logging
import argparse

from src.inference import inference as inf
from src.preprocessing import preprocessing as prep
from src import utils


# proj_root = utils.get_proj_root()
# config = utils.get_config('config/config.ini')

# raw_temperature_data_path = utils.get_full_path(config['data_paths']['raw_temp_data'])
# raw_temperature_data = pd.read_csv(raw_temperature_data_path, parse_dates=[0])
# preprocessed_temperature_data = prep.TempDataPreprocessor().fit(raw_temperature_data).transform(raw_temperature_data)
# # inf_year = 2008
# preprocessed_temperature_data = preprocessed_temperature_data[preprocessed_temperature_data.datetime.dt.year >= inf_year]
# preprocessed_temperature_data = preprocessed_temperature_data.iloc[1:, :]  # includes first date not in eval
# future_dates = preprocessed_temperature_data.datetime #+ p.to_timedelta(preprocessed_temperature_data.datetime.dt.hour, unit='h')


def main(model_name, n_days: int):

    prediction_dates = inf.generate_hr_dates_from_days(n_days=n_days)
    temperature_data = inf.get_datetime_temperature(prediction_dates)
    inference_input_data = inf.make_inference_data(prediction_dates, temperature_data)

    model = inf.load_model(model_name=model_name)
    predicted_load = model.predict(inference_input_data)

    prediction_datetime_load = pd.DataFrame(
        {"datetime": prediction_dates, "predicted_load": predicted_load}
    )

    return prediction_datetime_load


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--n_days", type=int, default=1)

    args = parser.parse_args()

    # args = parser.parse_args(['--model_name', 'logistic_regression', '--tune_trials', '1', '--balance_data'])

    main(model_name=args.model_name, n_days=args.n_days)
