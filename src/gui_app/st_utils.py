"""GUI app utilities"""

from src.inference import main as inf_main
from src.inference import inference as inf
from src.modelling import training
import pandas as pd
import altair as alt
from io import BytesIO
import streamlit as st
import matplotlib.figure


def get_prediction_data(n_days: int, model_name: str):

    prediction_dates = inf.generate_hr_dates_from_days(n_days=n_days)

    prediction_data = inf_main.main(
        model_name=model_name, n_days=n_days
    )  # TODO: consider changing this to dates not days?
    actual_data = inf.get_actual_load_data(prediction_dates)
    plot_data = pd.DataFrame(
        {
            "datetime": prediction_dates.values,
            "predicted": prediction_data.predicted_load.values,
            "actual": actual_data.load.values,
        }
    )
    plot_data = plot_data.melt(id_vars="datetime", var_name="source", value_name="load")
    return plot_data


def st_img_show(fig: matplotlib.figure.Figure):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)


def get_exploration_data():
    data = training.get_training_data()
    return data


def get_temperature_stations():
    temperature_stations = training.get_temperature_column_names()
    return temperature_stations


MODEL_CHOICES = {
    "Linear Model": "linear_model",
    "Boosted Decision Tree": "xgboost",
}


def get_chart(data):
    # data.date = data.index
    hover = alt.selection_single(
        fields=["datetime"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, height=500, title="Electricity Demand")
        .mark_line()
        .encode(
            x=alt.X("datetime", title="Date"),
            y=alt.Y("load", title="Load in Kwh"),
            color="source",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(datetime)",
            y="load",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("yearmonthdate(datetime)", title="date"),
                alt.Tooltip("load", title="load"),
            ],
        )
        .add_params(hover)
    )

    return (lines + points + tooltips).interactive()
