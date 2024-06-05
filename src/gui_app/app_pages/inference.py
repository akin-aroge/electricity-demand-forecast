"""gui inference page"""

import streamlit as st
from src.gui_app import st_utils


def main():

    ui_model_name = model_select_ui()
    model_name = st_utils.MODEL_CHOICES[ui_model_name]

    n_days = pred_horizon_ui()

    plot_data = st_utils.get_prediction_data(n_days=n_days, model_name=model_name)
    tseries_plot_ui(plot_data=plot_data, ui_model_name=ui_model_name)


def model_select_ui():
    st.sidebar.markdown("# Model")
    model_name = st.sidebar.selectbox("choose model: ", st_utils.MODEL_CHOICES.keys())
    return model_name


def pred_horizon_ui():
    st.sidebar.markdown("# Input")
    n_days = st.sidebar.slider(
        "Set number of days (pred. horizon):",
        min_value=1,
        max_value=365,
        value=3,
        step=1,
    )
    return n_days


def tseries_plot_ui(plot_data, ui_model_name):
    st.markdown(f"## Predicted Load (using {ui_model_name})")
    if ui_model_name == "Linear Model":
        st.write(
            """
            $$
            y_i = \\beta_0  + \\beta_1 *f_1(temp.) + \\beta_2*f_2(hour) + \\beta_3*f_3(month) + ... 
            $$
    """
        )
    show_actual_vals = st.checkbox("Show actual load values")
    if not show_actual_vals:
        plot_data = plot_data[plot_data["source"] == "predicted"]
    chart = st_utils.get_chart(plot_data)
    st.altair_chart(chart.interactive(), use_container_width=True)
