""" GUI app main script """

import streamlit as st
from app_pages import inference, exploration


st.set_page_config(page_title="Electricy Demand Predition", layout="wide")


APP_MODES = ["exploratory analysis", "inference"]


def main():

    st.sidebar.title("What to do")
    st.title("Electricity Demand Forecast")
    st.write(
        """
    This work presents   time series analysis and modelling of electricity Demand for a region
      North Carolina
    """
    )
    st.caption(
        "Scource Code: [link](https://github.com/akin-aroge/electricity-demand-forecast)" 
    )
    st.caption(
        "Analysis Notebook: [link](https://github.com/akin-aroge/electricity-demand-forecast/blob/main/notebooks/01-data-exploration.ipynb)"
    )
    app_mode = st.sidebar.selectbox("Choose the app mode", APP_MODES, index=1)
    if app_mode == "inference":
        inference.main()
    elif app_mode == "exploratory analysis":
        exploration.main()


if __name__ == "__main__":
    main()
