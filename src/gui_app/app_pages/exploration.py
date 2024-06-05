"""Data Exploration Page"""

import streamlit as st
import st_utils
from src.gui_app import viz, stories
from src import utils
from src.modelling import training
import pandas as pd
import numpy as np


max_year = 2007
min_year = 2005

proj_root = utils.get_proj_root()
def main():

    st.write(
        """ 
            # Data Exploration
            This section shows an exploratory walkthrough of the input timeseries dataset 
            to the model. This dataset consists of hourly load and temperature 
            values for a region in North Carolina. The analysis notebook which contains relevant
            code may be found [here](https://github.com/akin-aroge/electricity-demand-forecast/blob/main/notebooks/01-data-exploration.ipynb)
"""
    )

    data = st_utils.get_exploration_data()
    sec_24_hour_profile()
    sec_overall_load(data=data)
    sec_temperature_timeseries(data=data, temperature_stations_filter=True)
    sec_demand_by_time()
    sec_demand_by_month(data=data)
    sec_demand_by_wkday(data=data)
    sec_demand_by_hr(data=data)
    sec_correlations()
    sec_temp_load_corr(data=data)
    sec_temp_lags_corr(data=data)
    

def sub_sec_image_comment(im_name, comment):
    """ display an image and show the comment below it """

    f_path = str(utils.get_proj_root().joinpath(f'reports/{im_name}'))
    st.image(f_path)
    st.write(comment)


def sec_24_hour_profile():
    st.write(stories.ON_24HR_TREND_INTRO)

    sub_sec_image_comment(im_name='daily_trend.png', comment=stories.ON_24HR_TREND_PLOT)
    sub_sec_image_comment(im_name='pca_plot.png', comment=stories.ON_PCA_RESULT)
    sub_sec_image_comment(im_name='pca_coeffs.png', comment=stories.ON_PCA_COEFFS)
    sub_sec_image_comment(im_name='pca_cluster.png', comment=stories.ON_PCA_COEFFS_CLUSTER)
    sub_sec_image_comment(im_name='pca_cluster_profile.png', comment=stories.ON_PCA_CLUSTER_PROFILES)
    sub_sec_image_comment(im_name='pca_coeffs_month.png', comment=stories.ON_PCA_COEFFS_MONTHS)



def sec_overall_load(data: pd.DataFrame):
    # st.write("""""## Demand Trend Over Time")
    st.write(
        """
## Demand Trend Over Time
Now we would proceed to explicitly explore the correlation of the demand with various time components.

The figure shows a plot of hourly electricity demand.
"""
    )
    year_filter = year_slider()
    data = filter_data_year(year_filter=year_filter, data=data, date_column="datetime")

    fig = viz.plot_overall_load(data=data)

    st_utils.st_img_show(fig)

    st.write(
        """ 
The demand plot shows an annual periodicity with peaks including a peak around August 
and a smaller peak around February. This months likely represent the hottest and coldest
times of the year where air conditioning and heating needs lead to increase in electricity
demand. In contrastm the troughs around May are likely due to a moderate demand when the 
temperature is closer to room temperature.
"""
    )


def sec_temperature_stations():
    # stations = st_utils.get_temperature_stations()
    stations_range = np.arange(1, 10 + 1)
    select_stations = st.multiselect(
        label="select temperature stations:",
        options=stations_range,
        default=stations_range,
    )
    return select_stations


def sec_temperature_timeseries(data: pd.DataFrame, temperature_stations_filter=False):

    st.write(
        """
        ## Hourly Temperature over time
        The figure shows a plot of hourly temperatures collected across select temperature stations.
        """
    )

    if temperature_stations_filter:
        temp_station_idx = sec_temperature_stations()
        temp_col_names = st_utils.get_temperature_stations()
        select_names = [temp_col_names[idx - 1] for idx in temp_station_idx]
        data = data[select_names]

    try:
        fig = viz.plot_temperature(data)
        st_utils.st_img_show(fig)

        st.write(
            """
        The temperatures across the different stations appear to be significantly correlated.
        Here again, we notice an annual pattern which confirms the hypothesis of the electricity demand trend.
        The hottest temperatures occur around July/August (summer), while there are dips in January/February (winter). The peaks around
        August lead to increased air conditioning use and the load plot suggest this may be responsible for 
        highest electricity demand.

        This makes the temperature an important predicitve feature.
        """
        )
    except TypeError:
        st.write("select at least one temperature station to view.")


def sec_demand_by_time():
    st.write(
        """
        ## Demand by Time
        In this section, the distribution across different time levels are examined.
        """
    )


def sec_demand_by_month(data: pd.DataFrame):
    st.write(
        """
            ### Demand by month of the year.
            The plot shows the distribution of the demand by the month of the year.
            """
    )

    fig = viz.plot_demand_by_month(data)
    st_utils.st_img_show(fig)
    st.write(
        """
This plot makes clear the distribution of demand for each month showing the median temperature
rises from June to the peak in August followed by a drop in september. The lowest demand
is experienced in April which follows the slight bump in load due to heating deamnd in th winter months
starting from the October dip till February the following year.

It is also noticeable that there is a high variation in the demand when the median is higher, which may
reflect variations in consumer preferences, needs, or ability to meet energy cost.

This makes the month of the year an important predicitve feature.
 
"""
    )


def sec_demand_by_wkday(data: pd.DataFrame):
    st.write(
        """
        ### Demand by day of the week.
        The plot shows the distribution  of the demand for weekday groups.
        """
    )
    fig = viz.plot_demand_by_weekday(data)
    st_utils.st_img_show(fig)
    st.write(
        """
Interestingly, there appears to be no significant differnces in the distribution of loads for 
different days of the week.
"""
    )


def sec_demand_by_hr(data: pd.DataFrame):
    st.write(
        """
### Demand by hour of the day. 
Presumably, elctricity demand is relatively higher at times when people are most likely to be indoors
in the eveninig.
"""
    )
    fig = viz.plot_demand_by_hour(data)
    st_utils.st_img_show(fig)

    st.write(
        """
Expectedly, the distribution shows that the electricity demand peaks in the evening hours around 
17:00. Demand appears to be lowest just before dawn at 5:00. It appears that a descent in demand start
from later in the evening up until early mornings.

THis suggests a significant non-linear variability of load with hour which will be considered in the modelling.
"""
    )


def sec_correlations():
    st.write(
        """
## Correlations
We would now look at a few correlations.
"""
    )


def sec_temp_load_corr(data):
    st.write(
        """
### Load Vs. Temperature
Let's examine the correlation between temperature and the demand
"""
    )

    fig = viz.plot_temperature_load_correlations(data)
    st_utils.st_img_show(fig)
    st.write(
        """
The previous plots have clearly suggested that electricity demand increases in the winter and summer months.
This results in the u-sahped plot seen in the plot, showing the non-linearily that exist between
load and temperature.
"""
    )


def sec_temp_lags_corr(data):
    st.write(
        """
        # Temperature Lag correlations
        From one hour to the next, the temperatrure is load demand is expected to drop only slightly which would 
        result in a correlation between lags. 
        """
    )

    fig = viz.plot_temperature_lag_correlations(data)
    st_utils.st_img_show(fig)

    st.write(
        """
The plot shows a correlation between the demand and the 1 hr lag. This correlation appears to reduce with
increasing number of hours of the lag but becomes high again after 24 hours, due to same hour trends 
in demand.

Depending on the type of model being used, lag features may be engineered
        in the model building process.
"""
    )



def year_slider():

    value = st.slider(
        label="Set year interval:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )
    return value


def filter_data_year(year_filter, data: pd.DataFrame, date_column):

    (select_min_year, select_max_year) = year_filter

    data = data[
        (data[date_column] < str(select_max_year + 1))
        & (data[date_column] > str(select_min_year))
    ]

    return data
