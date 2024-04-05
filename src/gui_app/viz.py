"""visualizations for GUI app"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_overall_load(data: pd.DataFrame):

    fig, ax = plt.subplots()

    data.plot(x="datetime", y="load", ax=ax)
    ax.set_ylabel("Load in kWh")
    return fig


def plot_temperature(data: pd.DataFrame):

    fig, ax = plt.subplots()

    data.plot(ax=ax, legend=True)
    ax.set_ylabel("Temperature, $^\circ$F")
    return fig


def plot_demand_by_month(data: pd.DataFrame):

    temp_df = data.assign(month=data.datetime.dt.month)
    fig, ax = plt.subplots()
    sns.boxplot(x="month", y="load", data=temp_df, color="w", ax=ax)
    set_plot_labels(ax, xlabel="Month")
    return fig


def plot_demand_by_weekday(data: pd.DataFrame):

    temp_df = data.copy().assign(day_of_wk=data.datetime.dt.day_name())
    fig, ax = plt.subplots()
    sns.boxplot(x="day_of_wk", y="load", data=temp_df, color="w", ax=ax)
    set_plot_labels(ax, xlabel="Weekday")
    return fig


def plot_demand_by_hour(data: pd.DataFrame):
    temp_df = data.copy().assign(hr=data.datetime.dt.hour)
    fig, ax = plt.subplots()
    sns.boxplot(x="hr", y="load", data=temp_df, color="w", ax=ax)
    set_plot_labels(ax, xlabel="hour")
    return fig


def plot_temperature_load_correlations(data: pd.DataFrame):
    temp_df = (
        data.copy().assign(temp=data.iloc[:, 4]).assign(month=data.datetime.dt.month)
    )
    fig, ax = plt.subplots()
    month_of_year = data.datetime.dt.month
    plt.scatter(
        x=temp_df.temp,
        y=data.load,
        c=month_of_year,
        cmap=plt.cm.get_cmap("jet", 12),
        alpha=0.4,
        s=6,
    )
    cb = plt.colorbar(ticks=range(1, 12 + 1), label="month")
    set_plot_labels(ax, xlabel="Temperature, $^\circ F$", ylabel="Load, kW")
    plt.clim(0.5, 12.5)
    return fig


def plot_temperature_lag_correlations(data: pd.DataFrame):

    temp_df = (
        data.copy()
        .assign(lag1=data.load.shift(1))
        .assign(lag2=data.load.shift(2))
        .assign(lag7=data.load.shift(7))
        .assign(lag24=data.load.shift(24))
    )
    fig, axs = plt.subplots(2, 2)
    sns.scatterplot(x="lag1", y="load", data=temp_df, alpha=0.2, size=0.2, ax=axs[0, 0])
    sns.scatterplot(x="lag2", y="load", data=temp_df, alpha=0.2, size=0.2, ax=axs[0, 1])
    sns.scatterplot(x="lag7", y="load", data=temp_df, alpha=0.2, size=0.2, ax=axs[1, 0])
    sns.scatterplot(
        x="lag24", y="load", data=temp_df, alpha=0.2, size=0.2, ax=axs[1, 1]
    )
    # set_plot_labels(axs, xlabel="temp")
    fig.tight_layout()
    return fig


def set_plot_labels(ax, xlabel="datetime", ylabel="load, kWh"):

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
