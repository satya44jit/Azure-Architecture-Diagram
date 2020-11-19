"""
This script contains all the functions required for data processing and for other functionalities
"""
# built in
import sys
import logging
import os
import pandas as pd
from datetime import datetime, timedelta


# third party
import pandas as pd

LOGGING_LEVEL = "INFO"
PIPELINE_VERBOSE = LOGGING_LEVEL in ["INFO", "DEBUG"]

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def get_model_data(
    level_1, level_2, level_1_val, level_2_val, time_variable, dependent_variable, data
):
    """This function is used to return required dataset that is to be
    used in model by filtering out at required level

    Args:
        level_1 (int): Name of level variable
        level_2 (int): Name of level variable
        level_1_val (int): value of level_1 identifier
        level_2_val (int): value of level_2 identifier
        time_variable (str): Name of the variable that has time values
        dependent_variable (str): Name of the dependent variable
        data (Dataframe): Traning Dataset

    Returns:
        df(Dataframe): Filtered out time series for model
    """
    temp = data[(data[level_1] == level_1_val) & (data[level_2] == level_2_val)].copy()
    temp.sort_values(by=[time_variable], ascending=True, inplace=True)
    df = temp[[time_variable, dependent_variable]].copy()
    df.set_index(time_variable, inplace=True)
    return df


def generate_next_timestamps(x, periods):
    # TODO: The below line is required if the input data is treated as string

    cols = []
    while periods:
        x = x + 1
        cols.extend([x])
        periods -= 1
    return cols


def output_forecast(
    data,
    level_1,
    level_2,
    level_1_val,
    level_2_val,
    predictions,
    n_periods,
    config_data,
):
    # df = data[config_data["cols_for_table"]].copy()
    temp = data[(data[level_1] == level_1_val) & (data[level_2] == level_2_val)].copy()
    prev_x = temp["week"].max()
    future_weeks = generate_next_timestamps(prev_x, n_periods)

    df_list = []
    for i, j in zip(future_weeks, predictions):
        input_list = [i, level_1_val, level_2_val, j]
        df_list.append(input_list)

    df_temp = pd.DataFrame(df_list, columns=config_data["cols_for_table"])
    temp = temp.append(df_temp, ignore_index=True)

    temp.sort_values(by=config_data["time"], ascending=True, inplace=True)
    temp.to_csv(
        os.path.join(
            config_data["output_path"],
            f"forecast for {level_1_val}_{level_2_val}.csv",
        ),
        index=False,
    )


def output_forecast_list(
    data,
    level_1,
    level_2,
    level_1_val,
    level_2_val,
    predictions,
    n_periods,
    config_data,
    forecast,
):
    temp = data[(data[level_1] == level_1_val) & (data[level_2] == level_2_val)].copy()
    prev_x = temp["week"].max()
    future_weeks = generate_next_timestamps(prev_x, n_periods)

    for i, j in zip(future_weeks, predictions):
        input_list = [i, level_1_val, level_2_val, j]
        forecast.append(input_list)
