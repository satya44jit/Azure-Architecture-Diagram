"""
This code is to run an Auto ARIMA Model at each fulfillment center level.
"""

# Importing all the required built in modules
import os
import logging
import joblib
import sys
import timeit
from timeit import default_timer
from datetime import datetime, date, timedelta

# Importing all the necessary site packages
import pandas as pd
import matplotlib.pyplot as plt

# import google.cloud.bigquery as bigquery
import json
from arima import AutoARIMA
import pmdarima as pm
from util import load_data
from functions import get_model_data
from functions import output_forecast
from functions import output_forecast_list

LOGGING_LEVEL = "INFO"
PIPELINE_VERBOSE = LOGGING_LEVEL in ["INFO", "DEBUG"]

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

# Creating a date variable to save the output with today's date
run_date = date.today().strftime("%Y_%m_%d")


start = default_timer()
logger.info(f"Job Started. Starting at: {start}")

# DATA PREPERATION
# Importing the config json file

json_path = "/root/Hackathon 2020/dev/"
with open(os.path.join(json_path, "config_arima.json")) as f:
    config_data = json.load(f)

'''
# FOR BIG QUERY INPUT
PROJECT_ID = config_data["project_id"]
DATASET_ID = config_data["dataset_id"]
BQ_CLIENT = bigquery.Client(project=PROJECT_ID)


# To stream and read the data from Bigquery
def stream_bq_to_pd(query_str):
    """Execute BQ query and return Pandas DF

    Args:
        query_str (str): full BQ query to execute

    Returns:
        pandas DF:
    """
    df = BQ_CLIENT.query(query_str).to_dataframe()
    return df

# Importing Input data
df = stream_bq_to_pd(
    f"select * from `{PROJECT_ID}.{DATASET_ID}.preprocess_store_week_dummy`"
)
'''
# FOR LOCAL DATA PART
data_path = config_data["data_path"]
pre_train = load_data(data_path, "train.csv")
date_df = load_data(data_path, "week_date_mapping.csv")
date_df["date"] = pd.to_datetime(date_df["date"])
train = pre_train[config_data["cols_for_table"]].copy()
train.loc[:, "center_meal"] = (
    train.loc[:, "center_id"].astype(str) + "_" + train.loc[:, "meal_id"].astype(str)
)
abt = pd.DataFrame()
for i in train["center_meal"].unique().tolist():
    if train.loc[train["center_meal"] == i].shape[0] >= 56:
        temp = train.loc[train["center_meal"] == i]
        abt = abt.append(temp)

train_level = abt.groupby(by=config_data["list_attr"], as_index=False).agg(
    {config_data["dep_var"]: "sum"}
)

series_data = train_level[config_data["list_attr"]].copy()

# Creating Empty Dictionary for forecasts and dataframe for order
dict_forecast = {}
forecast_list = []
model_order_df = pd.DataFrame(columns=["series_id", "level_id", "order"])
# MODEL RUNNING
for row in series_data[0:10].iterrows():
    start_run = default_timer()
    logger.info(f"running the series identifier: {row[0]}")
    center_id = series_data[config_data["list_attr"][0]][row[0]]
    meal_id = series_data[config_data["list_attr"][1]][row[0]]
    logger.info(f"running the model for: {center_id}_{meal_id}")

    df = get_model_data(
        config_data["list_attr"][0],
        config_data["list_attr"][1],
        series_data[config_data["list_attr"][0]][row[0]],
        series_data[config_data["list_attr"][1]][row[0]],
        config_data["time"],
        config_data["dep_var"],
        train,
    )
    # pm.plot_acf(df)
    # Initiating Model
    try:
        model_arima = AutoARIMA(df, config_data, exog=False)
        # Train and Test Data
        train_data, test_data = model_arima.train_test_split(df)
        # Fitting model on train data
        model_fit = model_arima.fit(train_data)
        # Prediction
        predictions = model_arima.predict(test_data, model_fit)
        dict_forecast[row[0]] = predictions

        # Storing Model Order

        output_forecast(
            train,
            config_data["list_attr"][0],
            config_data["list_attr"][1],
            series_data[config_data["list_attr"][0]][row[0]],
            series_data[config_data["list_attr"][1]][row[0]],
            list(predictions),
            config_data["forecast_horizon"],
            config_data,
        )
        """
        output_forecast_list(
            train,
            config_data["list_attr"][0],
            config_data["list_attr"][1],
            series_data[config_data["list_attr"][0]][row[0]],
            series_data[config_data["list_attr"][1]][row[0]],
            list(predictions),
            config_data["forecast_horizon"],
            config_data,
            forecast_list,
        )
        """
        model_order = model_arima.model_order(model_fit)
        model_row = {
            "series_id": row[0],
            "level_id": f"{center_id}_{meal_id}",
            "order": model_order,
        }
        model_order_df = model_order_df.append(model_row, ignore_index=True)
        # i_ = i.replace("|", "_")
        # Saving Model
        # joblib.dump(model_fit, f"{model_path}/arima_{i_}.pkl")
        model_arima = None
        model_fit = None
        predictions = None
        end_run = default_timer()
        logger.info(
            f"Elapsed time for model: {center_id}_{meal_id} is {end_run - start_run}"
        )
    except Exception as e:
        logger.info(f"Exception: {e}")


# Storing Model Order(s)
model_order_df.to_csv(
    os.path.join(
        config_data["output_path"],
        "auto_arima_model_order.csv",
    ),
    index=False,
)
"""
forecast_df = pd.DataFrame(forecast_list, columns=config_data["cols_for_table"])
train_forecast = train.copy()
train_forecast = train_forecast.append(forecast_df, ignore_index=True)


train_forecast.sort_values(by=config_data["time"], ascending=True, inplace=True)

output_df = pd.merge(date_df, train_forecast, how="inner", on=config_data["time"])
output_df.to_csv(
    os.path.join(
        config_data["output_path"],
        "forecast_values.csv",
    ),
    index=False,
)
"""
logger.info(f"Job finished. Elapsed time: {default_timer() - start}")