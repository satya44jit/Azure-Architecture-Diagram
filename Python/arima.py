"""
AutoARIMA Module

"""
# built-in
import logging
import sys
import copy

# third party

from pmdarima import arima


LOGGING_LEVEL = "INFO"
PIPELINE_VERBOSE = LOGGING_LEVEL in ["INFO", "DEBUG"]

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


class AutoARIMA:
    def __init__(self, series, config, exog=False):
        # use deep copy so that we don't alter the reference
        # object.
        self.config = copy.deepcopy(config)
        self.series = series
        self.exog = exog
        return

    def train_test_split(self, series):
        forecast_horizon = self.config["forecast_horizon"]
        series = series.astype("float32")
        data_size = len(series)
        train_size = data_size - forecast_horizon
        self.train_data, self.test_data = series[:train_size], series[train_size:]
        return self.train_data, self.test_data

    def train(self, train_data, config):
        if self.exog:
            # TODO: Add Raise Exceptions
            # if the key for indep_var is missing or empty
            self.exog = self.train_data[config["indep_var"]]
            self.model_def = arima.auto_arima(
                self.train_data[config["dep_var"]],
                exog=self.exog,
                start_p=0,
                start_q=0,
                max_p=3,
                max_q=3,
                m=52,
                start_P=0,
                start_Q=0,
                trend="ct",
                seasonal=True,
                # trace=True,
                out_of_sample_size=config["holdout_period"],
                error_action="ignore",
                suppress_warnings=True,  # don't want convergence warnings
                stepwise=True,
            )
        else:
            self.model_def = arima.auto_arima(
                self.train_data[config["dep_var"]],
                start_p=0,
                start_q=0,
                max_p=3,
                max_q=3,
                m=52,
                start_P=0,
                start_Q=0,
                trend="ct",
                seasonal=True,
                # trace=True,
                out_of_sample_size=config["holdout_period"],
                error_action="ignore",
                suppress_warnings=True,  # don't want convergence warnings
                stepwise=True,
            )

        return self.model_def

    def fit(self, train_data):
        config = self.config
        model_def = self.train(train_data, self.config)
        if len(self.config["indep_var"]) > 0:
            model_fit = model_def.fit(
                train_data[config["dep_var"]], exogenous=train_data[config["indep_var"]]
            )
        else:
            model_fit = model_def.fit(train_data[config["dep_var"]])
        return model_fit

    def model_order(self, model):
        self.selected_order = []
        if len(self.config["indep_var"]) > 0:
            self.selected_order.extend((model.order, model.seasonal_order, model.trend))
        else:
            self.selected_order.extend((model.order, model.trend))
        return self.selected_order

    def predict(self, test, model):
        if len(self.config["indep_var"]) > 0:
            predictions = model.predict(
                n_periods=self.config["forecast_horizon"],
                exogenous=test[self.config["indep_var"]],
            )
        else:
            predictions = model.predict(
                n_periods=self.config["forecast_horizon"],
            )
        return predictions
