"""
This code is for the utilies for data connections and importing data from local
"""
# Importing all the necessary packages
import pandas as pd
import os


def load_data(path, data_file):
    """This function is to load a csv data file from a path on
    a local machine

    Args:
        path (str): path to the input csv file
        data_file (str): Name of the input csv file to be imported
    Returns:
        df (DataFrame): Output Dataframe that is converted from a csv input file
    """
    return pd.read_csv(os.path.join(path, data_file))