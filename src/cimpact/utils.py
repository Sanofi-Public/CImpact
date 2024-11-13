"""
Utils method to handle data loading, validation, and conversion.
"""

from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf


def load_data(file_path, index_col=None, target_col=None):
    """
    Load data from a CSV file and return it along with the target column.

    Parameters:
    - file_path (str): Path to the CSV file.
    - index_col (str, optional): Column to set as the index.
    - target_col (str, optional): Target column to return.

    Returns:
    - pd.DataFrame: Loaded data.
    - pd.Series: Target column data.

    Raises:
    - ValueError: If the target column is not found in the data.
    """
    data = pd.read_csv(file_path, index_col=index_col, parse_dates=True)
    if target_col and target_col in data.columns:
        return data, data[target_col]
    raise ValueError("Target column not found in data")


def validate_data(data, pre_period, post_period):
    """
    Validate that the specified pre and post periods exist in the data index.

    Parameters:
    - data (pd.DataFrame): Input data.
    - pre_period (list): Pre-intervention period [start_date, end_date].
    - post_period (list): Post-intervention period [start_date, end_date].

    Returns:
    - bool: True if validation passes.

    Raises:
    - ValueError: If specified dates do not match data index.
    """
    if (
        pd.to_datetime(pre_period[0]) not in data.index
        or pd.to_datetime(post_period[1]) not in data.index
    ):
        raise ValueError("Specified pre or post period dates do not match data index")
    return True


def regularize_time_series(data, date_col="DATE"):
    """
    Regularize a time series data to have a consistent frequency.

    Parameters:
    - data (pd.DataFrame): Input data frame containing the time series.
    - date_col (str): Column name containing the dates.

    Returns:
    - pd.DataFrame: Regularized time series data.
    """
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.set_index(date_col)
    data = data.asfreq(freq=pd.infer_freq(data.index),  method='pad')
    return data


def convert_dates_to_indices(data, date_range):
    """
    Convert date strings to DataFrame indices.

    Parameters:
    - data (pd.DataFrame): The DataFrame with a DateTimeIndex.
    - date_range (list): A list of two date strings [start_date, end_date].

    Returns:
    - list: A list of two indices [start_index, end_index].
    """
    start_index = data.index.get_loc(pd.to_datetime(date_range[0]))
    end_index = data.index.get_loc(pd.to_datetime(date_range[1]))
    return [start_index, end_index]


def calculate_posterior_probabilities(post_effects_samples):
    """
    Calculate posterior tail-area probability and probability of a causal effect.

    Parameters:
    - post_effects_samples (np.array): Samples from posterior distribution of the absolute effects.

    Returns:
    - tuple: (tail_area_prob, causal_effect_prob)
    """
    tail_area_prob = np.mean(post_effects_samples < 0)
    causal_effect_prob = 1 - tail_area_prob
    return tail_area_prob, causal_effect_prob


def compute_p_value(
    simulated_ys: Union[np.array, tf.Tensor], post_data_sum: float
) -> float:
    """
    Compute the p-value for hypothesis testing.

    Parameters:
    - simulated_ys (Union[np.array, tf.Tensor]): Forecast simulations for value of y.
    - post_data

    Returns:
    - float: tail area probability and causal effect probability.
    """
    # Ensure the tensor has at least 2 dimensions
    if len(simulated_ys.shape) == 1:
        simulated_ys = tf.expand_dims(simulated_ys, axis=-1)

    # Reduce sum across the appropriate axis
    sim_sum = tf.reduce_sum(simulated_ys, axis=0 if len(simulated_ys.shape) == 1 else 1)
    signal = min(np.sum(sim_sum > post_data_sum), np.sum(sim_sum < post_data_sum))
    tail_area_prob = signal / (len(sim_sum) + 1)
    causal_effect_prob = 1 - tail_area_prob
    return tail_area_prob, causal_effect_prob
