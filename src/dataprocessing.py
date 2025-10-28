import pandas as pd 
import numpy as np

def read_data(building: str) -> pd.DataFrame:
    """Reads the CSV data file into a pandas DataFrame."""
    weather_observed = pd.read_parquet(f'../data/raw/{building}/X_train_observed.parquet')
    weather_predicted = pd.read_parquet(f'../data/raw/{building}/X_train_estimated.parquet')
    weather = pd.concat((weather_observed, weather_predicted), axis = 0)
    weather = weather.set_index("date_forecast")
    weather = weather.drop(columns = "date_calc")

    target = pd.read_parquet(f'../data/raw/{building}/train_targets.parquet').set_index("time")

    return pd.concat((weather, target), axis=1)

def get_most_correlated_matrix(corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Get a list of features that have a correlation above the specified threshold.

    Parameters:
    corr_matrix (pd.DataFrame): The correlation matrix.
    threshold (float): The correlation threshold.

    Returns:
    pd.DataFrame: List of feature names with correlation above the threshold.
    """
    correlated_features = corr_matrix[corr_matrix.abs() >= threshold].index.tolist()
    uncorrelated_featr = corr_matrix[corr_matrix.abs() < threshold].index.tolist()
    print(correlated_features)

    return corr_matrix.drop(columns=uncorrelated_featr)