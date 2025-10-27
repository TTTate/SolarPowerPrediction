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