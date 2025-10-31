import lightgbm as lgb
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import dataprocessing as dp

#TODO:
#Read data
#Process/concat data
#Fit model
#Eval model

default_params = {
    "objective": "regression_l1",
    'metric': 'mae',
    "random_state":0,
    "n_estimators":500, 
    "bagging_freq":1,
    'learning_rate':  0.06389602179214016,  
    'num_leaves': 248, 
    'min_data_in_leaf': 2, 
    'bagging_fraction': 0.838404028491540, 
    'colsample_bytree': 0.7284801370182925,
    'early_stopping_round':50}

def preprocess_data(data: pd.DataFrame, features_to_keep: list = None, features_to_drop: list = None) -> pd.DataFrame:
    """
    Preprocess the input data for solar power prediction.
    This function removes redundant features based on a specified correlation threshold
    and identifies features with low correlation to the target feature. It allows for
    optional reindexing and retaining specific features.
    Args:
        data (pd.DataFrame): The input data as a pandas DataFrame.
        reindex (bool, optional): Whether to reindex the DataFrame. Defaults to False.
        features_to_keep (list, optional): A list of feature names to retain in the processed data despite correlation value.
            Defaults to None.
    Returns:
        pd.DataFrame: The preprocessed DataFrame with redundant features removed.
    """

    #Note: burde vel gjøre sun_azimuth om til radianer
    if not features_to_keep:
        features_to_keep = set(['pv_measurement', 'sun_azimuth:d', 'dew_or_rime:idx','date_forecast', 'building', "clear_sky_rad:W", "diffuse_rad:W", "direct_rad:W"] )
    if not features_to_drop:
        features_to_drop = set(["snow_drift:idx"])
    
    processed_data = data.copy()
    processed_data['building'] = LabelEncoder.fit_transform(processed_data['building'], processed_data['building'])

    processed_data = dp.remove_redundant_features(processed_data, threshold=0.93, target_feature='pv_measurement', return_list=False)
    corr_matrix = dp.get_correlation_matrix(processed_data, method='kendall')
    low_corr_features = dp.get_most_correlated_matrix(corr_matrix, threshold=0.11)

    #drop low correlation features except those in features_to_keep
    # print(low_corr_features)
    # print(f"Features to keep: {features_to_keep}")
    features_to_drop = (set(low_corr_features) | features_to_drop) - features_to_keep
    print(f"Total features to drop: {features_to_drop}")
    processed_data.drop(columns=features_to_drop, inplace=True)
    print(f"Dropped features due to low correlation: {features_to_drop}")
    print(f"Remaining features after preprocessing: {processed_data.columns.tolist()}, length: {len(processed_data.columns.tolist())}")

    #turn time features into cyclical features
    processed_data['hour_sin'] = np.sin(2 * np.pi * processed_data['date_forecast'].dt.hour / 24)
    processed_data['hour_cos'] = np.cos(2 * np.pi * processed_data['date_forecast'].dt.hour / 24)
    processed_data['day_sin'] = np.sin(2 * np.pi * processed_data['date_forecast'].dt.dayofyear / 365)
    processed_data['day_cos'] = np.cos(2 * np.pi * processed_data['date_forecast'].dt.dayofyear / 365)
    processed_data['month_sin'] = np.sin(2 * np.pi * processed_data['date_forecast'].dt.month / 12)
    processed_data['month_cos'] = np.cos(2 * np.pi * processed_data['date_forecast'].dt.month / 12)
    processed_data.drop(columns=['date_forecast'], inplace=True)

    return processed_data

def train_lightgbm_model(data: pd.DataFrame, target_column: str = 'pv_measurement', params: dict = default_params) -> lgb.Booster:
    """Train a LightGBM model on the provided data."""
    data  = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    print(data.columns)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['building', 'dew_or_rimeidx', 'precip_type_5minidx', 'snow_driftidx'])
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=['building', 'dew_or_rimeidx', 'precip_type_5minidx', 'snow_driftidx'])

    # model = lgb.LGBMRegressor(**params)
    # model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    train = lgb.Dataset(X_train, label=y_train)
    valid = lgb.Dataset(X_val, label=y_val, reference=train)

    model = lgb.train(params, train, valid_sets=[train, valid]) 

    """
    ['ceiling_height_aglm', 'clear_sky_radW', 'cloud_base_aglm',
       'dew_or_rimeidx', 'diffuse_radW', 'direct_radW',
       'effective_cloud_coverp', 'fresh_snow_1hcm', 'fresh_snow_3hcm',
       'fresh_snow_6hcm', 'precip_5minmm', 'precip_type_5minidx',
       'rain_waterkgm2', 'snow_densitykgm3', 'snow_driftidx',
       'snow_melt_10minmm', 'snow_waterkgm2', 'sun_azimuthd',
       'super_cooled_liquid_waterkgm2', 'visibilitym', 'wind_speed_10mms',
       'wind_speed_u_10mms', 'wind_speed_w_1000hPams', 'building', 'hour_sin',
       'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    """


    return model, y_val, X_val