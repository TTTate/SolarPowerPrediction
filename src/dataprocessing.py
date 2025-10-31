import pandas as pd 
import numpy as np

def read_data(building: str, reindex: bool = True) -> pd.DataFrame:
    """Reads the CSV data file into a pandas DataFrame."""
    weather_observed = pd.read_parquet(f'../data/raw/{building}/X_train_observed.parquet')
    weather_predicted = pd.read_parquet(f'../data/raw/{building}/X_train_estimated.parquet')
    weather = pd.concat((weather_observed, weather_predicted), axis = 0)


    if reindex:
        weather = weather.set_index("date_forecast")
        weather = weather.drop(columns = "date_calc")
        target = pd.read_parquet(f'../data/raw/{building}/train_targets.parquet').set_index("time")

        return pd.concat((weather, target), axis=1)

    # re-index is only used for eda. For model use, we use integer index and process later
    else:
        weather_observed["forecast"] = '0'
        weather_predicted["forecast"] = '1'

        weather['building'] = building

        target = pd.read_parquet(f'../data/raw/{building}/train_targets.parquet')
        return weather.merge(target, left_on='date_forecast', right_on='time', how='inner').drop(columns=['date_calc', 'time'])

def get_correlation_matrix(data: pd.DataFrame, method: str = 'kendall') -> pd.DataFrame:
    """
    Calculate the correlation matrix of the given DataFrame using the specified method.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    method (str): The correlation method to use ('pearson', 'kendall', 'spearman').

    Returns:
    pd.DataFrame: The correlation matrix.
    """
    correlation_matrix = data.corr(method=method)
    correlation_matrix.dropna(thresh=2, inplace=True)
    correlation_matrix.dropna(axis=1, thresh=2, inplace=True)
    return correlation_matrix

def get_most_correlated_matrix(corr_matrix: pd.DataFrame, threshold: float, target_feature: str = 'pv_measurement') -> pd.DataFrame:
    """
    Get a list of features that have a correlation above the specified threshold.

    Parameters:
    corr_matrix (pd.DataFrame): The correlation matrix.
    threshold (float): The correlation threshold.

    Returns:
    pd.DataFrame: List of feature names with correlation above the threshold.
    """
    target_corr = corr_matrix[target_feature].abs()

    # Find features with correlation above the threshold
    features_to_keep = target_corr[target_corr >= threshold].index.tolist()

    # Filter the correlation matrix to keep only the selected features
    filtered_corr_matrix = corr_matrix.loc[features_to_keep, features_to_keep]

    return filtered_corr_matrix


def filter_correlation_matrix(corr_matrix: pd.DataFrame, threshold: float, target_feature: str = 'pv_measurement', drop_low_corr: bool = True) -> pd.DataFrame:
    """
    Filter correlation matrix to keep only features with correlation to target above threshold.
    
    Parameters:
    corr_matrix (pd.DataFrame): The correlation matrix.
    threshold (float): The correlation threshold (absolute value) relative to target feature.
    target_feature (str): The target feature to filter by (default: 'pv_measurement').
    drop_low_corr (bool): If True, drop rows/cols below threshold to target.
    
    Returns:
    pd.DataFrame: Filtered correlation matrix containing only features correlated with target.
    """
    # Check if target exists in matrix
    if target_feature not in corr_matrix.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in correlation matrix")
    
    # Get correlations to target
    target_corr = corr_matrix[target_feature].abs()
    
    # Find features above threshold (including target itself)
    features_to_keep = target_corr[target_corr >= threshold].index.tolist()
    
    if drop_low_corr:
        # Keep only rows and columns for features above threshold
        filtered_matrix = corr_matrix.loc[features_to_keep, features_to_keep]
    else:
        # Keep all features but set low correlations to NaN
        filtered_matrix = corr_matrix.copy()
        features_to_drop = target_corr[target_corr < threshold].index.tolist()
        
        # Set correlations to NaN for features below threshold
        for feature in features_to_drop:
            filtered_matrix.loc[:, feature] = np.nan
            filtered_matrix.loc[feature, :] = np.nan
    
    return filtered_matrix

def remove_redundant_features(data: pd.DataFrame, threshold: float = 0.95, target_feature: str = 'pv_measurement', return_list: bool = True) -> pd.DataFrame: #TO-DO: Review this function
    """
    Remove redundant features that are highly correlated with each other based on Pearson correlation.
    
    For each pair of features with Pearson correlation above threshold, keeps the one 
    with higher correlation to the target feature.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    threshold (float): The Pearson correlation threshold for redundancy (default: 0.95).
    target_feature (str): The target feature to prioritize when choosing which feature to keep (default: 'pv_measurement').
    
    Returns:
    pd.DataFrame: DataFrame with redundant features removed.
    """
    # Calculate Pearson correlation matrix
    corr_matrix = data.corr(method='pearson')
    
    # Check if target exists
    if target_feature not in corr_matrix.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in data")
    
    # Get correlation to target
    target_corr = corr_matrix[target_feature].abs()
    
    # Create upper triangle of correlation matrix (excluding diagonal)
    corr_upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Decide which feature to drop from each pair
    redundant_pairs = find_redundant_pairs(corr_upper, threshold)
    features_to_drop = find_features_to_drop(redundant_pairs, target_corr, target_feature)
    
    print(f"\nRedundant feature removal (Pearson correlation >= {threshold}):")
    print(f"  Original features: {data.shape[1]}")
    print(f"  Redundant pairs found: {len(redundant_pairs)}")
    print(f"  Redundant pairs: {redundant_pairs}")
    print(f"  Features to drop: {sorted(features_to_drop)}")
    print(f"  Features remaining: {data.shape[1] - len(features_to_drop)}")
    
    if return_list:
        return features_to_drop
    else:
        # Drop redundant features
        data_filtered = data.copy()
        data_filtered.drop(columns=list(features_to_drop), inplace=True)
        
        return data_filtered

def find_redundant_pairs(corr_upper: pd.DataFrame, threshold: float) -> list:
    """
    Find all pairs of features with correlation above the specified threshold.
    
    Parameters:
    corr_upper (pd.DataFrame): Upper triangle of the correlation matrix.
    threshold (float): Correlation threshold to identify redundant pairs.
    
    Returns:
    list: List of tuples representing redundant feature pairs.
    """
    redundant_pairs = []
    for column in corr_upper.columns:
        # Find features highly correlated with this column
        high_corr = corr_upper[column][corr_upper[column].abs() >= threshold]
        
        for other_feature in high_corr.index:
            redundant_pairs.append((column, other_feature))
    
    return redundant_pairs

def find_features_to_drop(redundant_pairs: list, target_corr: pd.Series, target_feature: str) -> set:
    """
    Determine which features to drop from redundant pairs based on correlation to target.
    Parameters:
    redundant_pairs (list): List of tuples representing redundant feature pairs.
    target_corr (pd.Series): Series containing correlation of each feature to the target.
    target_feature (str): The target feature to prioritize when choosing which feature to keep.
    Returns:
    set: Set of feature names to drop.
    """

    features_to_drop = set()
    for feat1, feat2 in redundant_pairs:
        # Skip if both are already marked for dropping
        if feat1 in features_to_drop and feat2 in features_to_drop:
            continue
        
        # Don't drop the target feature
        if feat1 == target_feature:
            features_to_drop.add(feat2)
        elif feat2 == target_feature:
            features_to_drop.add(feat1)
        # Keep the feature with higher correlation to target
        elif target_corr[feat1] >= target_corr[feat2]:
            features_to_drop.add(feat2)
        else:
            features_to_drop.add(feat1)
    return features_to_drop