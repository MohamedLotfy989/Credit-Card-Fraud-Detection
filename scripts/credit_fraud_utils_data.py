import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import  RobustScaler


def load_data(train_file_path, val_file_path):
    """
    Load the training and validation datasets and split into features (X) and labels (y).

    Args:
        train_file_path (str): Path to the training dataset CSV file.
        val_file_path (str): Path to the validation dataset CSV file.

    Returns:
        tuple: X_train, y_train, X_val, y_val
    """
    # Load training data
    df_train = pd.read_csv(train_file_path)
    X_train = df_train.drop(columns=["Class"])
    y_train = df_train["Class"]

    # Load validation data
    df_val = pd.read_csv(val_file_path)
    X_val = df_val.drop(columns=["Class"])
    y_val = df_val["Class"]

    # Apply scaling to all features
    X_train, X_val = scale_features(X_train, X_val)

    return X_train, y_train, X_val, y_val

def scale_features(X_train, X_val):
    """
    Scale all features using RobustScaler.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_val (pd.DataFrame): Validation feature set.

    Returns:
        tuple: Scaled training and validation feature sets.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled



def resample_data(X, y, technique="smote"):
    """
    Resample data using the specified technique to handle class imbalance.

    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target labels.
        technique (str): Resampling technique to use. Options are "smote", "random_over", "random_under", "weighted_random".

    Returns:
        tuple: Resampled X and y.
    """
    if technique == "smote":
        sampler = SMOTE(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    elif technique == "random_over":
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    elif technique == "random_under":
        sampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    else:
        raise ValueError("Unsupported resampling technique. Choose from 'smote', 'random_over', 'random_under', 'weighted_random'.")

    return X_resampled, y_resampled
