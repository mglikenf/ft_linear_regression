import pandas as pd
import matplotlib.pyplot as plt


def normalize(raw_value, minimum, maximum):
    """
    Transforms raw value into normalized value using min-max normalization.
    """
    return (raw_value - minimum) / (maximum - minimum)


def denormalize(normalized_value, minimum, maximum):
    """
    Transforms normalized value back to its original scale using
    min-max denormalization.
    """
    return normalized_value * (maximum - minimum) + minimum


def estimate_price(theta0, theta1, mileage):
    """
    Returns estimated price for a given mileage using the linear model:
    theta0 + theta1 * mileage
    """
    return theta0 + theta1 * mileage


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads and validates a CSV dataset for training or display.
    Returns a valid DataFrame or raises an exception describing the failure.
    """
    if not isinstance(path, str):
        raise TypeError('Path must be a string')
    if not path.lower().endswith('.csv'):
        raise ValueError('Only .csv format is supported')
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f'{path} not found.')
    except Exception as e:
        raise OSError(f'Could not load dataset: {e}')

    if df.empty:
        raise ValueError('Dataset is empty')
    if not all(col in df.columns for col in ['km', 'price']):
        raise ValueError('Dataset must contain columns: km, price')
    if df[['km', 'price']].isnull().any().any():
        raise ValueError('Dataset contains missing values')
    return df
