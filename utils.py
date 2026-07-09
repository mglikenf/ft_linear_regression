import pandas as pd


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


def load(path: str) -> pd.DataFrame:
    """
    Loads a dataset from the given file path,
    displays its dimensions and returns a DataFrame.
    Returns None if path is invalid or file cannot be loaded.
    """
    if not isinstance(path, str):
        print('Error: path must be a string')
        return None
    if not path.lower().endswith('.csv'):
        print('Error: only .csv format is supported')
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
