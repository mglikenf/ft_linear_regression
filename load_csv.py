import pandas as pd


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
        print(f"Loading dataset of dimensions {df.shape}")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
