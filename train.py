from utils import normalize, load_dataset, estimate_price
import pandas as pd
import json
import time


LEARNING_RATE = 0.5
NUM_ITERATIONS = 1000


def train(df: pd.DataFrame, theta0: float, theta1: float) -> dict:
    """
    Runs gradient descent on normalized mileage and price data to find
    optimal theta0 and theta1, and returns them alongside normalization
    parameters for use in prediction.
    """

    mileage = df['km'].to_numpy()
    actual_price = df['price'].to_numpy()
    m = len(mileage)

    min_mileage = float(min(mileage))
    max_mileage = float(max(mileage))
    min_price = float(min(actual_price))
    max_price = float(max(actual_price))
    normalized_mileage = normalize(mileage, min_mileage, max_mileage)
    normalized_price = normalize(actual_price, min_price, max_price)

    for iteration in range(NUM_ITERATIONS):
        estimated_price = estimate_price(theta0, theta1, normalized_mileage)
        residuals = estimated_price - normalized_price

        tmp_theta0 = LEARNING_RATE * residuals.sum() / m
        tmp_theta1 = LEARNING_RATE * (residuals * normalized_mileage).sum() / m

        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

        time.sleep(0.02)
        print(f"\r{iteration + 1}/{NUM_ITERATIONS}: "
              f"theta0 = {theta0}, theta1 = {theta1}",
              end="", flush=True)
    print()

    result = {
        'theta0': float(theta0),
        'theta1': float(theta1),
        'min_mileage': min_mileage,
        'max_mileage': max_mileage,
        'min_price': min_price,
        'max_price': max_price
    }
    return result


def main():
    """
    Validates and loads the dataset, runs the training pipeline,
    and saves the resulting model parameters to thetas.json.
    """

    try:
        df = load_dataset('data.csv')
        result = train(df, 0.0, 0.0)
        with open('thetas.json', 'w') as f:
            json.dump(result, f)
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()
