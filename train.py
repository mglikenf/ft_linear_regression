from utils import normalize, load_dataset, estimate_price
import pandas as pd
import json


LEARNING_RATE = 0.1
NUM_ITERATIONS = 1000


def train(df: pd.DataFrame,
          theta0: float,
          theta1: float,
          learning_rate: float) -> dict:
    """
    Runs gradient descent on normalized mileage and price data to find
    optimal model coefficients.
    Returns a dict containing coefficients, normalization parameters,
    MSE data and coefficient data across iterations,
    for use in prediction and plotting.
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
    errors = []
    history = [[theta0, theta1]]
    for iteration in range(NUM_ITERATIONS):
        estimated_price = estimate_price(theta0, theta1, normalized_mileage)
        residuals = estimated_price - normalized_price
        tmp_theta0 = learning_rate * residuals.sum() / m
        tmp_theta1 = learning_rate * (residuals * normalized_mileage).sum() / m
        mse = (residuals ** 2).sum() / m
        errors.append(mse)
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
        history.append([float(theta0), float(theta1)])

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
        'max_price': max_price,
        'history': history,
        'errors': errors
    }
    return result


def main():
    """
    Prompts the user for a learning rate,
    loads the dataset, runs the training pipeline,
    and saves the result to model.json.
    """

    try:
        user_input = input('Enter a learning rate value: ')
        if user_input == '':
            learning_rate = LEARNING_RATE
        else:
            learning_rate = float(user_input)
        if learning_rate <= 0:
            raise ValueError
        df = load_dataset('data.csv')
        result = train(df, 0.0, 0.0, learning_rate)
        with open('model.json', 'w') as f:
            json.dump(result, f)
    except ValueError:
        print('Error: Learning rate must have a positive numeric value.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()
