from utils import load_dataset, estimate_price, normalize, denormalize
import json


def main():
    """
    Loads dataset, and model params from model.json.
    Calculates and displays the R-squared of the model.
    """
    try:
        df = load_dataset('data.csv')
        mileage = df['km'].to_numpy()
        actual_price = df['price'].to_numpy()
        with open('model.json', 'r') as f:
            data = json.load(f)
        theta0, theta1 = data['theta0'], data['theta1']
        min_mileage, max_mileage = data['min_mileage'], data['max_mileage']
        min_price, max_price = data['min_price'], data['max_price']

        mean_price = actual_price.mean()
        ss_tot = ((actual_price - mean_price) ** 2).sum()
        normalized_mileage = normalize(mileage, min_mileage, max_mileage)
        norm_est_price = estimate_price(theta0, theta1, normalized_mileage)
        estimated_price = denormalize(norm_est_price, min_price, max_price)
        ss_res = ((estimated_price - actual_price) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R-squared: {round(r_squared, 4)}, '
              f'({round(r_squared * 100, 2)}% of variance explained by model)')

    except (json.JSONDecodeError, KeyError):
        print('Error: model.json is malformed. Please run train.py again.')
    except FileNotFoundError as e:
        if e.filename == 'model.json':
            print('Error: model.json not found. Please run train.py first.')
        else:
            print(f'Error: {e}')
    except (ValueError, OSError) as e:
        print(f'Error: {e}')
    except (KeyboardInterrupt, EOFError):
        print('\nError: Aborted.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
