from utils import normalize, denormalize, estimate_price
import json


def main():
    """
    Prompts the user for a mileage value, loads trained model parameters
    from thetas.json, and prints the estimated car price.
    Requires train.py to have been run first.
    """
    try:
        with open('thetas.json', 'r') as f:
            data = json.load(f)
        theta0, theta1 = data['theta0'], data['theta1']
        min_mileage, max_mileage = data['min_mileage'], data['max_mileage']
        min_price, max_price = data['min_price'], data['max_price']

        user_input = input('Enter car\'s mileage to get price prediction: ')
        mileage = float(user_input)
        if mileage < 0:
            raise ValueError
        mileage = int(round(mileage))
        normalized_mileage = normalize(mileage, min_mileage, max_mileage)
        estimated_price = estimate_price(theta0, theta1, normalized_mileage)
        raw_estimated_price = denormalize(estimated_price, min_price, max_price)
        if raw_estimated_price < 0:
            print('Warning: mileage is higher than the training range. Prediction may be unreliable.')
            raw_estimated_price = 0
        print(f'Estimated price for {mileage} mile car: {round(raw_estimated_price, 2)}')

    except (json.JSONDecodeError, KeyError):
        print('Error: thetas.json is malformed. Please run train.py again.')
    except ValueError:
        print('Error: Input must be a numeric positive value.')
    except (FileNotFoundError, OSError) as e:
        print(f'Error: Could not open thetas.json: {e}')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()