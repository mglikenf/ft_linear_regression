from utils import normalize, denormalize, estimate_price
import json


def main():
    """
    Prompts the user for a mileage value.
    Loads trained model parameters from model.json.
    Computes and displays the estimated car price.
    Requires train.py to have been run first.
    """
    try:
        with open('model.json', 'r') as f:
            data = json.load(f)
        theta0, theta1 = data['theta0'], data['theta1']
        min_mileage, max_mileage = data['min_mileage'], data['max_mileage']
        min_price, max_price = data['min_price'], data['max_price']

        user_input = input("Enter car's mileage to get price prediction: ")
        mileage = float(user_input)
        if mileage < 0:
            raise ValueError('Input must be a positive value. '
                             'Minimum: 0 km')
        if mileage > 1000000:
            raise ValueError('Input exceeds reasonable range. '
                             'Maximum: 1,000,000 km')
        mileage = int(round(mileage))
        normalized_mileage = normalize(mileage, min_mileage, max_mileage)
        norm_estimated = estimate_price(theta0, theta1, normalized_mileage)
        raw_estimated = denormalize(norm_estimated, min_price, max_price)
        if raw_estimated < 0:
            print('Warning: mileage is higher than the training range. '
                  'Prediction may be unreliable.')
            raw_estimated = 0
        print(f'Estimated price for {mileage} mile car: '
              f'{round(raw_estimated, 2)}')

    except (json.JSONDecodeError, KeyError):
        print('Error: model.json is malformed. Please run train.py again.')
    except FileNotFoundError:
        print('Error: model.json not found. Please run train.py first.')
    except (KeyboardInterrupt, EOFError):
        print('\nError: Aborted.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()
