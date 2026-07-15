from utils import load_dataset, estimate_price, normalize, denormalize
import matplotlib.pyplot as plt
import numpy as np
import json


def main():
    """
    Loads trained model parameters from thetas.json and displays
    a scatter plot of the dataset with the regression line overlaid.
    """
    try:
        df = load_dataset('data.csv')
        with open('thetas.json', 'r') as f:
            data = json.load(f)
        theta0, theta1 = data['theta0'], data['theta1']
        min_mileage, max_mileage = data['min_mileage'], data['max_mileage']
        min_price, max_price = data['min_price'], data['max_price']

        x_line_raw = np.array([min_mileage, max_mileage])
        x_line_norm = normalize(x_line_raw, min_mileage, max_mileage)
        y_line_norm = estimate_price(theta0, theta1, x_line_norm)
        y_line_raw = denormalize(y_line_norm, min_price, max_price)

        plt.scatter(df['km'], df['price'])
        plt.title('Car price prediction with linear regression')
        plt.xlabel('mileage')
        plt.ylabel('price')
        plt.plot(x_line_raw, y_line_raw)
        plt.show()
    except (json.JSONDecodeError, KeyError):
        print('Error: thetas.json is malformed. Please run train.py again.')
    except FileNotFoundError as e:
        if e.filename == 'thetas.json':
            print(f'Error: thetas.json not found. Please run train.py first.')
        else:
            print(f'Error: {e}')
    except (ValueError, OSError) as e:
        print(f'Error: {e}')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()
