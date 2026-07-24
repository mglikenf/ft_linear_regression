from utils import load_dataset, estimate_price, normalize, denormalize
import matplotlib.pyplot as plt
import numpy as np
import json


def main():
    """
    Loads trained model parameters from model.json and displays
    a related plots.
    """
    try:
        df = load_dataset('data.csv')
        mileage, price = df['km'], df['price']
        with open('model.json', 'r') as f:
            data = json.load(f)
        theta0, theta1 = data['theta0'], data['theta1']
        min_mileage, max_mileage = data['min_mileage'], data['max_mileage']
        min_price, max_price = data['min_price'], data['max_price']
        errors = data['errors']
        history = data['history']

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))

        # left - scatter plot with regression line
        x_line = np.array([min_mileage, max_mileage])
        x_line_norm = normalize(x_line, min_mileage, max_mileage)
        y_line_norm = estimate_price(theta0, theta1, x_line_norm)
        y_line = denormalize(y_line_norm, min_price, max_price)

        axs[0].scatter(mileage, price, label='observations')
        axs[0].set_title('estimatePrice(mileage) = θ0 + (θ1 * mileage)')
        axs[0].set_xlabel('Mileage (km)')
        axs[0].set_ylabel('Price ($)')
        axs[0].plot(x_line, y_line, c='r', linewidth=2, label='model')
        axs[0].grid()
        axs[0].legend()

        # center - mse across iterations
        axs[1].set_title('MSE Over Iterations')
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel('MSE')
        axs[1].plot(range(len(errors)), errors, label='MSE', linewidth=2)
        axs[1].grid()
        axs[1].legend()

        # right - theta0 and theta1 across iterations
        t0, t1 = zip(*history)
        axs[2].set_title('θ0 and θ1 Convergence')
        axs[2].set_xlabel('Iterations')
        axs[2].set_ylabel('Theta Values')
        axs[2].plot(range(len(t0)), t0, label='θ0', linewidth=2)
        axs[2].plot(range(len(t1)), t1, label='θ1', linewidth=2)
        axs[2].grid()
        axs[2].legend()

        plt.tight_layout(pad=3.0)
        plt.show()

    except (json.JSONDecodeError, KeyError):
        print('Error: model.json is malformed. Please run train.py again.')
    except FileNotFoundError as e:
        if e.filename == 'model.json':
            print('Error: model.json not found. Please run train.py first.')
        else:
            print(f'Error: {e}')
    except (ValueError, OSError) as e:
        print(f'Error: {e}')
    except KeyboardInterrupt:
        print('\nError: Aborted.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()
