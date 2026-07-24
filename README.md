# ft_linear_regression

_This project has been created by [mglikenf](https://github.com/mglikenf) as part of the 42 School curriculum._

A from-scratch implementation of linear regression trained with gradient descent, predicting car prices from mileage.

---

## Description

This project trains a single-variable model by minimizing MSE with gradient descent, then uses the trained coefficients to predict car prices from mileage.

### How It Works

The model learns the relationship between a car's mileage and its price by minimizing the MSE using gradient descent. Both mileage and price are normalized using min-max scaling before training, and the normalization parameters are saved alongside the trained coefficients for use at prediction time.

The hypothesis:

```
estimatePrice(mileage) = θ0 + θ1 * mileage
```

where θ0 is the intercept (bias) and θ1 is the slope of the regression line.

## Project Structure

```
ft_linear_regression/
├── train.py            # Training program: gradient descent, saves model to model.json
├── predict.py          # Prediction program: prompts for mileage, outputs estimated price
├── display.py          # Bonus: data visualization
├── precision.py        # Bonus: computes and displays R-squared of the trained model
├── utils.py            # Shared utilities
├── data.csv            # Training dataset (24 car mileage/price pairs)
├── requirements.txt
├── LICENSE
└── README.md
```

**Note:** `train.py` must be run first. It generates `model.json`, which is required by `predict.py`, `precision.py`, and `display.py`.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv linear_venv
source linear_venv/bin/activate
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

---

## Usage

### 1. Train The Model (mandatory)

Prompts the user for a learning rate, and defaults to 0.1 if no value was provided.
Reads `data.csv`, runs gradient descent, and saves model parameters to `model.json`:

```bash
python3 train.py
```

### 2. Predict The Price (mandatory)

Prompts for a mileage value and outputs the estimated price:

```bash
python3 predict.py
```

If `train.py` has not been run yet, `predict.py` will exit with an error indicating that `model.json` is missing.

### 3. Display The Data (bonus)

Displays three plots:
- Scatter plot of the dataset with the trained regression line overlaid
- MSE convergence across iterations
- Model's parameters across iterations

```bash
python3 display.py
```

### 4. Evaluate Model Precision (bonus)

Computes and displays the R-squared of the trained model:

```bash
python3 precision.py
```
