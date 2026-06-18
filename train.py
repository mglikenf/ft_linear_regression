from load_csv import load
import matplotlib.pyplot as plt


def main():

    df = load("data.csv")
    print(df)
    print(df.dtypes)
    if df is None:
        return

    plt.scatter(df['km'], df['price'])
    plt.title('Car price prediction with linear regression')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.show()



if __name__ == "__main__":
    main()
