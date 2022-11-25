import pandas as pd
from src.preprocessing.text_preprocessing import clean_text


def read_data():
    """Read train and test data csvs
    """
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test


def main():
    train, test = read_data()
    print(clean_text(train))


if __name__ == "__main__":
    main()
