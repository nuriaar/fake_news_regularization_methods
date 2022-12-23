import numpy as np
import pandas as pd

from src.data_preprocessing import clean_text, transform_tfidf
from src.model import predict, pred_to_accuracy


def read_data():
    """Read train and test data csv's.

    Returns:
        train (Pandas DataFrame): training data including label
        test (Pandas DataFrame): test data with no label
        y_test (Pandas DataFrame): label of test data 
    """
    train = pd.read_csv("mathml_finalproj/data/train.csv")
    test = pd.read_csv("mathml_finalproj/data/test.csv")
    y_test = pd.read_csv("mathml_finalproj/data/submit.csv")
    return train, test, y_test


def main():

    # Read data
    train_raw, test_raw, y_test_raw = read_data()

    # Implement data cleaning and text preprocessing
    test_raw = test_raw.join(y_test_raw["label"])
    train_clean = clean_text(train_raw)
    test_clean = clean_text(test_raw)
    y_train = train_clean["label"].to_numpy()
    y_test = test_clean["label"].to_numpy()

    # tf-idf transformation
    X_train, X_test, words = transform_tfidf(train_clean["text"], test_clean["text"])
    X_train = X_train.todense()
    X_test = X_test.todense()

    n, _ = X_train.shape
    # Shuffle training data
    indices = np.random.permutation(n)
    X_train = X_train[indices,:]
    y_train = y_train[indices]

    # Test data predictions
    pred_ridge = predict(X_train, y_train,  X_test, 0, 0.001, 0.01, 100000)
    pred_lasso = predict(X_train, y_train,  X_test, 1, 0.001, 0.01, 100000)
    pred_en = predict(X_train, y_train,  X_test, 0.6, 0.001, 0.01, 100000)

    # Calculate test accuracy
    threshold = 0.5
    accuracy_r = pred_to_accuracy(pred_ridge, y_test, threshold)
    accuracy_l = pred_to_accuracy(pred_lasso, y_test, threshold)
    accuracy_en = pred_to_accuracy(pred_en, y_test, threshold)
    print(accuracy_r)
    print(accuracy_l)
    print(accuracy_en)


if __name__ == "__main__":
    main()
