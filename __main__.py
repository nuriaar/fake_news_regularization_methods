import pandas as pd

from src.data_preprocessing import clean_text, transform_tfidf
from src.model import gradient_descent, predict, pred_to_accuracy
from src.viz import graph_log_loss


def read_data():
    """Read train and test data csvs
    """
    train = pd.read_csv("mathml_finalproj/data/train.csv")
    test = pd.read_csv("mathml_finalproj/data/test.csv")
    y_test = pd.read_csv("mathml_finalproj/data/submit.csv")
    return train, test, y_test


def main():
    train_raw, test_raw, y_test = read_data()
    train_clean = clean_text(train_raw)
    y_train = train_clean["label"]
    test_clean = clean_text(test_raw)

    #train, test, words = transform_countvec(train_clean["title"], test_clean["title"])
    X_train, X_test, words = transform_tfidf(train_clean["title"], test_clean["title"])
    print(X_test.todense())


if __name__ == "__main__":
    main()
