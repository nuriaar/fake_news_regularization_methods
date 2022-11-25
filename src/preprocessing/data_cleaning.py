import pandas as pd

def clean_data(df):
    """
    """
    # Remove columns that are not articles
    df = df[pd.to_numeric(df['id'], errors='coerce').notnull()]
    df = df.dropna()

    return df
