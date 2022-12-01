'''
Article text and title preprocessing. 
'''

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

stop_words = "set(stopwords.words('english'))"
nltk.download('stopwords')
stopwords = stopwords.words('english')
#stopwords.extend([])


def clean_col(col, stopwords, stem = False):
    '''Clean text column and return a list of cleaned tokens. 

    Inputs:
        col (Pandas Series): text column
        stem (boolean): if we want to reduce tokens to stems
        stopwords (list): stop words to remove
    
    Output:
        (Pandas Series) text column with clean data
    '''
    col = col.str.replace('[^\w\s]', '')\
        .str.replace('\d+','')\
        .str.lower()\
        .str.split()

    if stem:
        st = PorterStemmer()
        col = col.apply(lambda x: " ".join([st.stem(word) for word in x if not word in stopwords]))
    else:
        col = col.apply(lambda x: " ".join([word for word in x if not word in stopwords]))

    return col


def clean_text(data, stem=False, stopwords=stopwords): 
    '''Clean article text data.

    Inputs:
        data (Pandas DataFrame): with title and text columns
        stopwords (list): stop words to be removed
        stem (boolean): indicates if we want to apply stemming
    
    Output:
        (Pandas DataFrame)
    '''
    # Remove rows that are not articles
    data = data[pd.to_numeric(data['id'], errors='coerce').notnull()]
    data = data.dropna()
    
    # Clean text columns
    #data["title"] = clean_col(data['title'], stopwords, stem)
    data["text"] = clean_col(data['text'], stopwords, stem)

    return data


def transform_countvec(train_str, test_str):
    """Transform train and test text column into count vectorizer form. 

    Inputs: 
        train (Pandas Series of strings): text column of train data
        test (Pandas Series of strings): text column of test data
    
    Output: 
        train_countvec: matrix of word counts for the train text column
        test_countvec: matrix of word counts for the test text column
        words_countvec: list of words in count vectorizer 
    """

    countvectorizer = CountVectorizer(analyzer= 'word', max_df=0.5, min_df=30) #stop_words='english'
    countvec_terms = countvectorizer.fit_transform(train_str)
    words_countvec = countvectorizer.get_feature_names_out()
    train_countvec  = countvectorizer.transform(train_str)
    test_countvec  = countvectorizer.transform(test_str)
    
    return (train_countvec, test_countvec, words_countvec)


def transform_tfidf(train_str, test_str):
    """Transform train and test text column into TF-IDF vectorizer form. 

    Inputs: 
        train (Pandas Series of strings): text column of train data
        test (Pandas Series of strings): text column of test data
    
    Output: 
        train_countvec: tf-idf matrix for the train text column
        test_countvec: tf-idf matrix for the test text column
        words: list of words in tf-idf vectorizer
    """

    tfidfvectorizer = TfidfVectorizer(analyzer='word', max_df=0.5, min_df=30) # stop_words='english'
    tfidf_terms = tfidfvectorizer.fit_transform(train_str)
    words_tfidf = tfidfvectorizer.get_feature_names_out()
    train_tdidf  = tfidfvectorizer.transform(train_str)
    test_tfidf  = tfidfvectorizer.transform(test_str)
    
    return (train_tdidf, test_tfidf, words_tfidf)

