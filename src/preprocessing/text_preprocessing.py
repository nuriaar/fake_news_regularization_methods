'''
Article text and title preprocessing. 
'''

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stop_words = "set(stopwords.words('english'))"
nltk.download('stopwords')
stopwords = stopwords.words('english')
#stopwords.extend([])


def clean_text(data, stem=False): #stopwords=stopwords, 
    '''Clean article title and text data.

    Inputs:
        data (Pandas DataFrame): with title and text columns
        stopwords (list): stop words to be removed
        stem (boolean): indicates if we want to apply stemming
    
    Output:
        (Pandas DataFrame)
    '''
    data["title"] = clean_col(data['title'], stopwords, stem)
    data["text"] = clean_col(data['text'], stopwords, stem)

    return data


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
