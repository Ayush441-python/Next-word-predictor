# from utils import clean_text

# def preprocess_data(df):
#     df['text']=df['text'].apply(clean_text)
#     return df

from utils import clean_text

def preprocess_data(df):
    df['text']=clean_text(df['text'])
    return df

