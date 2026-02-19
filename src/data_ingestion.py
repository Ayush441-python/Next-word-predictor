import pandas as pd

def load_data(path):
    df = pd.read_csv("fake_or_real_news.csv")
    df = df[['text', 'label']]
    return df


