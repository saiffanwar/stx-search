import pandas as pd
import numpy as np



def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

df = read_data('wikipedia.csv')
print(np.unique(df['label'], return_counts=True))
print(df.iloc[372])
