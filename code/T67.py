import pandas as pd
from analysis import *

df = pd.read_csv('data/numerai_training_data.csv')

# plot everything in boxplot, separating x, and y
for i in range(1, 22):
    y = 'feature' + str(i)
    boxplot(df, 'target', y)
    print i
    # signal is very weak