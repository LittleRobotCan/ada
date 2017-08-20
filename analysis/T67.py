import pandas as pd
import bokeh
import pickle
import numpy as np

# df = pd.read_csv('data/numerai_training_data.csv')
#
# # plot everything in boxplot, separating x, and y
# for i in range(1, 22):
#     y = 'feature' + str(i)
#     boxplot(df, 'target', y)
#     print i
#     # signal is very weak

# plot the results of the models
train_meta = pd.read_csv('output/stacking10fold/train_meta.csv',index_col=0)
data = np.array(train_meta.iloc[:,:-1])
data_t = np.transpose(data)
cor_data = np.corrcoef(data_t)

"""
array([[ 1.        ,  0.74126359,  0.48310194,  0.32699856],
       [ 0.74126359,  1.        ,  0.40176485,  0.27775405],
       [ 0.48310194,  0.40176485,  1.        ,  0.41494169],
       [ 0.32699856,  0.27775405,  0.41494169,  1.        ]])
"""

holdout_meta = pd.read_csv('output/stacking10fold/holdout_meta.csv', index_col=0)
data = np.array(holdout_meta.iloc[:,:-1])
data_t = np.transpose(data)
cor_data = np.corrcoef(data_t)

"""
array([[ 1.        ,  0.78937079,  0.49843242,  0.34426891],
       [ 0.78937079,  1.        ,  0.40620843,  0.28949447],
       [ 0.49843242,  0.40620843,  1.        ,  0.43602881],
       [ 0.34426891,  0.28949447,  0.43602881,  1.        ]])
"""