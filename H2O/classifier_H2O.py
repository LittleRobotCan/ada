import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from __future__ import print_function
h2o.init()

from pandas import read_csv

data = read_csv('data/numerai_training_data.csv')
t_data = read_csv('data/numerai_tournament_data.csv')
holdout = t_data[t_data['data_type'] == 'validation']

data = data.ix[:,3:]
x = data.columns
y = 'response'
x.remove('target')
data.columns = x+['target']

x[-1] = 'response'
x = data.columns

y = "response"
x.remove(y)