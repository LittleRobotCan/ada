import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

# resources
# https://stackoverflow.com/questions/35533253/how-would-you-do-randomizedsearchcv-with-votingclassifier-for-sklearn
def prep_matrix(data):
    """
    :param data: pandas data frame n feature 1 label
    :return: numpy arrays X and y
    """
    _y = data.iloc[:,-1]
    _X = data.iloc[:,:-1]
    y = np.array(_y)
    X = np.array(_X)
    return X, y


gb_params = {
    'random_state': 10,
    'n_estimators': 500,
    'max_depth':  8, #should be 5-8 depending on the number of observations and predictors
    'min_samples_leaf': 50,
    'min_samples_split': 500, # should be 0.5-1% of the total observations
    'verbose': 0,
    'max_features': 'sqrt', # general rule of thumb
    'subsample': 0.8 # commonly used start value
}

if __name__ == '__main__':
    from pandas import read_csv

    data = read_csv('data/numerai_training_data.csv')
    t_data = read_csv('data/numerai_tournament_data.csv')
    holdout = t_data[t_data['data_type']=='validation']
    #live = t_data[t_data['data_type']=='live']

    X, y = prep_matrix(data.ix[:,3:])
    X_holdout, y_holdout = prep_matrix(holdout.ix[:,3:])

    base_gb = GradientBoostingClassifier(random_state=10)
    base_gb.fit(X, y)
    proba_predictions = base_gb.predict_proba(X_holdout)
    l_loss = log_loss(y_holdout, proba_predictions)
    print l_loss
    #0.692627115819

