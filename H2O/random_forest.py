import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

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


def split_by_era(eras, n_splits):
    era_set = list(set(eras))
    splits = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    for _, test_index in kf.split(era_set):
        test_era = [era_set[i] for i in test_index]
        train_era = [era for era in era_set if era not in test_era]
        test_index = [i for i in range(len(eras)) if eras[i] in test_era]
        train_index = [i for i in range(len(eras)) if eras[i] in train_era]
        splits.append({'test':test_index, 'train':train_index})
    return splits


rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'random_state': 0,
    # 'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

rf_grid_params = {
    'n_jobs': [-1],
    'n_estimators': [200,500,700],
    'warm_start': [True],
    'random_state': [0],
    'max_depth': [6],
    'min_samples_leaf': [2, 50, 70],
    'max_features': [4, 7, 10, 4],
    'verbose': [True]
}

if __name__ == '__main__':
    from pandas import read_csv

    data = read_csv('data/numerai_training_data.csv')
    t_data = read_csv('data/numerai_tournament_data.csv')
    holdout = t_data[t_data['data_type']=='validation']
    #live = t_data[t_data['data_type']=='live']

    X, y = prep_matrix(data.ix[:,3:])
    X_holdout, y_holdout = prep_matrix(holdout.ix[:,3:])

    base_rf = RandomForestClassifier(**rf_params)
    base_rf.fit(X, y)
    proba_predictions = base_rf.predict_proba(X_holdout)
    l_loss = log_loss(y_holdout, proba_predictions)
    print l_loss
    #0.692586716885

    # grid search
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, rf_grid_params)
    clf.fit(X, y)