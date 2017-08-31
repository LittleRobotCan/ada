import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


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

rf_grid_params_1 = {
    'n_jobs': [-1],
    'n_estimators': [150, 200,250],
    'warm_start': [True],
    'random_state': [0],
    'max_depth': [6],
    'min_samples_leaf': [2, 5, 7, 80],
    'max_features': [10, 14, 18],
    'verbose': [True]
}

rf_grid_params_2 = {
    'n_jobs': [-1],
    'n_estimators': [50, 150, 200, 300],
    'warm_start': [True],
    'random_state': [0],
    'max_depth': [6],
    'min_samples_leaf': [2,80,150,200],
    'max_features': [7, 14, 21],
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

    """
    SEARCH 1
    """
    # grid search
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, rf_grid_params)
    clf.fit(X, y)
    print clf.best_estimator_
    print clf.best_params_
    # {'warm_start': True, 'n_jobs': -1, 'verbose': True, 'min_samples_leaf': 2, 'n_estimators': 200, 'random_state': 0,
    #  'max_features': 10, 'max_depth': 6}

    best_rf = clf.best_estimator_
    proba_predictions = best_rf.predict_proba(X_holdout)
    l_loss = log_loss(y_holdout, proba_predictions)
    print l_loss
    # 0.692491910196

    """
    SEARCH 2
    """
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, rf_grid_params_1)
    clf.fit(X, y)
    print clf.best_estimator_
    print clf.best_params_
    # {'warm_start': True, 'n_jobs': -1, 'verbose': True, 'min_samples_leaf': 80, 'n_estimators': 150, 'random_state': 0,
    # 'max_features': 14, 'max_depth': 6}
    best_rf_1 = clf.best_estimator_
    proba_predictions = best_rf_1.predict_proba(X_holdout)
    l_loss = log_loss(y_holdout, proba_predictions)
    print l_loss
    # 0.692619310732

    """
    SEARCH 3
    """
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, rf_grid_params_2)
    clf.fit(X, y)
    print clf.best_estimator_
    print clf.best_params_
    best_rf_2 = clf.best_estimator_
    proba_predictions = best_rf_2.predict_proba(X_holdout)
    l_loss = log_loss(y_holdout, proba_predictions)
    print l_loss