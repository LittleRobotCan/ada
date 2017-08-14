from sklearn.svm import SVC
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from math import sqrt
import pandas as pd

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


class SKlearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        try:
            return self.clf.predict(x)
        except:
            print "nomodel is not fitted yet"

    def fit(self, x, y):
        self.clf.fit(x,y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x,y).feature_importances_)

# Put in our parameters for said classifiers
# Random Forest parameters
class stacker():
    def __init__(self):
        rf_params = {
            'n_jobs': -1,
            'n_estimators': 500,
             'warm_start': True,
             #'max_features': 0.2,
            'max_depth': 6,
            'min_samples_leaf': 2,
            'max_features' : 'sqrt',
            'verbose': 0
        }

        # Extra Trees Parameters
        et_params = {
            'n_jobs': -1,
            'n_estimators':500,
            #'max_features': 0.5,
            'max_depth': 8,
            'min_samples_leaf': 2,
            'verbose': 0
        }

        # AdaBoost parameters
        ada_params = {
            'n_estimators': 500,
            'learning_rate' : 0.75
        }

        # Gradient Boosting parameters
        gb_params = {
            'n_estimators': 500,
             #'max_features': 0.2,
            'max_depth': 5,
            'min_samples_leaf': 2,
            'verbose': 0
        }

        # Support Vector Classifier parameters
        svc_params = {
            'kernel' : 'linear',
            'C' : 0.025
            }

        SEED = 0 # for reproducibility
        self.rf = SKlearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
        self.et = SKlearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
        self.ada = SKlearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
        self.gb = SKlearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
        self.svc = SKlearnHelper(clf=SVC, seed=SEED, params=svc_params)

    # TODO: k-fold by era not target
    def stacking(self, X, y, X_holdout, y_holdout):
        n_splits = 5
        base_models = [self.rf, self.et, self.ada, self.gb, self.svc]
        stacks = []
        kf = KFold(n_splits=n_splits)
        holdout_matrix = []
        for model in base_models:
            X_stack = np.zeros((len(X)))
            holdout_col = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                model.fit(X_train, y_train)
                X_stack[test_index] = model.predict(X_test)
                holdout_col.extend(model.predict(X_holdout))
            stacks.append(X_stack)
            holdout_matrix.append(holdout_col)
        stacks_data = np.array(X_stack).transpose()
        stacks_df = pd.DataFrame(stacks_data, columns = ['rf', 'et', 'ada', 'gb', 'svc'])
        stacks_df['labels'] = y
        holdout_data = np.array(holdout_matrix).transpose()
        holdout_df = pd.DataFrame(holdout_data, columns = ['rf', 'et', 'ada', 'gb', 'svc'])
        holdout_df['labels'] = y_holdout*n_splits
        return stacks_df, holdout_df

    def second_learning(self, stacks_df):
        X, y = prep_matrix(stacks_df)
        gbm = xgb.XGBClassifier(        # learning_rate = 0.02,
            n_estimators=2000,
            max_depth=4,
            min_child_weight=2,
            # gamma=1,
            gamma=0.9,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=-1,
            scale_pos_weight=1).fit(X, y)
        return gbm


if __name__ == '__main__':
    from pandas import read_csv
    data = read_csv('data/numerai_training_data.csv')
    t_data = read_csv('data/numerai_tournament_data.csv')
    holdout = t_data[t_data['data_type']=='validation']
    X, y = prep_matrix(data.ix[:,3:])
    X_holdout, y_holdout = prep_matrix(holdout.ix[:,3:])
    stack = stacker()
    stacks_df, holdout_df = stack.stacking(X, y, X_holdout, y_holdout)
