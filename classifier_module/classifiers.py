import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import pandas as pd
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


# TODO: svc predicting all "0" in base learner
# TODO: predict proba instead of actual predictions
class base_learner():
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

    def stacking(self, X, y, X_holdout, y_holdout, X_live, y_live, splits):
        base_models = [self.rf, self.et, self.ada, self.gb]
        stacks = []
        holdout_matrix = []
        live_matrix = []
        for model in base_models:
            print "model"
            X_stack = np.zeros((len(X)))
            holdout_col = []
            live_col = []
            for index in splits:
                print "split"
                X_train, X_test = X[index['train']], X[index['test']]
                y_train = y[index['train']]
                model.fit(X_train, y_train)
                X_stack[index['test']] = model.predict(X_test)
                holdout_col.extend(model.predict(X_holdout))
                live_col.extend(model.predict(X_live))
            stacks.append(X_stack)
            holdout_matrix.append(holdout_col)
            live_matrix.append(live_col)
        stacks_data = np.array(stacks).transpose()
        stacks_df = pd.DataFrame(stacks_data, columns = ['rf', 'et', 'ada', 'gb'])
        stacks_df['labels'] = y
        holdout_data = np.array(holdout_matrix).transpose()
        live_data = np.array(live_matrix).transpose()
        holdout_df = pd.DataFrame(holdout_data, columns = ['rf', 'et', 'ada', 'gb'])
        live_df = pd.DataFrame(live_data, columns = ['rf', 'et', 'ada', 'gb'])
        holdout_df['labels'] = list(y_holdout)*len(splits)
        live_df['labels'] = 'unkonwn'
        return stacks_df, holdout_df, live_df


def top_learner(stacks_df):
    X, y = prep_matrix(stacks_df)
    gbm = XGBClassifier(
        # learning_rate = 0.02,
        n_estimators=2000,
        max_depth=4,
        min_child_weight=2,
        # gamma=1,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        scale_pos_weight=1).fit(X, y)
    return gbm


def evaluate(truth, predict_proba):
    l_loss = log_loss(truth, predict_proba)
    return l_loss


if __name__ == '__main__':
    from pandas import read_csv
    import pickle

    n_splits = 10

    data = read_csv('data/numerai_training_data.csv')
    #data = raw_data.sample(100)
    t_data = read_csv('data/numerai_tournament_data.csv')
    holdout = t_data[t_data['data_type']=='validation']
    live = t_data[t_data['data_type']=='live']
    #holdout = holdout_raw.sample(100)
    #live = live_raw.sample(100)

    X, y = prep_matrix(data.ix[:,3:])
    X_holdout, y_holdout = prep_matrix(holdout.ix[:,3:])
    X_live, y_live = prep_matrix(live.ix[:,3:])

    eras = data['era'].tolist()
    splits = split_by_era(eras, n_splits)
    stack = base_learner()
    stacks_df, holdout_df, live_df = stack.stacking(X, y, X_holdout, y_holdout, X_live, y_live, splits)
    gbm = top_learner(stacks_df)
    X_test, y_test = prep_matrix(holdout_df)
    cv_predictions = gbm.predict_proba(X_test)
    l_loss = log_loss(y_test, cv_predictions)

    print l_loss

    holdout_df.to_csv('output/holdout_df.csv')
    live_df.to_csv('output/live_df.csv')
    stacks_df.to_csv('output/stacks_df.csv')

    cv = [gbm, cv_predictions, l_loss]
    f = open('output/cv.pickle', 'w')
    pickle.dump(cv, f)
    f.close()

    print "finished"