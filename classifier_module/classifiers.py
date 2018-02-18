import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_fscore_support

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



# TODO: use different random states for each model in ensemble
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
            print "model is not fitted"

    def predict_proba(self, x):
        try:
            return self.clf.predict_proba(x)
        except:
            print "model not fitted"

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
# TODO: predict proba instead of actual predictions for base output
# DONE
# TODO: compare logloss against that of a random forest model
# DONE
# TODO: RBM feature engineering
# TODO: test whether base models are different enough
# DONE base model correlation low
# TODO: use different base models ... classification, cluster, D-reduction, regression
# TODO: determine optimal weights for base models
# TODO: plot learning curves and determine optimal cutoff
# TODO: convert binary response to a factor
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

    def stacking(self, X, y, X_holdout, y_holdout, splits):
        base_models = {'rf':self.rf, 'et':self.et, 'ada':self.ada, 'gb':self.gb}
        models_names = ['rf', 'et', 'ada', 'gb']
        train_meta = pd.DataFrame(data=None, columns=['rf', 'et', 'ada', 'gb'])
        # get the base predictions and make a meta feature matrix
        for name in models_names:
            print name
            model = base_models[name]
            X_stack = np.zeros((len(X)))
            for index in splits:
                print "split"
                X_train, X_test = X[index['train']], X[index['test']]
                y_train = y[index['train']]
                model.fit(X_train, y_train)
                probabilities = model.predict_proba(X_test)
                probability = [i[1] for i in probabilities]
                X_stack[index['test']] = probability
            train_meta[name] = list(X_stack)
        train_meta['labels'] = y
        # get the predictions on the holdout set
        holdout_meta = pd.DataFrame(data=None, columns=['rf', 'et', 'ada', 'gb'])
        for name in models_names:
            print name
            model = base_models[name]
            model.fit(X, y)
            probabilities = model.predict_proba(X_holdout)
            probability = [i[1] for i in probabilities]
            holdout_meta[name] = probability
        holdout_meta['labels'] = y_holdout
        return train_meta, holdout_meta


def top_learner(train_meta):
    X, y = prep_matrix(train_meta)
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


class single_learners():
    def __init__(self):
        self.rf_params = {
            'n_jobs': -1,
            'n_estimators': 2000,
             'warm_start': True,
             #'max_features': 0.2,
            'max_depth': 15,
            'max_features' : 'sqrt',
            'min_samples_leaf': 2,
            'verbose': 0
        }

        # Extra Trees Parameters
        self.et_params = {
            'n_jobs': -1,
            'n_estimators':500,
            #'max_features': 0.5,
            'max_depth': 6,
            'min_samples_leaf': 2,
            'verbose': 0
        }

        # AdaBoost parameters
        self.ada_params = {
            'n_estimators': 500,
            'learning_rate' : 0.75
        }

        # Gradient Boosting parameters
        self.gb_params = {
            'n_estimators': 500,
             #'max_features': 0.2,
            'max_depth': 5,
            'min_samples_leaf': 2,
            'verbose': 0
        }

        # Support Vector Classifier parameters
        self.svc_params = {
            'kernel' : 'linear',
            'C' : 0.025
            }

        SEED = 0 # for reproducibility
        self.rf = SKlearnHelper(clf=RandomForestClassifier, seed=SEED, params=self.rf_params)
        self.et = SKlearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=self.et_params)
        self.ada = SKlearnHelper(clf=AdaBoostClassifier, seed=SEED, params=self.ada_params)
        self.gb = SKlearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=self.gb_params)
        self.svc = SKlearnHelper(clf=SVC, seed=SEED, params=self.svc_params)

    def single_learner(self, X, y, X_holdout, y_holdout, model_name):
        base_models = {'rf': self.rf, 'et': self.et, 'ada': self.ada, 'gb': self.gb}
        model = base_models[model_name]
        model.fit(X, y)
        holdout_prediction_proba = model.predict_proba(X_holdout)
        holdout_prediction = model.predict(X_holdout)
        training_prediction_proba = model.predict_proba(X)
        training_prediction = model.predict(X)
        l_loss_holdout = log_loss(y_holdout, holdout_prediction_proba)
        l_loss_training = log_loss(y, training_prediction_proba)
        training_scores = precision_recall_fscore_support(y, training_prediction)
        holdout_scores = precision_recall_fscore_support(y_holdout, holdout_prediction)
        return l_loss_holdout, l_loss_training, holdout_prediction, training_scores, holdout_scores


if __name__ == '__main__':
    from pandas import read_csv
    import pickle

    n_splits = 10
    """
    DEBUGGING
    """
    # data_raw = read_csv('data/numerai_training_data.csv')
    # t_data = read_csv('data/numerai_tournament_data.csv')
    # holdout_raw = t_data[t_data['data_type']=='validation']
    # live_raw = t_data[t_data['data_type']=='live']
    # data = data_raw.sample(100)
    # holdout = holdout_raw.sample(100)
    # live = live_raw.sample(100)

    """
    RUNNING
    """
    data = read_csv('data/numerai_training_data.csv')
    t_data = read_csv('data/numerai_tournament_data.csv')
    holdout = t_data[t_data['data_type']=='validation']
    #live = t_data[t_data['data_type']=='live']

    X, y = prep_matrix(data.ix[:,3:])
    X_holdout, y_holdout = prep_matrix(holdout.ix[:,3:])
    #X_live, y_live = prep_matrix(live.ix[:,3:])
    #
    # """
    # multi-learners
    # """
    # eras = data['era'].tolist()
    # splits = split_by_era(eras, n_splits)
    # stack = base_learner()
    # train_meta, holdout_meta = stack.stacking(X, y, X_holdout, y_holdout, splits)
    # gbm = top_learner(train_meta)
    # X_holdout_meta, y_holdout_meta = prep_matrix(holdout_meta)
    # cv_predictions = gbm.predict_proba(X_holdout_meta)
    # l_loss = log_loss(y_holdout_meta, cv_predictions)
    #
    # print l_loss
    #
    # train_meta.to_csv('output/train_meta.csv')
    # holdout_meta.to_csv('output/holdout_meta.csv')
    #
    # cv = [gbm, cv_predictions, l_loss]
    # f = open('output/cv.pickle', 'w')
    # pickle.dump(cv, f)
    # f.close()
    #
    # print "finished"

    """
    single learner
    """
    for model_name in ['rf']:
        single_learner = single_learners()
        l_loss_holdout, l_loss_training, holdout_prediction, training_scores, holdout_scores = single_learner.single_learner(X, y, X_holdout, y_holdout, model_name)
        scores = [item for sublist in training_scores for item in sublist] + [item for sublist in holdout_scores for item in sublist]
        l_losses = [l_loss_training, l_loss_holdout]
        params = [single_learner.rf_params['n_estimators'], single_learner.rf_params['max_depth'],
                  single_learner.rf_params['max_features'], single_learner.rf_params['min_samples_leaf']]
        output_row = ['rf'] + params + l_losses + scores
        logger = open('output/single_rf_model.csv', 'a')
        logger.write(output_row)
        logger.close()

    #0.69258
    #0.69397


# count then print
def printer(n):
    if n==0:
        return
    else:
        print n
        n = n-1
        printer(n)


test = [1,2,1,2,1,4,2,2,3,4]
counts = [(i,test.count(i)) for i in test]

def printer(l):
    length = len(l)
    j = 0
    def _printer(_l,j,length):
        if j == length:
            return
        else:
            print str(_l[j][0]) + ':' + str(_l[j])[1]
            j += 1
            _printer(_l,j,length)
    _printer(l, j, length)
    return

# anagram checker
a = 'GOD'
b = 'DOG'


def pair_sum(arr, k):
    if len(arr) < 2:
        return

    # Sets for tracking
    seen = set()
    output = set()

    # For every number in array
    for num in arr:

        # Set target difference
        target = k - num
        print target

        # Add it to set if target hasn't been seen
        if target not in seen:
            print "not seen"
            seen.add(num)

        else:
            # Add a tuple with the corresponding pair
            output.add((min(num, target), max(num, target)))

    # FOR TESTING
    return output, seen

pair_sum([1,2,3,2,3,5,4,1],5)


def anagram_checker(a,b):
    if len(a)!=len(b):
        return False
    a_lst = [_a.lower for _a in list(a)]
    b_lst = [_b.lower for _b in list(b)]
    unique_letters = set(a_lst + b_lst)
    for l in unique_letters:
        if a_lst.count(l) != b_lst.count(l):
            return False
    return True


# output allp



# Python program to for tree traversals

# A class that represents an individual node in a
# Binary Tree
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key


# A function to do inorder tree traversal
def printInorder(root):
    if root:
        # First recur on left child
        printInorder(root.left)

        # then print the data of node
        print(root.val),

        # now recur on right child
        printInorder(root.right)


# A function to do postorder tree traversal
def printPostorder(root):
    if root:
        # First recur on left child
        printPostorder(root.left)

        # the recur on right child
        printPostorder(root.right)

        # now print the data of node
        print(root.val),


# A function to do postorder tree traversal
def printPreorder(root):
    if root:
        # First print the data of node
        print(root.val),

        # Then recur on left child
        printPreorder(root.left)

        # Finally recur on right child
        printPreorder(root.right)


# Driver code
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
print "Preorder traversal of binary tree is"
printPreorder(root)

print "\nInorder traversal of binary tree is"
printInorder(root)

print "\nPostorder traversal of binary tree is"
printPostorder(root)


