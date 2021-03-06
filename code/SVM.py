import numpy as np
import pandas as pd
import scipy
import sklearn as sk
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm 
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import metrics
import pickle
import math
import profit as pf

######################### get label and feature from set ###############################
def get_lb_ft(df,label):
    labels = df[label].values
    featurelist  = list(set(df.columns.values) - set(["Y_midprice", "Y_spread"]))
    features = df[featurelist].values
    return features, labels



################ Script ##############
train1 = pd.read_csv("../data/train_set_midprice.csv")
test1 = pd.read_csv("../data/test_set_midprice.csv")
# tune parameter
param_grid = dict(C = np.arange(0.03,2,1/3.0), gamma = np.logspace(-5, 3, 7))
train_data,train_label  = get_lb_ft(train1,"Y_midprice")
grid = GridSearchCV(svm.SVC(), param_grid = param_grid, cv = 3, n_jobs = -1)
grid.fit(train_data, train_label)

print("The best parameters are %s with a score of %0.4f"  % (grid.best_params_, grid.best_score_))


# fit the best model
clf = svm.SVC(C = grid.best_params_['C'], gamma = grid.best_params_['gamma'])
clf.fit(train_data,train_label)

# On the train data
train_ret = clf.predict(train_data)
print("On the train data")
print(metrics.accuracy_score(train_ret, train_label))
print(metrics.confusion_matrix(train_ret, train_label))
print(metrics.classification_report(train_ret, train_label))

# On the test data
test_data,test_label = get_lb_ft(test1,"Y_midprice")
test_ret = clf.predict(test_data)

print("On the test data")
print(metrics.accuracy_score(test_ret, test_label))
print(metrics.confusion_matrix(test_ret, test_label))
print(metrics.classification_report(test_ret, test_label))
