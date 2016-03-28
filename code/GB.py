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
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import math

######################### get label and feature from set ###############################
def get_lb_ft(df,label):
    labels = df[label].values
    featurelist  = list(set(df.columns.values) - set(["Y_midprice", "Y_spread"]))
    features = df[featurelist].values
    return features, labels



################ Script ##############
#load data
train1 = pd.read_csv("../data/train_set_midprice.csv")
test1 = pd.read_csv("../data/test_set_midprice.csv")
predict_data = pd.read_csv("../data/predict_data.csv")


########## tune the para ##########
# learning_rate 
lambdas = [0.0001,0.001,0.01,0.1,1]
# n_estimators
ntree_list = [50, 100, 250, 500]
# max_depth 
depth = [10, 25, 50]

param_grid = dict(learning_rate  = lambdas, n_estimators = ntree_list, max_depth = depth)
# param_grid = dict(learning_rate  = lambdas, n_estimators = ntree_list, max_depth = depth)
train_data,train_label  = get_lb_ft(train1,"Y_midprice")
cv = StratifiedKFold(labels, n_folds = 3, random_state = 20151204,shuffle = TRUE)
grid = GridSearchCV(GradientBoostingClassifier(), param_grid = param_grid, cv = cv, n_jobs = -1)
grid.fit(train_data, train_label)


print(grid.grid_scores_)
print("The best parameters are %s with a score of %0.4f"  % (grid.best_params_, grid.best_score_))


# fit the best model
clf = GradientBoostingClassifier(
    learning_rate = grid.best_params_['learning_rate'], 
    n_estimators = grid.best_params_['n_estimators'],
    max_depth = grid.best_params_['max_depth'])

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