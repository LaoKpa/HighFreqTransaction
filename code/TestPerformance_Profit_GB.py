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
import profit as pf


train1 = pd.read_csv("/accounts/grad/xinyue233/Desktop/finance/data/train_set_midprice.csv")
test1 = pd.read_csv("/accounts/grad/xinyue233/Desktop/finance/data/test_set_midprice.csv")
train2 = pd.read_csv("/accounts/grad/xinyue233/Desktop/finance/data/train_set_spread.csv")
test2 = pd.read_csv("/accounts/grad/xinyue233/Desktop/finance/data/test_set_spread.csv")
predict_data = pd.read_csv("/accounts/grad/xinyue233/Desktop/finance/data/predict_data.csv")

########################################## Gradient Boosting ###########################################
################
### Midprice ###
################
learning_rate1  = 0.5
n_estimators1 = 500
max_depth1 = 10

train_data1,train_label1  = pf.get_lb_ft(train1,"Y_midprice")
clf1 = GradientBoostingClassifier(learning_rate = learning_rate1, n_estimators = n_estimators1,max_depth = max_depth1)

clf1.fit(train_data1,train_label1)

test_data1,test_label1 = pf.get_lb_ft(test1,"Y_midprice")
ret_test1 = clf1.predict(test_data1)
print("Test Report for Midprice")
print("------------------------------------------------")
print(metrics.accuracy_score(ret_test1, test_label1))
print(metrics.confusion_matrix(ret_test1, test_label1, labels = ["downward","stationary", "upward"]))
print(metrics.classification_report(ret_test1, test_label1))
print("------------------------------------------------")
print ("\n")

#######################
### Crossing Spread ###
#######################
learning_rate2  = 0.5
n_estimators2 =  500
max_depth2 = 10


train_data2,train_label2  = pf.get_lb_ft(train2,"Y_spread")
clf2 = GradientBoostingClassifier(learning_rate = learning_rate2, n_estimators = n_estimators2, max_depth = max_depth2)
clf2.fit(train_data2,train_label2)

test_data2,test_label2 = pf.get_lb_ft(test2,"Y_spread")
ret_test2 = clf2.predict(test_data2)
print("Test Report for Crossing Spread")
print("------------------------------------------------")
print(metrics.accuracy_score(ret_test2, test_label2))
print(metrics.confusion_matrix(ret_test2, test_label2, labels = ["downward","stationary", "upward"]))
print(metrics.classification_report(ret_test2, test_label2))
print("------------------------------------------------")
print("\n")

print("Profit Using Gradient Boosting")
print("-------------------------------")
print("Midprice:")
(profit_gb_total1, profit_gb_vector1), predit_gb_labels1 = pf.calcProfit(predict_data, step = 40, label = "Y_spread",clf = clf1, Thr_val = False)
print(profit_gb_total1)


print("-------------------------------")
print("Crossing Spread:")
(profit_gb_total2, profit_gb_vector2), predit_gb_labels2 = pf.calcProfit(predict_data, step = 100, label = "Y_spread",clf = clf2, Thr_val = False)
print(profit_gb_total2)

print("-------------------------------")