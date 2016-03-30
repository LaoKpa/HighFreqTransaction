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


train1 = pd.read_csv("../data/train_set_midprice.csv")
test1 = pd.read_csv("../data/test_set_midprice.csv")
train2 = pd.read_csv("../data/train_set_spread.csv")
test2 = pd.read_csv("../data/test_set_spread.csv")
predict_data = pd.read_csv("../data/predict_data.csv")

########################################## SVM ###########################################
################
### Midprice ###
################
C = 1
gamma = 0.00001
train_data1,train_label1  = pf.get_lb_ft(train1,"Y_midprice")
clf1 = svm.SVC(C = C, gamma = gamma)
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
C = 1
gamma = 0.00001

train_data2,train_label2  = pf.get_lb_ft(train2,"Y_spread")
clf2 = svm.SVC(C = C, gamma = gamma)
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

print("Profit Using SVM")
print("-------------------------------")
print("Midprice:")
(profit_svm_total1, profit_svm_vector1), predit_svm_labels1 = pf.calcProfit(predict_data, step = 40, label = "Y_spread",clf = clf1, Thr_val = False)
print(profit_svm_total1)


print("-------------------------------")
print("Crossing Spread:")
(profit_svm_total2, profit_svm_vector2), predit_svm_labels2 = pf.calcProfit(predict_data, step = 100, label = "Y_spread",clf = clf2, Thr_val = False)
print(profit_svm_total2)

print("-------------------------------")



# Test Report for Midprice
# ------------------------------------------------
# 0.69896
# [[19215   809  7626]
#  [  421  1018   342]
#  [ 5278   576 14715]]
#              precision    recall  f1-score   support

#    downward       0.77      0.69      0.73     27650
#  stationary       0.42      0.57      0.49      1781
#      upward       0.65      0.72      0.68     20569

# avg / total       0.71      0.70      0.70     50000

# ------------------------------------------------


# Test Report for Crossing Spread
# ------------------------------------------------
# 0.95494
# [[ 1184   720     0]
#  [  511 45375   421]
#  [    1   600  1188]]
#              precision    recall  f1-score   support

#    downward       0.70      0.62      0.66      1904
#  stationary       0.97      0.98      0.98     46307
#      upward       0.74      0.66      0.70      1789

# avg / total       0.95      0.95      0.95     50000

# ------------------------------------------------


# Profit Using SVM
# -------------------------------
# Midprice:
# -24049.33
# -------------------------------
# Crossing Spread:
# -337.85
# -------------------------------





