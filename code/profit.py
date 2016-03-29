'''
Given a buy and sell strategy, calculate the profits base on the predicted labels and the profit based on the true labels.

'''
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

def get_lb_ft(df,label):
    labels = df[label].values
    featurelist  = list(set(df.columns.values) - set(["Y_midprice", "Y_spread"]))
    features = df[featurelist].values
    return features, labels

def calcProfit_help (profit_df, step, label ,Thr_val = False):
    # if upward
    new_label = "pre_label_" + label[2:]
    ask_t2 = pd.Series(0, index= range(profit_df.shape[0]))
    ask_t2[step:] = profit_df["ASK_PRICE1"][0:-step]
    long_profit = profit_df["BID_PRICE1"] - ask_t2
    long_profit.ix[0:step,] = 0

    # if downward
    bid_t2 = pd.Series(0, index= range(profit_df.shape[0]))
    bid_t2[step:] = profit_df["BID_PRICE1"][0:-step]
    short_profit = bid_t2 - profit_df["ASK_PRICE1"]
    short_profit.ix[0:step,] = 0

    profit_pred = pd.Series(0, index= range(profit_df.shape[0]))
    msk_up = (profit_df[new_label] == "upward")
    msk_down = (profit_df[new_label] == "downward")
    profit_pred[msk_up] = long_profit[msk_up]
    profit_pred[msk_down] = short_profit[msk_down]
    
    profit_true = pd.Series(0, index= range(profit_df.shape[0]))
    msk_up_true = (profit_df[label] == "upward")
    msk_down_true = (profit_df[label] == "downward")
    profit_true[msk_up_true] = long_profit[msk_up_true]
    profit_true[msk_down_true] = short_profit[msk_down_true]
    
    if (Thr_val):
        return profit_pred, sum(profit_pred),profit_true, sum(profit_true)
    #return profit_df["ASK_PRICE1"][0:-30]
    return sum(profit_pred), profit_pred


def calcProfit(predict_data,step,label,clf,Thr_val = False):
    '''
    Input
    ---------
    predict_data: predict frame including all features and labels
    step: the step we chose. defaulf for midprice is 30 and for spread is 100
    label: Y_midprice or Y_spread
    clf: the best classifier trained in each model
    Thr_val: Do you want to return the Theoretical value?

    return:
    ---------
    profit vector; sum of profits;(if Thr_val == true, profit vector; sum of profits in theory)
    labels from the prediction.
    '''
    predict_feature,predict_label  = get_lb_ft(predict_data,label)
    temp_pred = clf.predict(predict_feature)
    new_label = "pre_label_" + label[2:]
    my_label = pd.DataFrame({new_label:temp_pred})
    profit_df = pd.concat([predict_data, my_label],axis = 1).loc[:,["BID_PRICE1","ASK_PRICE1",label, new_label]]
    
    return calcProfit_help (profit_df, step, label ,Thr_val),temp_pred







