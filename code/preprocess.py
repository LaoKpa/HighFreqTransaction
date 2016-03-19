# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 22:00:29 2016

@author: haolyu
"""
import numpy as np
import pandas as pd
#data1 = pd.read_csv("../data/AAPL_05222012_0930_1300_LOB_2.csv")
data1 = pd.read_csv("AAPL_05222012_0930_1300_LOB_2.csv")
# column name
col1 = list(data1.columns.values)[2:62]
len(col1)
col2 = [i[16:] for i in col1]
new_names = list(data1.columns.values)[0:2]+ [i[:-19] for i in col2]
data1.columns = new_names

# delete times
data1 = data1.head(203350)
data1.shape
data1.head(10)

#create desired names 
level  = range(1,11)
"BID_PRICE"+str(level[1])
bidpc_name = ["BID_PRICE"+str(level[i]) for i in range(10)]
askpc_name = ["ASK_PRICE"+str(level[i]) for i in range(10)]
bidsz_name = ["BID_SIZE"+str(level[i]) for i in range(10)]
asksz_name = ["ASK_SIZE"+str(level[i]) for i in range(10)]

#define feature4
def feature4 (data,level) :
    askpc_mean=np.sum(data1.ix[:,askpc_name],axis=1)/level
    bidpc_mean=np.sum(data1.ix[:,bidpc_name],axis=1)/level
    asksz_mean=np.sum(data1.ix[:,asksz_name],axis=1)/level
    bidsz_mean=np.sum(data1.ix[:,bidsz_name],axis=1)/level
    askpc_mean=askpc_mean.tolist()
    bidpc_mean=bidpc_mean.tolist()
    asksz_mean=asksz_mean.tolist()
    bidsz_mean=bidsz_mean.tolist() 
    v4 = pd.DataFrame({"Mean Ask Price":askpc_mean,"Mean Bid Price":bidpc_mean,
                  "Mean Ask Size":asksz_mean,"Mean Bid Size":bidsz_mean})
    return v4
v4 = feature4(data1,10)
v4.head(10)

#define feature5
def feature5 (data):
    pc_diff = np.sum(data1.ix[:,askpc_name],axis=1)- np.sum(data1.ix[:,bidpc_name],axis=1)
    sz_diff = np.sum(data1.ix[:,asksz_name],axis=1)- np.sum(data1.ix[:,bidsz_name],axis=1)
    pc_diff = pc_diff.tolist()
    sz_diff = sz_diff.tolist()
    v5 = pd.DataFrame({"Acc. Price Diff":pc_diff,"Acc. Size Diff":sz_diff})
    return v5;
v5 = feature5(data1)
v5.head(10)

#define ratio
def volumn_ratio (data):
    v_ratio=np.sum(data1.ix[:,asksz_name[0:4]],axis=1)/np.sum(data1.ix[:,bidsz_name[0:4]],axis=1)
    v_ratio=v_ratio.tolist()
    volumn_ratio = pd.DataFrame({"top five volumn ratio":v_ratio})
    return volumn_ratio;
volumn_ratio(data1).head(10)

#define feature3
bidlist = range(2,62,6)
asklist = range(5,65,6)
def v3feature(data):
    bid = data.ix[:,bidlist]
    bid_feature = abs(bid.diff(axis = 1).ix[:,range(1,10)])
    names = ["bid"+str(i+1)+"-"+"bid"+str(i) for i in range(1,10)]
    bid_feature.columns = names
    bid_feature2 = pd.DataFrame(bid["BID_PRICE1"] - bid["BID_PRICE10"])
    bid_feature2.columns = ["bid1-bid10"]
    bid_feature = pd.concat([bid_feature,bid_feature2],axis=1)
    ask = data.ix[:,asklist]
    ask_feature = abs(ask.diff(axis = 1).ix[:,range(1,10)])
    names = ["ask"+str(i+1)+"-"+"ask"+str(i) for i in range(1,10)]
    ask_feature.columns = names
    ask_feature2 = pd.DataFrame(ask["ASK_PRICE1"] - ask["ASK_PRICE10"])
    ask_feature2.columns = ["ask1-ask10"]
    ask_feature = pd.concat([ask_feature,ask_feature2],axis=1)
    v3 = pd.concat([ask_feature,bid_feature],axis=1)
    return v3;

v3 = v3feature(data1)
v3.shape

#define feature 6
volbidlist = range(4,64,6)
volasklist = range(7,67,6)
def v6feature(data,delta = 10):
    v6_1 = pd.concat([data.ix[:,bidlist],data.ix[:,asklist]],axis=1).diff(periods = delta, axis = 0).ix[delta:,:]/delta
    v6_2 = pd.concat([data.ix[:,volbidlist],data.ix[:,volasklist]],axis=1).diff(periods = delta, axis = 0).ix[delta:,:]/delta
    v6 = pd.concat([v6_1,v6_2],axis = 1)
    return v6;
v6= v6feature(data1)
v6