'''
Given a buy and sell strategy, calculate the profits base on the predicted labels and the profit based on the true labels.

'''

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
    return profit_pred, sum(profit_pred)


def calcProfit(predict_data,step,label,clf,Thr_val = False):
    
    predict_feature,predict_label  = get_lb_ft(predict_data,label)
    temp_pred = clf.predict(predict_feature)
    new_label = "pre_label_" + label[2:]
    my_label = pd.DataFrame({new_label:temp_pred})
    profit_df = pd.concat([predict_sub, my_label],axis = 1).loc[:,["BID_PRICE1","ASK_PRICE1",label, new_label]]
    
    return calcProfit_help (profit_df, step, label ,Thr_val)







