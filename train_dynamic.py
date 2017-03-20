import pandas as pd
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from datetime import datetime
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB



def convertDate(date):
    return datetime.datetime.strptime(date, "%Y/%m/%d").strftime("%Y-%m-%d")

def buySell(open_close):
    if open_close[1] > open_close[0]:
        return 1
    return 0

def diffVal(priceList, weight = 1):
    rtrnList = []
    for itr,today in enumerate(priceList):
        if itr != 0 and today != priceList[itr-1]:
            yesterday = priceList[itr-1]
            rtrnList.append(float((today - yesterday)/ yesterday) * weight)
        else:
            rtrnList.append(0.0)
    return rtrnList

def buildLabel(closeList,openList):
    action_list = []
    for itr,val in enumerate(closeList):
        if itr != len(closeList)-1:
            # action_list.append(int(((openList[itr]-closeList[itr])/closeList[itr])*1000))##percent increase
            if closeList[itr] < closeList[itr+1]:##Tomorrow's close above Today's Close
                action_list.append(1)##Buy
            else:
                action_list.append(0)##Sell
        else:
            action_list.append(0)
    return action_list

def calTotalGain(total, pred, open, close):
    agg_total = total
    for itr,value in enumerate(pred):
        if itr < len(pred)-1:
            if value == 0:
                diff = close[itr+1]-close[itr]
                agg_total = (agg_total + diff)
    return (((agg_total-total) / total) *100)

def trainModel(exxon_data, currency_data, brent_data):
    'Format Date'
    exxon_data['date'] = exxon_data['Date']

    ##Compares Today's close vs. yesterday's
    diff_val = diffVal(list(exxon_data['Close']))
    df = pd.DataFrame(diff_val)
    exxon_data['exxon_price_df'] = df
    exxon_data['exxon_close'] = exxon_data['Close']
    exxon_data['exxon_open'] = exxon_data['Open']

    'Format Currency'
    currency_data['value'].replace('.', np.nan, inplace=True)
    currency_data.dropna(subset=['value'], inplace=True)
    currency_data['value'] = currency_data['value'].apply(np.float)

    diff_val = diffVal(list(currency_data['value']), weight =100)
    df = pd.DataFrame(diff_val).apply(lambda x: x)
    currency_data['DEXUSEU'] = df

    'Format Brent'
    brent_data['value'].replace('.', np.nan, inplace=True)
    brent_data.dropna(subset=['value'], inplace=True)
    brent_data['value'] = brent_data['value'].apply(np.float)

    diff_val = diffVal(list(brent_data['value']), weight = 100)
    df = pd.DataFrame(diff_val).apply(lambda x: x)
    brent_data['DCOILBRENTEU'] = df

    'Add Frames Together'
    frames = [currency_data,exxon_data,brent_data]
    merge = pd.merge(currency_data, exxon_data, on='date')
    new_merge = pd.merge(merge,brent_data,on='date')

    'Build Label'
    labels = buildLabel(list(new_merge['exxon_close']),list(new_merge['exxon_open']))
    df = pd.DataFrame(labels)
    new_merge['action'] = df

    'Drop Others'
    clean_data = new_merge[['date','action','DCOILBRENTEU','DEXUSEU','exxon_price_df','exxon_open','exxon_close']]
    print clean_data.head(20)
    clean_data = clean_data.fillna(0)
    'Split into Training Testing'
    train ,test = train_test_split(clean_data,test_size=0.4)
    labels_train = train['action'].values
    features_train = train.drop(['action','date','exxon_open','exxon_close'],axis=1)
    labels_test = test['action'].values
    features_test = test.drop(['date','action','exxon_open','exxon_close'],axis=1)
    print ('Fitting over set of ', len(train))
    'Predict Some Stuff'
    # clf = GaussianNB()
    clf = tree.DecisionTreeClassifier()

    print('fitting')
    clf = clf.fit(features_train, labels_train)
    print('data has been fit')
    pred = clf.predict(features_test)

    print 'it has been predicted'

    total_before_drop = clean_data[(clean_data['date'] > '2016-01-01') & (clean_data['date'] < '2017-01-01')]
    total = total_before_drop.drop(['date','action','exxon_open','exxon_close'],axis=1)

    my_pred = clf.predict(total)
    print calTotalGain(100, my_pred, total_before_drop['exxon_open'].values, total_before_drop['exxon_close'].values)


    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    print(acc,' :accuracy-score')
    return clf
