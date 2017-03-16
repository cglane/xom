import pandas as pd
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from sklearn.cross_validation import train_test_split
from datetime import datetime

def convertDate(date):
    return datetime.datetime.strptime(date, "%Y/%m/%d").strftime("%Y-%m-%d")
def buySell(open_close):
    if open_close[1] > open_close[0]:
        return 1
    return 0

def diffVal(priceList):
    rtrnList = []
    for itr,today in enumerate(priceList):
        if itr != 0 and today != priceList[itr-1]:
            yesterday = priceList[itr-1]
            rtrnList.append(float((today - yesterday)/ yesterday))
        else:
            rtrnList.append(0.0)
    return rtrnList

def buildLabel(openList,closeList):
    action_list = []
    for itr,val in enumerate(closeList):
        if itr != len(closeList)-1:
            if closeList[itr] < closeList[itr+1]:##Tomorrow's close above Today's Close
                action_list.append(1)##Buy
            else:
                action_list.append(0)##Sell
        else:
            action_list.append(0)
    return action_list


def trainModel(exxon_data, currency_data, brent_data):
    'Format Date'
    # exxon_data['date'] = exxon_data['Date'].apply(convertDate)
    exxon_data['date'] = exxon_data['Date']

    ##Compares Today's volume vs. yesterday's
    diff_val = diffVal(list(exxon_data['Volume']))
    df = pd.DataFrame(diff_val)
    exxon_data['exxon_volume_df'] = df

    ##Compares Today's open vs. yesterday's
    diff_val = diffVal(list(exxon_data['Open']))
    df = pd.DataFrame(diff_val)
    exxon_data['exxon_price_df'] = df
    exxon_data['exxon_close'] = exxon_data['Close']
    exxon_data['exxon_open'] = exxon_data['Open']

    'Format Currency'
    currency_data['value'].replace('.', np.nan, inplace=True)
    currency_data.dropna(subset=['value'], inplace=True)
    currency_data['value'] = currency_data['value'].apply(np.float)

    diff_val = diffVal(list(currency_data['value']))
    df = pd.DataFrame(diff_val).apply(lambda x: x)
    currency_data['DEXUSEU'] = df

    'Format Brent'
    brent_data['value'].replace('.', np.nan, inplace=True)
    brent_data.dropna(subset=['value'], inplace=True)
    brent_data['value'] = brent_data['value'].apply(np.float)

    diff_val = diffVal(list(brent_data['value']))
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
    clean_data = new_merge[['date','action','exxon_volume_df','DCOILBRENTEU','DEXUSEU','exxon_price_df']]
    clean_data = clean_data.fillna(0)
    'Split into Training Testing'
    train ,test = train_test_split(clean_data,test_size=0.3)
    labels_train = train['action']
    features_train = train.drop(['action','date','exxon_volume_df'],axis=1)
    labels_test = test['action']
    features_test = test.drop(['date','action','exxon_volume_df'],axis=1)

    print ('Fitting over set of ', len(train))
    'Predict Some Stuff'
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    print('fitting')
    clf = clf.fit(features_train, labels_train)
    print('data has been fit')
    pred = clf.predict(features_test)
    test['prediction'] = pred
    print 'it has been predicted'

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    print(acc,' :accuracy-score')
    return clf
