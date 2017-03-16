import pandas as pd
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from sklearn.cross_validation import train_test_split
API_KEY = '39428be872fac30f87b247d8c92e56af'

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
            if openList[itr] < closeList[itr]:##Tomorrow's close above Today's Close
                action_list.append(1)##Buy
            else:
                action_list.append(0)##Sell
        else:
            action_list.append(0)
    return action_list

brent = './Brent.csv'
us_euro = './US_EURO.csv'
exxon = './XOM.csv'

brent_data = pd.read_csv(brent)
currency_data = pd.read_csv(us_euro)
exxon_data = pd.read_csv(exxon)

'Format Exxon'
exxon_data['date'] = exxon_data['date'].apply(convertDate)

diff_val = diffVal(list(exxon_data['volume']))
df = pd.DataFrame(diff_val)
exxon_data['volume'] = df

diff_val = diffVal(list(exxon_data['open']))
df = pd.DataFrame(diff_val)
exxon_data['price'] = df

'Format Currency'
currency_data['DEXUSEU'].replace('.', np.nan, inplace=True)
currency_data.dropna(subset=['DEXUSEU'], inplace=True)
currency_data['DEXUSEU'] = currency_data['DEXUSEU'].apply(np.float)

diff_val = diffVal(list(currency_data['DEXUSEU']))
df = pd.DataFrame(diff_val).apply(lambda x: x*100)
currency_data['DEXUSEU'] = df

'Format Brent'
brent_data['DCOILBRENTEU'].replace('.', np.nan, inplace=True)
brent_data.dropna(subset=['DCOILBRENTEU'], inplace=True)
brent_data['DCOILBRENTEU'] = brent_data['DCOILBRENTEU'].apply(np.float)

diff_val = diffVal(list(brent_data['DCOILBRENTEU']))
df = pd.DataFrame(diff_val).apply(lambda x: x*10)
brent_data['DCOILBRENTEU'] = df

'Add Frames Together'
frames = [currency_data,exxon_data,brent_data]
merge = pd.merge(currency_data, exxon_data, on='date')
new_merge = pd.merge(merge,brent_data,on='date')

'Build Label'
labels = buildLabel(list(new_merge['close']),list(new_merge['open']))
df = pd.DataFrame(labels)
new_merge['action'] = df

'Drop Others'
clean_data = new_merge[['date','action','volume','DCOILBRENTEU','DEXUSEU','price']]
clean_data = clean_data.fillna(0)

'Split into Training Testing'
train ,test = train_test_split(clean_data,test_size=0.3)
labels_train = train['action']
features_train = train.drop(['action','date'],axis=1)

labels_test = test['action']
features_test = test.drop(['date','action'],axis=1)

print clean_data.head(10)
print len(clean_data)
'Predict Some Stuff'
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
print('fitting')
clf = clf.fit(features_train, labels_train)
print('data has been fit')
pred = clf.predict(features_test)
test['prection'] = pred
print test.head(20)
print 'it has been predicted'


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print(acc,' :accuracy-score')
