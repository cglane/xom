from query_api import getStockHistory, fredValueToday, fredCategory,getStockPriceNow
from train_dynamic import trainModel,diffVal
import pandas as pd
import json

def mostRecentRow(df,column,focus):
    row = df.loc[df[column]== df[column].max()]
    print ('Last Date',row[column].values[0])
    print ('Value',row[focus].values[0])
    return float(row[focus].values[0])


# BRENT =  pd.DataFrame.from_dict(fredCategory('DCOILBRENTEU'))
# BRENT.to_csv('./dynamic_files/BRENT.csv')
# US_EURO = pd.DataFrame.from_dict(fredCategory('DEXUSEU'))
# US_EURO.to_csv('./dynamic_files/US_EURO.csv')
# XOM = pd.DataFrame.from_dict(getStockHistory('XOM'))
# XOM.to_csv('./dynamic_files/XOM.csv')
#
brent = './dynamic_files/Brent.csv'
us_euro = './dynamic_files/US_EURO.csv'
exxon = './dynamic_files/XOM.csv'
#
brent_data = pd.read_csv(brent)
currency_data = pd.read_csv(us_euro)
exxon_data = pd.read_csv(exxon)
#
'exxon_volume_df','DCOILBRENTEU','DEXUSEU'


'exxon predict'
exxon_open_yesterday = mostRecentRow(exxon_data,column='Date', focus="Open")
exxon_price_today = getStockPriceNow('XOM')
print (exxon_price_today, 'exxon price today')
price = [exxon_open_yesterday,exxon_price_today]
print price

'currency predict'
currency_close_yesterday = mostRecentRow(currency_data,column='date', focus="value")
currency_price_today = getStockPriceNow('EURUSD=X')
print (currency_price_today, 'currency price today')
currency = [currency_close_yesterday,currency_price_today]
print currency

'brent predict'

brent_close_yesterday = mostRecentRow(brent_data,column='date', focus="value")
brent_price_today = getStockPriceNow('BZ=F')
print (brent_price_today, 'brent price today')
brent = [brent_close_yesterday,brent_price_today]
print brent

'predict array'
# 'DCOILBRENTEU','DEXUSEU','exxon_price_df'
brent_val = diffVal(brent, weight = 100)[1]
print (brent_val, 'brent_val')
currency_val =  diffVal(currency, weight = 100)[1]
price_val = diffVal(price)[1]

predict_array = [brent_val,currency_val,price_val]
model = trainModel(exxon_data,currency_data,brent_data)

today_prediction = model.predict(predict_array)
if today_prediction == 1:
    print 'Buy'
else:
    print 'Sell'
