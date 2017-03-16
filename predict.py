from query_api import getStockHistory, fredValueToday, fredCategory
from train_dynamic import trainModel,diffVal
import pandas as pd
import json

# print fredValueToday('DCOILBRENTEU')
# print fredValueToday('DEXUSEU')
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
currency = [1.0700,1.07039]
volume = [11970000,436558]
price = [81.34,82.00]
brent = [52.13, 52.15]

predict_val = [diffVal(brent)[1],diffVal(currency)[1],diffVal(price)[1]]
# print predict_val
model = trainModel(exxon_data,currency_data,brent_data)
print model.predict(predict_val)
