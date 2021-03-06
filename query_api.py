import fred
from datetime import datetime
from yahoo_finance import Share
import pandas as pd
from lxml import html
import requests
API_KEY = '39428be872fac30f87b247d8c92e56af'
fred.key(API_KEY)

def getStockHistory(stock_symbol,start='2010-01-01'):
    end = datetime.now().strftime('%Y-%m-%d')
    yahoo = Share(stock_symbol)
    return yahoo.get_historical(start, end)

def getStockPriceNow(stock_symbol):
    yahoo = Share(stock_symbol)
    print yahoo.get_price()
    return float(yahoo.get_price())

def fredValueToday(arg):
    page = requests.get('https://fred.stlouisfed.org/series/'+arg)
    tree = html.fromstring(page.content)
    return float(tree.xpath('//span[@class="series-meta-observation-value"]/text()')[0])

def fredCategory(symbol):
    return fred.observations(symbol)['observations']

def bloombergCommodity(arg,label):
    page = requests.get('https://www.bloomberg.com/quote/'+arg)
    tree = html.fromstring(page.content)
    labels = tree.xpath('//div[@class="cell__label"]/text()')
    values = tree.xpath('//div[@class="cell__value cell__value_"]/text()')
    value_index = [x for itr,x in enumerate(values) if labels[itr] == label][0]
    return float(value_index)
