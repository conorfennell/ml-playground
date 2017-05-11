"""semo service"""
from __future__ import absolute_import
from urllib.request import urlopen
import json
from datetime import datetime

epoch = datetime.utcfromtimestamp(0)

def unix_time_seconds(dt):
    return (dt - epoch).total_seconds()

def retrieve(currency_pair, from_seconds, to_seconds):
  url_string = f"https://poloniex.com/public?command=returnChartData&currencyPair={currency_pair}&start={from_seconds}&end={to_seconds}&period=1800"
  url = urlopen(url_string).read()
  candles = json.loads(url)
  
  return candles

def default_retrieve():
  currency_pair = 'USDT_ETH'
  from_seconds = 1483838000
  from_seconds = 1453838000
  to_seconds = 9999999999
  url_string = f"https://poloniex.com/public?command=returnChartData&currencyPair={currency_pair}&start={from_seconds}&end={to_seconds}&period=900"
  url = urlopen(url_string).read()
  candles = json.loads(url)
  averages = list(map(lambda candle: { 'price': candle['weightedAverage']}, candles))


