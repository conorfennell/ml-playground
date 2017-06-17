"""semo service"""
from __future__ import absolute_import
from urllib.request import urlopen
import json
from utils.utils import END_OF_TIME, INTERVAL_SECONDS

REF_CRC = "USDT"

def get_currency_pair(currency, ref_currency = REF_CRC):
    return f"{ref_currency}_{currency}"

def retrieve(currency, from_seconds, to_seconds = END_OF_TIME, interval = INTERVAL_SECONDS, ref_currency = REF_CRC):
  currency_pair = get_currency_pair(currency, ref_currency)
  url_string = f"https://poloniex.com/public?command=returnChartData&currencyPair={currency_pair}&start={from_seconds}&end={to_seconds}&period={interval}"
  url = urlopen(url_string).read()
  candles = json.loads(url)
  
  return candles

def default_retrieve():
  currency = 'ETH'
  from_seconds = 1453838000 #Tuesday 26th January 2016 07:53:20 PM
  candles = retrieve (currency, from_seconds)
  averages = list(map(lambda candle: { 'price': candle['weightedAverage']}, candles))


