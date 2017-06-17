"""semo service"""
from __future__ import absolute_import
from urllib.request import urlopen
import json
from datetime import datetime

def get_currency_pair (crc1, crc2):
  return f"Z{crc1}Z{crc2}"

def retrieve(source_currency, target_currency, since = "", interval = 1800):
  currency_pair = get_currency_pair(source_currency, target_currency)
  url_string = f" https://api.kraken.com/0/public/OHLC?pair={currency_pair}&since={since}&interval={interval}"
  url = urlopen(url_string).read()
  candles = json.loads(url)["result"]["currency_pair"]
  
  return candles



