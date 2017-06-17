"""prepares data for training"""

from __future__ import absolute_import

from utils import mylogger
from service import polo
from utils import utils

LOG = mylogger.get_logger(__name__)

def load_data(target_currency, input_currencies, shift_size, from_seconds, to_seconds = utils.END_OF_TIME):
    full_data = list(map(lambda currency: polo.retrieve(currency, from_seconds, to_seconds), input_currencies))
    
    # featurise currency_pairs
    x = select_x(full_data)

    # get the output (which is the weighted average of the target currency)
    y = polo.retrieve(target_currency, from_seconds, to_seconds)
    y = select_y(y)

    return (x, y)

def select_x(currency_pairs):
    """format input data to a 1d array"""
    x_features = list(map(lambda candle: [], currency_pairs[0]))

    # {"date":1405699200,"high":0.0045388,"low":0.00403001,"open":0.00404545,"close":0.00427592,"volume":44.11655644,"quoteVolume":10259.29079097,"weightedAverage":0.00430015}
    for candles in currency_pairs:
      featuresIndex = 0
      for candle in candles:
        x = [candle['date'], candle['high'], candle['low'], candle['open'], candle['close'], candle['volume'], candle['quoteVolume']]
        x_features[featuresIndex].extend(x)
        featuresIndex = featuresIndex + 1

    return x_features


def select_y(candles):
    """format input data to one value"""
    return [candle['weightedAverage'] for candle in candles]
