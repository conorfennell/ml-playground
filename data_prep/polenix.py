"""prepares data for training"""

from __future__ import absolute_import

from utils import mylogger
from service import polenix

LOG = mylogger.get_logger(__name__)

def load_data(currency_pair_target, currency_pairs, shift_size, from_seconds, to_seconds):
    # featureize currency_pairs
    x = list(map(lambda currency_pair: polenix.retrieve(currency_pair, from_seconds, to_seconds), currency_pairs))
    x_features = select_features_x(x)

    # featureise currency_pair_target
    y = polenix.retrieve(currency_pair_target, from_seconds, to_seconds)
    y_features = select_features_y(y)

    # shift
    x_features = x_features[shift_size:]
    y_features = y_features[:-shift_size]
   
    return (x_features, y_features)


def select_features_x(currency_pairs):
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


def select_features_y(candles):
    """format input data to one value"""
    return [candle['weightedAverage'] for candle in candles]
