"""
load and run models
"""
from __future__ import absolute_import
from datetime import datetime
import matplotlib.pyplot as plt
from utils import mylogger
from utils import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import models as mods
from model import scorer

from data_prep import polenix

LOG = mylogger.get_logger(__name__)

def find_best_model():
    is_plot_enabled = True
    shift_size = 12
    currency_pair_target = 'USDT_ETH'
    currency_pairs = ['USDT_BTC', 'USDT_LTC', 'USDT_ETC']
    from_seconds = 1483838000
    #from_seconds = 1453838000
    to_seconds = 1494536581
    x_features, y_features = polenix.load_data(currency_pair_target, currency_pairs, shift_size, from_seconds, to_seconds)

    print(len(x_features))
    print(len(y_features))

    LOG.info('finished loading data')
    models = mods.get_best_models_so_far()
    # models = mods.get_linear_regression_models()
    # models = mods.get_ridge_models()
    # models = mods.get_lasso_models()
    # models = mods.get_sgd_regressor()
    # models = mods.get_lasso_lars()
    # models = mods.get_bayesian_ridge()
    # models = mods.get_svr_models()
    # models = mods.get_mlp_models()
    # models = mods.get_all()
    scored_models = scorer.score_models(models, x_features, y_features, is_plot_enabled)
    sorted_scored_models = sorted(scored_models, key=lambda x: x[1])
    utils.write_json_to_file(sorted_scored_models, '/logs/best_models_so_far.json')