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
from model import predictor

from data_prep import polo

LOG = mylogger.get_logger(__name__)

# models = mods.get_linear_regression_models()
# models = mods.get_ridge_models()
# models = mods.get_lasso_models()
# models = mods.get_sgd_regressor()
# models = mods.get_lasso_lars()
# models = mods.get_bayesian_ridge()
# models = mods.get_svr_models()
# models = mods.get_mlp_models()
# models = mods.get_all()

def best_models_predict(): 
    is_plot_enabled = True
    shift_size = 12
    target_currency = 'ETH'
    input_currencies = ['BTC', 'LTC', 'ETC']
    from_seconds = 1483838000 #Tuesday 26th January 2016 07:53:20 PM

    x, y = polo.load_data(target_currency, input_currencies, shift_size, from_seconds)

    LOG.info('finished loading data')

    models = mods.get_best_models_so_far()
    
    predictor.predict_models(models, x, y, is_plot_enabled)

def find_best_model():
    is_plot_enabled = True
    shift_size = 12
    target_currency = 'ETH'
    input_currencies = ['BTC', 'LTC', 'ETC']
    from_seconds = 1483838000 #Tuesday 26th January 2016 07:53:20 PM

    x, y = polo.load_data(target_currency, input_currencies, shift_size, from_seconds)

    LOG.info('finished loading data')

    models = mods.get_best_models_so_far()
    
    scored_models = scorer.score_models(models, x, y, is_plot_enabled)
    
    sorted_scored_models = sorted(scored_models, key=lambda x: x[1])
    utils.write_json_to_file(sorted_scored_models, 'best_models_so_far.json')