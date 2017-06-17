"""
This script runs the forecast semo model
"""
from __future__ import absolute_import
from utils import utils
import polo_models
from service import kraken

polo_models.best_models_predict()

#polo_models.find_best_model()