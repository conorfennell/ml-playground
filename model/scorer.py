"""
score models
"""
from __future__ import absolute_import
from datetime import datetime
import matplotlib.pyplot as plt
from utils import mylogger
from utils import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import models as mods

LOG = mylogger.get_logger(__name__)

def score_model(model_details, x_features, y_features, is_plot_enabled):
    """ splits trainging data, scales data, trains models and returns a score """
    model_name, model = model_details

    # Split training and test data
    len_train = int(0.8 * len(x_features))
    x_train = x_features[0:len_train]
    y_train = y_features[0:len_train]
    x_test = x_features[len_train:]
    y_test = y_features[len_train:]
    # Scale training data
    scaler = StandardScaler().fit(x_features)
    train_scaled_x = scaler.transform(x_train)
    test_scaled_x = scaler.transform(x_test)
    # train the model
    model.fit(train_scaled_x, y_train)
    # score the model
    predicted_y = model.predict(test_scaled_x)
    score = model.score(test_scaled_x, y_test)

    LOG.info('{0} model score: {1}'.format(model_name, score))
    if is_plot_enabled:
        show_plot(y_train, y_test, predicted_y)

    return score


def score_models(models, x_features, y_features, is_plot_enabled):
    """runs all the models with x and y features, returning all the models with their score """
    scored_models = []
    max_score = 0
    max_model_name = ''

    for model in models:
        score = score_model(model, x_features, y_features, is_plot_enabled)
        scored_models.append((model[0], score))
        if score > max_score:
            max_score = score
            max_model_name = model[0]
            LOG.info('-------------------------------------------new max')
    LOG.info('\n{0} best model score: {1}'.format(max_model_name, max_score))
    return scored_models

def show_plot(y_train, y_test, y_predicted):
    train_range = []
    index = 0
    for entry in y_train :
        train_range.append(index)
        index = index + 1
    test_range = []
    for entry in y_test :
        test_range.append(index)
        index = index + 1
    plt.plot(train_range, y_train, color='green')
    plt.plot(test_range, y_test, color='green')
    plt.plot(test_range, y_predicted, color = 'red')
    plt.show()

def show_plot_future(y_train, y_test, y_predicted, future_predicted_y):
    train_range = []
    index = 0
    for entry in y_train :
        train_range.append(index)
        index = index + 1
    test_range = []
    for entry in y_test :
        test_range.append(index)
        index = index + 1
    future_range = []
    for entry in future_predicted_y :
        future_range.append(index)
        index = index + 1
    plt.plot(train_range, y_train, color='green')
    plt.plot(test_range, y_test, color='green')
    plt.plot(test_range, y_predicted, color = 'red')
    plt.plot(future_range, future_predicted_y, color = 'blue')
    plt.show()