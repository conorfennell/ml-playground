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

def shift_predict (model_details, x, y, shift_size, interval, is_plot_enabled):
    # the shift
    # we have 
    # x = 1 2 3 4 5 6 7 8 9
    # y = 1 2 3 4 5 6 7 8 9
    # for a size of 3 the shift is:
    # train x = 1 2 3 4 5 6
    # train y = 4 5 6 7 8 9 
    # test x = 7 8 9
    # we also increase the time field for the whole of x since 
    # we shifted it into the future 
    # the time increase does not really matter since it is a linear transformation 
    # but it will be nice for when we extract the date

    model_name, model = model_details

    for item in x:
        item[0] = item[0] + shift_size * interval
    
    x_train = x[:-shift_size]
    y_train = y[shift_size:]

    x_test = x[-shift_size:]

    scaler = StandardScaler().fit(x)
    train_scaled_x = scaler.transform(x_train)
    test_scaled_x = scaler.transform(x_test)
    
    model.fit(train_scaled_x, y_train)
    
    predicted_y = model.predict(test_scaled_x)
    
    if is_plot_enabled:
        show_plot(x_train, y_train, x_test, predicted_y)

    return predicted_y


def predict_models(models, x_features, y_features, is_plot_enabled):
    """runs all the models with x and y features, returning all the models with their score """
    scored_models = []
    max_score = 0
    max_model_name = ''

    for model in models:
        shift_predict (model, x_features, y_features, 24, 1800, is_plot_enabled)

def show_plot(x_train, y_train, x_test, y_predicted):
    index = 0
  
    train_range = []
    for entry in y_train :
        train_range.append(index)
        index = index + 1
  
    test_range = []
    for entry in x_test :
        test_range.append(index)
        index = index + 1

    plt.plot(train_range, y_train, color='green')
    plt.plot(test_range, y_predicted, color='red')
    plt.show()
