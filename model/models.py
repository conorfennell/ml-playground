"""different testing models"""

from __future__ import absolute_import
import numpy as np
from sklearn import svm
from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge, LassoLars, SGDRegressor
from sklearn.neural_network import MLPRegressor


def get_linear_regression_models():
    return [('LinearRegression', LinearRegression())]

def get_ridge_models():
    models = []

    for alpha in np.arange(0.001, 10, 0.001):
        models.append((f'Ridge_alpha:{alpha}|fit_intercept:True|normalize:False|copy_X:True|max_iter:None|tol:0.001|solver:auto|random_state:None',
                       Ridge(alpha=alpha, 
                             fit_intercept=True, 
                             normalize=False, 
                             copy_X=True, 
                             max_iter=None, 
                             tol=0.001, 
                             solver='auto', 
                             random_state=None)))
    return models

def get_lasso_models():
    models = []
    models.append(('Lasso', 
                   Lasso(alpha=0.1, 
                         copy_X=True, 
                         fit_intercept=True,
                         max_iter=1000,
                         normalize=False,
                         positive=False,
                         precompute=False,
                         random_state=None,
                         selection='cyclic',
                         tol=0.0001,
                         warm_start=False)))
    return models

def get_sgd_regressor():
    models = []
    models.append(('SGDRegressor',
                   SGDRegressor(alpha=0.0001,
                                average=False,
                                epsilon=0.1,
                                eta0=0.01,
                                fit_intercept=True,
                                l1_ratio=0.15,
                                learning_rate='invscaling',
                                loss='squared_loss',
                                n_iter=5,
                                penalty='l2',
                                power_t=0.25,
                                random_state=None,
                                shuffle=True,
                                verbose=0,
                                warm_start=False)))
    return models

def get_lasso_lars():
    models = []
    models.append(('LassoLars',
                   LassoLars(alpha=0.1,
                             copy_X=True,
                             fit_intercept=True,
                             fit_path=True,
                             max_iter=500,
                             normalize=True,
                             positive=False,
                             precompute='auto',
                             verbose=False)))
    return models

def get_bayesian_ridge():
    models = []
    models.append(('BayesianRidge',
                   BayesianRidge(alpha_1=1e-06,
                                 alpha_2=1e-06,
                                 compute_score=False,
                                 copy_X=True,
                                 fit_intercept=True,
                                 lambda_1=1e-06,
                                 lambda_2=1e-06,
                                 n_iter=300,
                                 normalize=False,
                                 tol=0.001,
                                 verbose=False)))
    return models

def get_svr_models():
    models = []
    deg = 3 # has no effect on the 'rbf' kernal
    c = 136
    cache = 200 # has no visible effect when values were changed
    coef0 = 0.0 # has no effect on the 'rbf' kernal
    max_i = -1 # -1 no cap on iterations
    for eps in np.arange(0.5, 0.7, 0.01):
        for c in range(1, 150, 1):
            models.append((f'svr_degree:{deg}|C:{c}|cache_size:{cache}|coef0:{coef0}|epsilon:{eps}',
                        svm.SVR(kernel='rbf',
                                degree=deg,
                                C=c,
                                cache_size=cache,
                                coef0=coef0,
                                epsilon=eps,
                                gamma='auto',
                                max_iter=max_i,
                                shrinking=True,
                                tol=0.0002,
                                verbose=False)))
    return models

def get_mlp_models():
    models = []
    sizes = []
    for size1 in range(10,100,10):
        sizes.append((size1))
        for size2 in range(10,100,10):
            sizes.append((size1, size2))

    for activation in ['identity', 'logistic']: 
        for solver in ['lbfgs', 'sgd', 'adam']:
            for learningrate in ['constant', 'invscaling', 'adaptive']:
                for layersizes in sizes:
                    alpha = 0.0001              
                    max_iter = 200
                    tol = 0.0001
                    models.append((f'mlp_hidden_layer_sizes:{layersizes}|activation:{activation}|solver:{solver}|alpha:{alpha}|learning_rate:{learningrate}|max_iter:{max_iter}|tol:{tol}',
                                   MLPRegressor(hidden_layer_sizes=layersizes,
                                                activation=activation,
                                                solver=solver,
                                                alpha=alpha,
                                                batch_size='auto',
                                                learning_rate=learningrate,
                                                learning_rate_init=0.001,
                                                power_t=0.5,
                                                max_iter=max_iter,
                                                shuffle=True,
                                                random_state=None,
                                                tol=tol,
                                                verbose=False,
                                                warm_start=False,
                                                momentum=0.9,
                                                nesterovs_momentum=True,
                                                early_stopping=False,
                                                validation_fraction=0.1,
                                                beta_1=0.9,
                                                beta_2=0.999,
                                                epsilon=1e-08)))
    
    for alpha in np.arange(0.0001, 0.001, 0.0001):
        for max_iter in range(200, 300, 20):
            for tol in np.arange(0.00001, 0.0001, 0.00001):
                
                models.append((f'mlp_hidden_layer_sizes:(3)|activation:logistic|solver:lbfgs|alpha:{alpha}|learning_rate:constant|max_iter:{max_iter}|tol:{tol}',
                               MLPRegressor(hidden_layer_sizes=(3), activation='logistic', solver='lbfgs', alpha=alpha,  learning_rate='constant',  max_iter=max_iter, tol=tol)))
                models.append((f'mlp_hidden_layer_sizes:(3,50)|activation:logistic|solver:lbfgs|alpha:{alpha}|learning_rate:adaptive|max_iter:{max_iter}|tol:{tol}',
                               MLPRegressor(hidden_layer_sizes=(3,50), activation='logistic', solver='lbfgs', alpha=alpha,  learning_rate='adaptive',  max_iter=max_iter, tol=tol)))
                models.append((f'mlp_hidden_layer_sizes:(3,26)|activation:logistic|solver:lbfgs|alpha:{alpha}|learning_rate:invscaling|max_iter:{max_iter}|tol:{tol}',
                               MLPRegressor(hidden_layer_sizes=(3,26), activation='logistic', solver='lbfgs', alpha=alpha,  learning_rate='invscaling',  max_iter=max_iter, tol=tol)))
                models.append((f'mlp_hidden_layer_sizes:(3,64)|activation:logistic|solver:lbfgs|alpha:{alpha}|learning_rate:constant|max_iter:{max_iter}|tol:{tol}',
                               MLPRegressor(hidden_layer_sizes=(3,64), activation='logistic', solver='lbfgs', alpha=alpha,  learning_rate='constant',  max_iter=max_iter, tol=tol)))
                models.append((f'mlp_hidden_layer_sizes:(3,76)|activation:logistic|solver:lbfgs|alpha:{alpha}|learning_rate:invscaling|max_iter:{max_iter}|tol:{tol}',
                               MLPRegressor(hidden_layer_sizes=(3,76), activation='logistic', solver='lbfgs', alpha=alpha,  learning_rate='invscaling',  max_iter=max_iter, tol=tol)))
                models.append((f'mlp_hidden_layer_sizes:(3,70)|activation:logistic|solver:lbfgs|alpha:{alpha}|learning_rate:adaptive|max_iter:{max_iter}|tol:{tol}',
                               MLPRegressor(hidden_layer_sizes=(3,70), activation='logistic', solver='lbfgs', alpha=alpha,  learning_rate='adaptive',  max_iter=max_iter, tol=tol)))
                models.append((f'mlp_hidden_layer_sizes:(8,86)|activation:logistic|solver:lbfgs|alpha:{alpha}|learning_rate:adaptive|max_iter:{max_iter}|tol:{tol}',
                               MLPRegressor(hidden_layer_sizes=(8,86), activation='logistic', solver='lbfgs', alpha=alpha,  learning_rate='adaptive',  max_iter=max_iter, tol=tol)))
    return models

def get_best_models_so_far():
    models = []
    
    models.append((f'Ridge_alpha:5.918|fit_intercept:True|normalize:False|copy_X:True|max_iter:None|tol:0.001|solver:auto|random_state:None',
                Ridge(alpha=5.918, 
                        fit_intercept=True, 
                        normalize=False, 
                        copy_X=True, 
                        max_iter=None, 
                        tol=0.001, 
                        solver='auto', 
                        random_state=None)))
                        
    models.append(('LassoLars',
                   LassoLars(alpha=0.1,
                             copy_X=True,
                             fit_intercept=True,
                             fit_path=True,
                             max_iter=500,
                             normalize=True,
                             positive=False,
                             precompute='auto',
                             verbose=False)))
    return models

def get_all():
    models = []
    models.extend(get_linear_regression_models())
    models.extend(get_ridge_models())
    models.extend(get_lasso_models())
    models.extend(get_sgd_regressor())
    models.extend(get_lasso_lars())
    models.extend(get_bayesian_ridge())
    models.extend(get_svr_models())
    models.extend(get_mlp_models())
    return models