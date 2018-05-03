# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		numpy 1.13.1
#		pandas 0.20.3
#		sklearn 0.19.0
#
# -*- author: Hsingmin Lee
#
# gbmc.py -- Gradient Boosting Machine Classifier.
#
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

non_critical_features = ['is_trade', 'item_category_list', 'item_property_list',
                             'predict_category_property', 'instance_id', 'context_id',
                             'realtime', 'context_timestamp']

# Classifier and predictor based on LightGBM. 
def lgbmc(train, validate):

    col = [c for c in train if c not in non_critical_features]
    X_train = train[col]
    y_train = train['is_trade'].values
    X_validate = validate[col]
    y_validate = validate['is_trade'].values

    print("Training lgbmc Model Start  ........................ ")

    # params:
    #       boosting_type           -- algorithm for gradient boosting model
    #       objective               -- task type 'regression'/'binary'/'multiclass' 
    #       num_leaves              -- max leaves for base learner
    #       max_depth               -- max depth for base learner
    #       learning_rate           -- boosting learning rate
    #       colsample_bytree        -- subsample ratio of columns when constructing each tree
    #       subsample               -- subsample ratio of training distance
    #       min_sum_hessian_in_leaf -- minimal sum hessian in one leaf used to deal with over-fitting
    #       n_estimators            -- number of boosted trees to fit
    lgbm_classifier = lgb.LGBMClassifier(boosting_type='gbdt',
                                         objective='binary',
                                         num_leaves=35,
                                         max_depth=8,
                                         learning_rate=0.03,
                                         colsample_bytree=0.8,
                                         subsample=0.9,
                                         min_sum_hessian_in_leaf=100,
                                         n_estimators=20000)
    # params:
    #       X_train                -- input feature matrix
    #       y_train                -- input label matrix
    #       eval_set               -- validate data in tuple type
    #       early_stopping_rounds  -- training rounds for evaluating validate error to activate early-stopping
    # returns:
    #       self object
    #
    lgbm_model = lgbm_classifier.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], early_stopping_rounds=200)
    # class attribute best_iteration_ is the best iteration of fitted model 
    # when early_stopping_rounds parameter specified.   
    best_iter = lgbm_model.best_iteration_
    # Array including all numerical features.
    predictors = [c for c in X_train.columns]
    # Array including feature importances. 
    feature_importance = pd.Series(lgbm_model.feature_importances_, predictors).sort_values(ascending=False)
    print("Output Feature Importance Series as : ")
    print("======================================")
    print(feature_importance)

    # Class method predict_proba(X, raw_score=False, num_iteration=0)
    # params:
    #       X                     -- input feature matrix 
    # returns:
    #       predicted_probability -- predicted probability for each class for each sample
    #                                in shape of [n_samples, n_classes]
    predicted_prob = lgbm_model.predict_proba(validate[col])[:, 1]
    validate['predict'] = predicted_prob
    validate['index'] = range(len(validate))
    print("Evaluate Model : ")
    print('Logistic Loss = ', log_loss(validate['is_trade'], validate['predict']))
    print("The Best Iteration is : ", best_iter)

    return best_iter






